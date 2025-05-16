import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from PIL import Image

from SLAC25.utils import evaluate_model
from SLAC25.models import BaselineCNN

print("Libraries Imported")

class newEncoder(nn.Module):
    def __init__(self, encoded_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        )
        self.L = nn.Sequential(nn.ReLU(), nn.Linear(128 * 32, encoded_dim))

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(-1)
        x = x.flatten(start_dim=1)
        x = self.L(x)
        return x

class AutoEncoderTrainingDataset(Dataset):
    def __init__(self, csv_path, device):
        self.df = pd.read_csv(csv_path)
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        img = Image.open(image_path).convert('RGB')
        img = transforms.Resize((512, 512))(img)
        img = transforms.ToTensor()(img).to(self.device)
        return img, torch.tensor(row['label_id'], dtype=torch.long)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=4):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class AutoEncodedDataset(Dataset):
    def __init__(self, csv_path, ae_model, device):
        self.df = pd.read_csv(csv_path)
        self.ae_model = ae_model.eval()
        self.device = device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        img = transforms.Resize((512, 512))(img)
        img = transforms.ToTensor()(img).to(self.device)
        compressed = self.ae_model(img.unsqueeze(0))
        return compressed.view(-1).detach(), torch.tensor(row['label_id'], dtype=torch.long)

ap = ArgumentParser()
ap.add_argument("--sample_frac", type=float, default=0.15)
ap.add_argument("--num_epochs", type=int, default=5)
ap.add_argument("--learning_rate", type=float, default=0.001)
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--encoded_dim", type=int, default=128)
ap.add_argument("--outdir", type=str, default="./models")
ap.add_argument("--verbose", action="store_true")
args = ap.parse_args()

csv_train_file = os.path.join(os.path.dirname(__file__), "../../data/train_info.csv")
df = pd.read_csv(csv_train_file)
from sklearn.utils import resample

df_sampled = df.sample(frac=args.sample_frac, random_state=42)
majority_count = df_sampled['label_id'].value_counts().max()
balanced_df = pd.concat([
    resample(g, replace=True, n_samples=majority_count, random_state=42)
    for _, g in df_sampled.groupby('label_id')
])
balanced_df = balanced_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

sampled_csv_file = os.path.join(os.path.dirname(__file__), "../../data/train_info_sampled.csv")
balanced_df.to_csv(sampled_csv_file, index=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = newEncoder(encoded_dim=args.encoded_dim).to(device)

dataset = AutoEncodedDataset(sampled_csv_file, autoencoder, device)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

model = MLPClassifier(input_dim=args.encoded_dim, num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

def fit(model, dataloader, num_epochs, optimizer, criterion, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}")

print(f"Training classifier with new encoder compression (dim={args.encoded_dim})...")
fit(model, data_loader, args.num_epochs, optimizer, criterion, device)

print("Evaluating model...")
test_loss, test_acc = evaluate_model(model, data_loader, criterion, device)
results = {"nosampling": {"loss": test_loss, "accuracy": test_acc, "compression_dim": args.encoded_dim}}

os.makedirs(args.outdir, exist_ok=True)
output_path = os.path.join(args.outdir, f"sampling_results_nosampling_encoder_dim{args.encoded_dim}.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print("Experiment completed! Results saved at:", output_path)
