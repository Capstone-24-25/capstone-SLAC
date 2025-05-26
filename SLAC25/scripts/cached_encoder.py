import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os, json
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
from argparse import ArgumentParser

# --------- Model Definitions ---------
class newEncoder(nn.Module):
    def __init__(self, encoded_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        )
        self.L = nn.Sequential(nn.ReLU(), nn.Linear(128 * 32, encoded_dim))

    def forward(self, x):
        x = self.encoder(x)
        x = x.mean(-1)
        x = x.flatten(start_dim=1)
        x = self.L(x)
        return x

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# --------- Dataset to Load Images ---------
class ImageDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label_id"])
        img = self.transform(img)
        return img, label

# --------- Train and Evaluate ---------
def fit(model, dataloader, num_epochs, optimizer, criterion, device):
    model.train()
    for epoch in range(num_epochs):
        total, correct, loss_total = 0, 0, 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_total += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss_total/total:.4f} Accuracy: {correct/total:.4f}")

def evaluate(model, dataloader, criterion, device):
    model.eval()
    loss_total, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_total += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return loss_total / total, correct / total

# --------- Main ---------
if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--sample_frac", type=float, default=0.1)
    ap.add_argument("--num_epochs", type=int, default=10)
    ap.add_argument("--learning_rate", type=float, default=0.001)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--encoded_dim", type=int, default=128)
    ap.add_argument("--outdir", type=str, default="./models")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # --------- Load and Sample CSV ---------
    data_path = os.path.join(os.path.dirname(__file__), "../../data/train_info.csv")
    df = pd.read_csv(data_path).sample(frac=args.sample_frac, random_state=42).reset_index(drop=True)
    sampled_csv = os.path.join(os.path.dirname(__file__), "../../data/train_info_sampled.csv")
    df.to_csv(sampled_csv, index=False)

    # --------- Image Transform ---------
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # --------- Precompute Features ---------
    encoder = newEncoder(encoded_dim=args.encoded_dim).to(device)
    encoder.eval()

    image_dataset = ImageDataset(sampled_csv, transform)
    image_loader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    X, y = [], []
    with torch.no_grad():
        for images, labels in image_loader:
            images = images.to(device)
            embeddings = encoder(images).cpu()
            X.append(embeddings)
            y.append(labels)

    X_tensor = torch.cat(X, dim=0)
    y_tensor = torch.cat(y, dim=0)

    # --------- Train Classifier ---------
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    model = MLPClassifier(input_dim=args.encoded_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    fit(model, loader, args.num_epochs, optimizer, criterion, device)

    # --------- Evaluate and Save ---------
    test_loss, test_acc = evaluate(model, loader, criterion, device)
    results = {
        "nosampling_cached": {
            "loss": test_loss,
            "accuracy": test_acc,
            "compression_dim": args.encoded_dim
        }
    }

    outpath = os.path.join(args.outdir, f"sampling_results_nosampling_cached_encoder_dim{args.encoded_dim}.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Training complete. Results saved to:", outpath)
