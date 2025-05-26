import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from SLAC25.dataset import ImageDataset
from SLAC25.utils import evaluate_model
from SLAC25.models import BaselineCNN, ResNet
from SLAC25.sampler import StratifiedSampler, WeightedRandomSampler, EqualGroupSampler, create_sample_weights

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


# Argument Parsing
ap = ArgumentParser()
ap.add_argument("--method", type=str, choices=["original", "stratified", "equal", "weighted"], help="Sampling method")
ap.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
ap.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
ap.add_argument("--batch_size", type=int, default=32, help="Batch size")
ap.add_argument("--model", type=str, choices=["BaselineCNN", "ResNet"], default="BaselineCNN", help="Model type")
ap.add_argument("--outdir", type=str, default="./models", help="Directory to save results")
ap.add_argument("--verbose", action="store_true", help="Enable verbose output")
args = ap.parse_args()

# Data Augmentation (optional, can be used inside ImageDataset or transform pipeline if needed)
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
])

# Load dataset CSV (sampled)
dir_name = os.path.dirname(__file__)
csv_train_file = os.path.join(dir_name, "../../data/train_info.csv")
df = pd.read_csv(csv_train_file)
df_sampled = df.sample(frac=0.05, random_state=42)
sampled_csv_file = os.path.join(dir_name, "../../data/train_info_sampled.csv")
df_sampled.to_csv(sampled_csv_file, index=False)

# Load dataset
print("Loading dataset...")
dataset = ImageDataset(sampled_csv_file)

# Define sampling strategy
sampler = None
if args.method == "stratified":
    sampler = StratifiedSampler(dataset, samplePerGroup=100)
elif args.method == "equal":
    sampler = EqualGroupSampler(dataset, samplePerGroup=100, bootstrap=True)
elif args.method == "weighted":
    weights = create_sample_weights(dataset)
    sampler = WeightedRandomSampler(dataset, weights, total_samples=1000, allowRepeat=True)

# Build DataLoader
if sampler:
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)
else:
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Select Model
model_class = BaselineCNN if args.model == "BaselineCNN" else ResNet
model = model_class(num_classes=4, keep_prob=0.75).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Training
print(f"Training with {args.method} using {args.model} model...")
fit(model, data_loader, args.num_epochs, optimizer, criterion, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Evaluation
print("Evaluating model...")
test_loss, test_acc = evaluate_model(model, data_loader, criterion, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
results = {args.method: {"loss": test_loss, "accuracy": test_acc}}

# Save results
os.makedirs(args.outdir, exist_ok=True)
output_path = os.path.join(args.outdir, f"sampling_results_{args.method}.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

print("Experiment completed! Results saved at:", output_path)