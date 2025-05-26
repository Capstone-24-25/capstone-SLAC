import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm

from SLAC25.utils import evaluate_model
from SLAC25.sampler import StratifiedSampler, WeightedRandomSampler, EqualGroupSampler, create_sample_weights

# Autoencoder Definition
class AutoEncoder(nn.Module):
    def __init__(self, encoded_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, encoded_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 128 * 32 * 32),
            nn.Unflatten(1, (128, 32, 32)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

# Pretraining function for autoencoder
def pretrain_autoencoder(model, dataloader, num_epochs, optimizer, criterion, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"[AutoEncoder Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}")

# MLP Classifier
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

# Main Execution
if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--method", type=str, choices=["original", "stratified", "equal", "weighted"])
    ap.add_argument("--num_epochs", type=int, default=5)
    ap.add_argument("--ae_epochs", type=int, default=10)
    ap.add_argument("--learning_rate", type=float, default=0.001)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--encoded_dim", type=int, default=128)
    ap.add_argument("--outdir", type=str, default="./models")
    ap.add_argument("--sample_frac", type=float, default=0.05)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir_name = os.path.dirname(__file__)
    csv_train_file = os.path.join(dir_name, "../../data/train_info.csv")
    df = pd.read_csv(csv_train_file)
    df_sampled = df.sample(frac=args.sample_frac, random_state=42).reset_index(drop=True)
    sampled_csv_file = os.path.join(dir_name, "../../data/train_info_sampled.csv")
    df_sampled.to_csv(sampled_csv_file, index=False)

    # Pretrain Autoencoder
    autoencoder = AutoEncoder(encoded_dim=args.encoded_dim).to(device)
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

    class ImageDataset(Dataset):
        def __init__(self, df):
            self.df = df

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img = Image.open(row['image_path']).convert('RGB')
            img = transform(img)
            return img, img

    ae_dataset = ImageDataset(df_sampled)
    ntrain = int(0.9 * len(ae_dataset))
    train_dset, test_dset = torch.utils.data.random_split(ae_dataset, [ntrain, len(ae_dataset)-ntrain])
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)

    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=args.learning_rate)
    ae_criterion = nn.MSELoss()
    pretrain_autoencoder(autoencoder, train_loader, args.ae_epochs, ae_optimizer, ae_criterion, device)

    # Freeze encoder
    for p in autoencoder.encoder.parameters():
        p.requires_grad = False

    # Cache encoded features
    print("Caching encoder outputs...")
    autoencoder.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for idx in tqdm(range(len(df_sampled))):
            row = df_sampled.iloc[idx]
            img = Image.open(row['image_path']).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            _, z = autoencoder(img)
            embeddings.append(z.squeeze(0).cpu())
            labels.append(torch.tensor(row['label_id'], dtype=torch.long))

    X_tensor = torch.stack(embeddings)
    y_tensor = torch.stack(labels)

    cached_dir = os.path.join(args.outdir, "cached_features")
    os.makedirs(cached_dir, exist_ok=True)
    torch.save(X_tensor, os.path.join(cached_dir, "X_tensor.pt"))
    torch.save(y_tensor, os.path.join(cached_dir, "y_tensor.pt"))
    print("✅ Cached features saved.")

    # Load cached data
    X_tensor = torch.load(os.path.join(cached_dir, "X_tensor.pt"))
    y_tensor = torch.load(os.path.join(cached_dir, "y_tensor.pt"))
    dataset = TensorDataset(X_tensor, y_tensor)

    # Apply sampler
    sampler = None
    if args.method == "stratified":
        sampler = StratifiedSampler(dataset, samplePerGroup=100)
    elif args.method == "equal":
        sampler = EqualGroupSampler(dataset, samplePerGroup=100, bootstrap=True)
    elif args.method == "weighted":
        weights = create_sample_weights(dataset)
        sampler = WeightedRandomSampler(dataset, weights, total_samples=1000, allowRepeat=True)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler) if sampler else \
                  DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Train classifier
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
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f"[Classifier Epoch {epoch+1}] Loss: {running_loss/total:.4f} Accuracy: {correct/total:.4f}")

    fit(model, data_loader, args.num_epochs, optimizer, criterion, device)

    # Evaluate
    test_loss, test_acc = evaluate_model(model, data_loader, criterion, device)
    results = {args.method: {"loss": test_loss, "accuracy": test_acc, "compression_dim": args.encoded_dim}}

    os.makedirs(args.outdir, exist_ok=True)
    output_path = os.path.join(args.outdir, f"sampling_results_{args.method}_autoenc_dim{args.encoded_dim}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("Experiment completed! Results saved at:", output_path)
