import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, json, random, string
import pandas as pd
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from argparse import ArgumentParser
from torch.cuda.amp import autocast, GradScaler
from SLAC25.utils import evaluate_model
from SLAC25.sampler import StratifiedSampler, WeightedRandomSampler, EqualGroupSampler, create_sample_weights

print("Libraries Imported")

def random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

class AutoEncoder(nn.Module):
    def __init__(self, encoded_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, encoded_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 128 * 32 * 32),
            nn.Unflatten(1, (128, 32, 32)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

class AutoEncoderTrainingDataset(Dataset):
    def __init__(self, csv_path, device):
        self.df = pd.read_csv(csv_path)
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        img = self.transform(img)  # stay on CPU
        return img, img

def pretrain_autoencoder(model, dataloader, num_epochs, optimizer, criterion, device, outdir, test_dataloader=None):
    model.train()
    scaler = GradScaler()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
        print(f"[AE Epoch {epoch+1}] Loss: {running_loss / len(dataloader.dataset):.4f}")

        if test_dataloader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
            print(f"[AE Epoch {epoch+1}] Val Loss: {val_loss / len(test_dataloader.dataset):.4f}")
            model.train()

        torch.save(model.state_dict(), os.path.join(outdir, f"autoencoder_epoch{epoch+1}.pt"))

class AutoEncodedDataset(Dataset):
    def __init__(self, csv_path, ae_model, device):
        self.df = pd.read_csv(csv_path)
        self.z_list, self.y_list = [], []
        self.labeldict = {label: self.df[self.df['label_id'] == label].index.tolist()
                          for label in self.df['label_id'].unique()}
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])
        with torch.no_grad():
            ae_model.eval()
            for _, row in self.df.iterrows():
                img = Image.open(row['image_path']).convert('RGB')
                img = transform(img).unsqueeze(0).to(device)
                _, z = ae_model(img)
                self.z_list.append(z.squeeze().cpu())
                self.y_list.append(row['label_id'])

    def __len__(self):
        return len(self.z_list)

    def __getitem__(self, idx):
        return self.z_list[idx], torch.tensor(self.y_list[idx], dtype=torch.long)

class ResNetClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

def fit(model, dataloader, num_epochs, optimizer, criterion, device, outdir):
    model.train()
    scaler = GradScaler()
    for epoch in range(num_epochs):
        total, correct, running_loss = 0, 0, 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f"[Classifier Epoch {epoch+1}] Loss: {running_loss / total:.4f} Acc: {correct / total:.4f}")
        torch.save(model.state_dict(), os.path.join(outdir, f"classifier_epoch{epoch+1}.pt"))

if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--sample_frac", type=float, default=0.15)
    ap.add_argument("--method", type=str, choices=["original", "stratified", "equal", "weighted"])
    ap.add_argument("--num_epochs", type=int, default=5)
    ap.add_argument("--ae_epochs", type=int, default=10)
    ap.add_argument("--learning_rate", type=float, default=0.001)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--encoded_dim", type=int, default=128)
    ap.add_argument("--outdir", type=str, default="./models")
    args = ap.parse_args()

    uid = random_string(10)
    outdir = os.path.join(args.outdir, uid)
    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = os.path.join(os.path.dirname(__file__), "../../data/train_info.csv")
    df = pd.read_csv(csv_path)
    sampled_df = df.sample(frac=args.sample_frac, random_state=42)
    sampled_csv = os.path.join(os.path.dirname(__file__), "../../data/train_info_sampled.csv")
    sampled_df.to_csv(sampled_csv, index=False)

    ae_dataset = AutoEncoderTrainingDataset(sampled_csv, device)
    n = len(ae_dataset)
    train_dset, test_dset = random_split(ae_dataset, [int(n * 0.9), n - int(n * 0.9)])
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    autoencoder = AutoEncoder(args.encoded_dim).to(device)
    ae_optim = optim.Adam(autoencoder.parameters(), lr=args.learning_rate)
    ae_criterion = nn.MSELoss()
    pretrain_autoencoder(autoencoder, train_loader, args.ae_epochs, ae_optim, ae_criterion, device, outdir, test_loader)

    for param in autoencoder.encoder.parameters():
        param.requires_grad = False

    dataset = AutoEncodedDataset(sampled_csv, autoencoder, device)
    if args.method == "stratified":
        sampler = StratifiedSampler(dataset, samplePerGroup=100)
    elif args.method == "equal":
        sampler = EqualGroupSampler(dataset, samplePerGroup=100, bootstrap=True)
    elif args.method == "weighted":
        weights = create_sample_weights(dataset)
        sampler = WeightedRandomSampler(dataset, weights, total_samples=1000, allowRepeat=True)
    else:
        sampler = None

    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=(sampler is None), num_workers=4, pin_memory=True)
    model = ResNetClassifier(input_dim=args.encoded_dim, num_classes=4).to(device)
    optim_clf = optim.Adam(model.parameters(), lr=args.learning_rate)
    clf_criterion = nn.CrossEntropyLoss()
    fit(model, loader, args.num_epochs, optim_clf, clf_criterion, device, outdir)

    test_loss, test_acc = evaluate_model(model, loader, clf_criterion, device)
    results = {args.method: {"loss": test_loss, "accuracy": test_acc, "compression_dim": args.encoded_dim}}
    with open(os.path.join(outdir, f"results_{args.method}_autoenc_dim{args.encoded_dim}.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("✅ Training complete. Results and models saved to:", outdir)

