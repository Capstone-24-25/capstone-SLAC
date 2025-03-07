import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from SLAC25.dataset import ImageDataset
from SLAC25.dataloader import ImageDataLoader
from SLAC25.utils import evaluate_model, compute_auc
#from Models import CNN

def random_undersample(dataset, target_ratio=0.5):
    labels = [dataset[i][1] for i in range(len(dataset))]
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    min_count = int(min(counts) * target_ratio)
    selected_indices = []
    
    for label in unique_labels:
        label_indices = [i for i, lbl in enumerate(labels) if lbl == label]
        selected_indices.extend(random.sample(label_indices, min_count))
    
    return Subset(dataset, selected_indices)

def random_oversample(dataset):
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    unique_labels, counts = np.unique(labels, return_counts=True)

    max_count = max(counts)
    oversampled_indices = []

    for label in unique_labels:
        label_indices = [i for i, lbl in enumerate(labels) if lbl == label]
        oversampled_indices.extend(random.choices(label_indices, k=max_count))

    return Subset(dataset, oversampled_indices)

def augment_minority(dataset, transform):
    augmented_data = []
    labels = [dataset[i][1] for i in range(len(dataset))]
    unique_labels, counts = np.unique(labels, return_counts=True)

    max_count = max(counts)
    class_data = {label: [] for label in unique_labels}

    for i in range(len(dataset)):
        class_data[dataset[i][1]].append(dataset[i])
    
    for label, images in class_data.items():
        while len(images) < max_count:
            img, lbl = random.choice(images)
            img = transform(img)
            images.append((img, lbl))
        augmented_data.extend(images)
    
    return augmented_data


# script starts here:
from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument("method", type=str,  choices=["A","B","C"], help="method name")
args = ap.parse_args()


# change your code to check args.method in order to set the sampling routine

augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
])

csv_train_file = './data/train_info.csv'
dataset = ImageDataset(csv_train_file)

sampling_methods = {
    "original": dataset,
    "random_undersampling": random_undersample(dataset),
    "random_oversampling": random_oversample(dataset),
    "augmented_smote": augment_minority(dataset, augmentation)
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 5
learning_rate = 0.001
batch_size = 32

results = {}

if args.method=="A":
    method_name, sampled_data = "original", dataset
else:
    raise NotImplementedError()
 
#for method_name, sampled_data in sampling_methods.items():
print(f"Training with {method_name}...")
train_loader = DataLoader(sampled_data, batch_size=batch_size, shuffle=True)
    
model = CNN(num_classes=4, keep_prob=0.75).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
fit(model, train_loader, num_epochs, optimizer, criterion, device)
    
auc_score = compute_auc(model, train_loader, device)
results[method_name] = auc_score

with open(f"sampling_results_method{args.method}.json", "w") as f:
    json.dump(results, f, indent=4)

print("Sampling experiment completed!")
