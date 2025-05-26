# class for learning rate scheduler and early stopping, etc.
import torch
import random
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from models import *

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluation helper function for a model to evaluate on a dataset.
    Handles both standard models and VAE models.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            # For standard classification models
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
            running_loss += loss.item()

    test_loss = running_loss / len(dataloader)
    test_acc = correct / total if total > 0 else 0

    return test_loss, test_acc

def split_train_val(train_csv, test_csv, val_csv_destination: str, seed: int = 42):
    """
    Splits a dataset into a smaller training set and a validation set.
    The validation set size = the test set size
    """

    random.seed(seed)
    np.random.seed(seed)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    test_size = len(test_df)

    # Shuffle data with a fixed random seed
    train_shuffled = train_df.sample(frac=1, random_state=seed).reset_index(drop=True) # random sample of train set (frac = 1 means all data)
    val_df = train_shuffled.iloc[:test_size]

    val_df.to_csv(val_csv_destination, index=False)

    print(f"Validation set size: {len(val_df)}")
    print("Validation Set Sample:")
    print(val_df.head())


class EarlyStopping:
    '''Early stopping if the validation loss does not improve after a given patience'''
    def __init__(self, patience=7, delta=0, path='checkpoint.pt', verbose=False):
        '''
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for saving model.
            verbose (bool): If True, prints a message for each validation loss improvement.
        '''
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0 # counter to store the number of epochs since last improvement
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = torch.inf # initialize best loss to infinity
        
    def __call__(self, val_loss, model):
        score = val_loss  # use validation loss directly
        # if no best loss we set to current loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        # if loss doesn't improve (is greater or equal), we increment the counter by 1
        elif score >= self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping")
        else:  # if loss improves (decreases), we save the model and continue training
            self.best_score = score
            self.save_checkpoint(val_loss, model)  # save the model
            self.counter = 0  # reset the counter
    
    def save_checkpoint(self, val_loss, model):
        '''
        Saves the model when validation loss decreases.
        - Useful when we start to overfit, we can return to the previous best model.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# create a function for visualizing the performance of the model via train log from the json file

def visualize_performance(log_file: str, out_dir: str = None, file_name: str = None, model_name: str = None) -> None:
    '''
    Visualize the performance of the model via train log from the json file
    Args:
        train_log: train log variable from the model wrapper, ModelWrapper.train()
        out_dir: directory to save the plot
        file_name: name of the file to save the plot
    '''

    # Load the train log from the JSON file
    train_log = json.load(open(log_file))

    # variables for plotting
    epochs = train_log['epoch']
    train_acc = train_log.get('train_acc', None)
    val_acc = train_log.get('val_acc', None)
    test_acc = train_log.get('test_acc', None)

    # Check for zero values in accuracy data
    if val_acc is not None and all(v == 0 for v in val_acc):
        print("Warning: Validation accuracy data is all zeros.")
        val_acc = None

    if test_acc is not None and all(v == 0 for v in test_acc):
        print("Warning: Test accuracy data is all zeros.")
        test_acc = None
    
    # Set a clean style with light background
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set global RC parameters for font weights and sizes
    plt.rcParams.update({
        'font.weight': 'bold',
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'font.size': 14
    })
    
    # Create the figure
    fig, ax_acc = plt.subplots(1, 1, figsize=(10, 6), dpi=120)
    
    # Define professional but eye-catching colors
    colors = {
        'train': '#1f77b4',  # blue
        'val': '#ff7f0e',    # orange
        'test': '#2ca02c'    # green
    }
    
    # Plot lines with moderate width and distinct markers
    ax_acc.plot(epochs, train_acc, marker='o', label='Training Accuracy', 
                lw=2, color=colors['train'], markersize=6, alpha=0.9)
    
    if val_acc is not None:
        ax_acc.plot(epochs, val_acc, marker='s', label='Validation Accuracy', 
                    lw=2, color=colors['val'], markersize=6, alpha=0.9)
    
    if test_acc is not None:
        ax_acc.plot(epochs, test_acc, marker='^', label='Test Accuracy', 
                    lw=2, color=colors['test'], markersize=6, alpha=0.9)
        
    # Highlight max accuracy points for validation and test
    if val_acc is not None and test_acc is not None:
        best_val_idx = val_acc.index(max(val_acc))
        best_test_idx = test_acc.index(max(test_acc))

        ax_acc.axvline(x=epochs[best_val_idx], color='red', linestyle='--', alpha=0.7, label='Best Val Epoch')
        ax_acc.axvline(x=epochs[best_test_idx], color='blue', linestyle='--', alpha=0.7, label='Best Test Epoch')

        ax_acc.scatter(epochs[best_val_idx], val_acc[best_val_idx], s=150, 
                       facecolor='orange', edgecolor=colors['val'], linewidth=2, zorder=10, alpha=0.5)
        
        ax_acc.scatter(epochs[best_test_idx], test_acc[best_test_idx], s=150, 
                       facecolor='green', edgecolor=colors['test'], linewidth=2, zorder=10, alpha=0.5)
        
        # Annotation for best test accuracy
        ax_acc.annotate(f"Best Val: {val_acc[best_val_idx] * 100:.2f}%\nEpoch: {epochs[best_val_idx]}", 
                        (epochs[best_val_idx], val_acc[best_val_idx]),
                        xytext=(-125, -100),  # Move text 30 points right and 30 points down
                        textcoords='offset points',
                        fontsize=14, fontweight='bold',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        
        ax_acc.annotate(f"Best Test: {test_acc[best_test_idx] * 100:.2f}%\nEpoch: {epochs[best_test_idx]}", 
                        (epochs[best_test_idx], test_acc[best_test_idx]),
                        xytext=(-160, -115),  # Move text 30 points right and 30 points down
                        textcoords='offset points',
                        fontsize=14, fontweight='bold',
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    
    # Fine-tune grid
    ax_acc.grid(True, linestyle='--', alpha=0.7)
    
    # Style the axes and labels
    ax_acc.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax_acc.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax_acc.tick_params(axis='both', which='major', labelsize=16, labelweight='bold')
    
    # Add a title
    ax_acc.set_title(f'{model_name} Accuracy', fontsize=16, fontweight='bold', pad=15)
    
    # Customize legend
    legend = ax_acc.legend(loc='upper left', frameon=True, fontsize=16)
    frame = legend.get_frame()
    frame.set_alpha(0.8)
    frame.set_edgecolor('lightgray')
    
    # Set y-axis limits with a little padding
    all_values = train_acc[:]
    if val_acc is not None: all_values.extend(val_acc)
    if test_acc is not None: all_values.extend(test_acc)
    ax_acc.set_ylim([min(all_values)*0.95, max(all_values)*1.03])
    
    # Add a subtle border around the plot area
    for spine in ax_acc.spines.values():
        spine.set_visible(True)
        spine.set_color('lightgray')
    
    # Adjust wspace
    plt.subplots_adjust(wspace=0.4)
    
    # Save or display
    if file_name is None:
        plt.show()
    else:
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(os.path.join(out_dir, f"{file_name}.pdf"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(out_dir, f"{file_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance plot saved to {out_dir}/{file_name}")



def visualize_reconstruction(model, dataloader, device, outdir, num_samples=5, save_fig=True, testmode=False):
    '''
    Visualize the reconstruction of the model
    '''
    model.eval()
    
    # get a batch of images
    images, labels = next(iter(dataloader))
    images = images.to(device)

    with torch.no_grad():
        reconstructed, _, _ = model(images)

    # select a subset of images to visualize
    if testmode:
        num_samples = 1

    num_samples = min(num_samples, len(images))
    images = images[:num_samples]
    reconstructed = reconstructed[:num_samples]
    labels = labels[:num_samples]

    # convert to cpu and numpy
    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    # plot the original and reconstructed images
    fig, axs = plt.subplots(2, num_samples, figsize=(num_samples*3, 10))

    # Plot original images on top row
    for i, ax in enumerate(axes[0]):
        img = np.transpose(images[i], (1, 2, 0))  # Change from (C,H,W) to (H,W,C)
        ax.imshow(img)
        ax.set_title(f"Original\nLabel: {labels[i].item()}")
        ax.axis('off')
    
    # Plot reconstructed images on bottom row
    for i, ax in enumerate(axes[1]):
        recon = np.transpose(reconstructed[i], (1, 2, 0))  # Change from (C,H,W) to (H,W,C)
        ax.imshow(recon)
        ax.set_title("Reconstructed")
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save the figure if requested
    if save_fig:
        fig_path = os.path.join(outdir, "vae_reconstructions.png")
        plt.savefig(fig_path)
        print(f"Reconstructions saved to {fig_path}")
    
    return fig

if __name__ == "__main__":
    package_root = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(package_root, "..", "data", "train_info.csv")
    train_path = os.path.abspath(train_path)
    test_path = os.path.join(package_root, "..", "data", "test_info.csv")
    test_path = os.path.abspath(test_path)
    des_path = os.path.join(package_root, "..", "data", "val_info.csv")
    des_path = os.path.abspath(des_path)
    split_train_val(train_path, test_path, des_path, seed=42)
    # print(train_path)
    # print(test_path)
