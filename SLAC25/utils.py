# class for learning rate scheduler and early stopping, etc.
import torch
import random
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt


def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluation helper function for a model to evaluate on a dataset
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(dataloader)
    test_acc = correct / total

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
def visualize_performance(train_log, out_dir: str, file_name: str) -> None:
    '''
    Visualize the performance of the model via train log from the json file
    Args:
        train_log: train log variable from the model wrapper, ModelWrapper.train()
        out_dir: directory to save the plot
        file_name: name of the file to save the plot
    '''

    # Load the train log from the JSON file
    train_log = train_log

    # variables for plotting
    epochs = train_log['epoch']
    train_loss = train_log['train_loss']
    val_loss = train_log['val_loss']
    train_acc = train_log['train_acc']
    val_acc = train_log['val_acc']
    test_loss = train_log['test_loss']
    test_acc = train_log['test_acc']

    # Check if validation data is all zeros
    if all(v == 0 for v in val_loss):
        print("Warning: Validation loss data is all zeros.")
        val_loss = None

    if all(v == 0 for v in val_acc):
        print("Warning: Validation accuracy data is all zeros.")
        val_acc = None

    if all(v == 0 for v in test_loss):
        print("Warning: Test loss data is all zeros.")
        test_loss = None

    if all(v == 0 for v in test_acc):
        print("Warning: Test accuracy data is all zeros.")
        test_acc = None

    # plotting
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(epochs, train_loss, label='Training Loss')
    if val_loss is not None:
        axs[0].plot(epochs, val_loss, label='Validation Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    axs[1].plot(epochs, train_acc, label='Training Accuracy')
    if val_acc is not None:
        axs[1].plot(epochs, val_acc, label='Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    if test_loss is not None:
        axs[0].plot(epochs, test_loss, label='Test Loss')
    if test_acc is not None:
        axs[1].plot(epochs, test_acc, label='Test Accuracy')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir + '/' + file_name)
    plt.close()

    print(f"Performance plot saved to {out_dir}/performance_plot.png")

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
