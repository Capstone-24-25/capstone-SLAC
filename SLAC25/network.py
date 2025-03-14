import json
import os
import torch
import sys
import math
import time
import pytz
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torchmetrics import AUROC
from datetime import datetime

from SLAC25.dataset import ImageDataset
from SLAC25.dataloader import DataLoaderFactory
from SLAC25.utils import split_train_val, evaluate_model, EarlyStopping
from SLAC25.models import *

class Wrapper:
    def __init__(self, model, num_epochs=10, outdir='./models', verbose=False, testmode=False):
        self.model = model
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss() # internally computes the softmax so no need for it. 
        self.optimizer = optim.Adam(self.model.parameters())
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=5, min_lr=1e-6)
        self.EarlyStopping = EarlyStopping(patience=7, verbose=False)
        self.outdir = outdir
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # Move model to device
        self.testmode = testmode
        self._prepareDataLoader()

    def _prepareDataLoader(self, batch_size=32, testmode=False, max_imgs=None, nwork=0):
        fileDir = os.path.dirname(os.path.abspath(__file__))
        if fileDir.endswith("SLAC25"):
            trainDataPath = os.path.join(fileDir, "..", "data", "train_info.csv")
            testDataPath = os.path.join(fileDir, "..", "data", "test_info.csv")
            valDataPath = os.path.join(fileDir, "..", "data", "val_info.csv")
            trainDataPath = os.path.abspath(trainDataPath)
            testDataPath = os.path.abspath(testDataPath)
            valDataPath = os.path.abspath(valDataPath)
        
        elif fileDir.endswith("capstone-SLAC"):
            trainDataPath = os.path.join(fileDir, "data", "train_info.csv")
            testDataPath = os.path.join(fileDir, "data", "test_info.csv")
            valDataPath = os.path.join(fileDir, "data", "val_info.csv")
            trainDataPath = os.path.abspath(trainDataPath)
            testDataPath = os.path.abspath(testDataPath)
            valDataPath = os.path.abspath(valDataPath)
        
        trainDataset = ImageDataset(trainDataPath)
        testDataset = ImageDataset(testDataPath)
        valDataset = ImageDataset(valDataPath)

        if testmode: # take only first 50 data
            trainSubDataset = Subset(trainDataset, list(range(50)))
            testSubDataset = Subset(testDataset, list(range(10)))
            valSubDataset = Subset(valDataset, list(range(10)))
            train_factory = DataLoaderFactory(trainSubDataset, batch_size=5)
            test_factory = DataLoaderFactory(testSubDataset, batch_size=5)
            val_factory = DataLoaderFactory(valSubDataset, batch_size=5)
        elif max_imgs is not None:
            ntrain = int(.9*max_imgs)
            ntest=max_imgs-ntrain
            trainSubDataset = Subset(trainDataset, list(range(ntrain)))
            testSubDataset = Subset(testDataset, list(range(ntest)))
            valSubDataset = Subset(valDataset, list(range(ntest)))
            train_factory = DataLoaderFactory(trainSubDataset, batch_size, num_workers=nwork)
            test_factory = DataLoaderFactory(testSubDataset, batch_size, num_workers=nwork)
            val_factory = DataLoaderFactory(valSubDataset, batch_size, num_workers=nwork)
        
        else:
            train_factory = DataLoaderFactory(trainDataset, batch_size)
            test_factory = DataLoaderFactory(testDataPath, batch_size)
            val_factory = DataLoaderFactory(valDataset, batch_size)

        train_factory.setSequentialSampler()
        test_factory.setSequentialSampler()
        val_factory.setSequentialSampler()

        self.train_loader = train_factory.outputDataLoader()
        self.test_loader = test_factory.outputDataLoader()
        self.val_loader = val_factory.outputDataLoader()


class ModelWrapper(Wrapper): # inherits from Wrapper class
    def __init__(self, model_class, num_classes=4, keep_prob=0.75, num_epochs=10, outdir='./models', verbose=False, testmode=False):
        # check first if the model_class is already an instantiated model
        if isinstance(model_class, nn.Module):
            model = model_class
            if verbose:
                print("Using pre-instantiated model. num_classes and keep_prob are ignored.\n")
        else:
            if isinstance(model_class, str):
              if model_class=="BaselineCNN":
                 model_class = BaselineCNN
            model = model_class(num_classes, keep_prob)
            

            #model = model_class(num_classes, keep_prob)
            if verbose:
                print(f"Instantiating new model with num_classes={num_classes} and keep_prob={keep_prob}")
        super().__init__(model, num_epochs, outdir, verbose, testmode)
        
    def summary(self):
        """
        Print a summary of the training setup configuration.

        Displays:
            - Device being used for training
            - Number of epochs and batch size
            - Output directory for saving model files
            - Whether test mode is enabled
            - Whether verbose output is enabled 
            - Whether early stopping is enabled
            - Whether learning rate scheduling is enabled
            - Optimizer and initial learning rate
        """
        print('{:#^70}'.format(''))
        print('Training Setup:\n{} Starting training on {} {}'.format('-'*10, self.device, '-'*10))
        print(f"MODEL: {self.model.__class__.__name__}")
        print(f"EPOCH COUNT: {self.num_epochs}, BATCH SIZE: {self.train_loader.batch_size}")
        print(f"SAVING MODEL TO: {self.outdir}")
        print(f"TEST MODE: {self.testmode}")
        print(f"VERBOSE: {self.verbose}")
        print(f"OPTIMIZER: {self.optimizer.__class__.__name__}, INITIAL LR: {self.optimizer.param_groups[0]['lr']}")
        print(f"EARLY STOPPING: Enabled (patience={self.EarlyStopping.patience})")
        print(f"LEARNING RATE SCHEDULER: Enabled (patience={self.lr_scheduler.patience}, factor={self.lr_scheduler.factor})")
        if self.testmode:
            print("{} NOTE: MODEL IS RUNNING IN TEST MODE {}".format('-'*10, '-'*10))
        print('{:#^70}'.format(''))
    
    def train(self):
        """
        Train the model.
        """
        if self.verbose:
            self.summary()
            print("\n{:=^70}".format(" Training Started "))

        train_log = {
          'epoch': [],
          'train_loss': [],
          'train_acc': [],
          'val_loss': [],
          'val_acc': [],
          'test_loss': [], # added for testing function during each epoch
          'test_acc': [],
          'learning_rates': [],
          'model_checkpoints': [],
          'time_per_epoch': []
        }
        if self.testmode:
            self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=1, min_lr=1e-6) # test if reduce lr works

        # training loop
        for epoch in range(self.num_epochs):
            print("\n{:-^70}".format(f" Epoch {epoch+1}/{self.num_epochs} "))
            print("Training Phase:")
            print(f"{'Batch':>10} {'Loss':>12} {'Accuracy':>12} {'Progress':>12}")
            print("-" * 46)
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            start_time = time.time()
            nbatch = len(self.train_loader)

            #tall = time.time()
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                #tbatch = time.time()

                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad() # zero the gradients
                outputs = self.model(images) # forward pass
                loss = self.criterion(outputs, labels) # compute the loss
                loss.backward() # backpropagation
                self.optimizer.step() # update the weights
                
                if loss is None or math.isnan(loss) or math.isinf(loss):
                    print(f"Error: Loss became undefined or infinite at Epoch: {epoch + 1}/{self.num_epochs} | Batch: {batch_idx + 1}.")
                    print(f"Stopping training.")
                    sys.exit(1)
                
                # Update running stats
                running_loss += loss.item() # extract the tensor and return a float
                _, predicted = outputs.max(1) # gets the class with the highest probability
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item() # if the predicted label equals the actual label, add 1 to the correct
                #tbatch = time.time()-tbatch
                #print("Time batch:", tbatch)
            #tall = time.time()-tall
            #print("Time all:", tall)
                if self.verbose and batch_idx % 10 == 0:
                        progress = f"{batch_idx+1}/{nbatch}"
                        batch_acc = correct / total
                        print(f"{batch_idx+1:>10d} {loss.item():>12.4f} {batch_acc:>12.4f} {progress:>12}")

            # Calculate epoch stats
            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = correct / total
            train_log['epoch'].append(epoch + 1)
            train_log['train_loss'].append(epoch_loss)
            train_log['train_acc'].append(epoch_acc)

            if self.verbose:
                print("\nTraining Epoch Summary:")
                print(f"Average Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")
                print("\n{:-^70}".format(" Validation Phase "))

            # After training phase: Validation phase
            self.model.eval()
            min_val_loss = float('inf')
            val_running_loss = 0.0  # Separate variable for validation
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss = val_running_loss / len(self.val_loader)
            val_acc = val_correct / val_total

            # update the minimum validation loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss

            # update the validation loss and accuracy
            train_log['val_loss'].append(val_loss)
            train_log['val_acc'].append(val_acc)

            if self.verbose:
                print(f"Validation Results:")
                print(f"Average Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
                print("\n{:-^70}".format(" Testing Phase "))

            # After validation phase: Test phase
            test_results = self.test() # make a call to the test function, returns a dictionary
            test_loss = test_results['test_loss']
            test_acc = test_results['test_accuracy']
            train_log['test_loss'].append(test_loss)
            train_log['test_acc'].append(test_acc)

            if self.verbose:
                print(f"Testing Results:")
                print(f"Average Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

            # update the learning rate
            train_log['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            end_time = time.time()
            time_per_epoch = end_time - start_time
            train_log['time_per_epoch'].append(time_per_epoch)

            # save model if the loss is the lowest so far
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                timeNow = datetime.now(pytz.timezone("America/Los_Angeles")).strftime("%Y%m%d%H%M%S")
                model_name = os.path.join(self.outdir, f"ResNet_50_Transfer_Learning_{timeNow}_ep{epoch+1}.net")
                save_file = os.path.abspath(model_name) 
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': epoch_loss,
                        'train_acc': epoch_acc,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    }, save_file)
                    train_log['model_checkpoints'].append(save_file)
                    print("New best model saved to {}".format(save_file))
                except Exception as e:
                    print("Error saving model: {}".format(e))
            
            
            ##### Learning Rate Scheduler #####
            # store the current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            # check if the LR needs to be updated
            self.lr_scheduler.step(val_loss)
            # store the new LR
            new_lr = self.optimizer.param_groups[0]['lr']
            # check if the LR has been updated
            if new_lr != current_lr:
                print("Learning rate updated from {} to {}\n".format(current_lr, new_lr))

            ##### Early Stopping #####
            # check if early stopping is triggered
            if self.EarlyStopping is not None:
                self.EarlyStopping(val_loss, self.model)
                if self.EarlyStopping.early_stop: # flag that is raised or not
                    print("Early stopping")
                    sys.exit(1) # stop training

            # save the train log to a file
            log_file = f"{self.outdir}/train_log.json"
            with open(log_file, "w") as f:
                json.dump(train_log, f, indent=4)

        # once training has finished, print finished
        if self.verbose:
            print("\n{:=^70}".format(" Training Complete "))
        return train_log
    
    def test(self):
        # set model to eval mode so we dont update the weights
        test_loss, test_acc = evaluate_model(self.model, self.test_loader, self.criterion, self.device)
        
        # save as a dictionary
        test_log = {
            'test_loss': test_loss,
            'test_accuracy': test_acc
        }

        # load existing log file...if it exists
        log_file = f"{self.outdir}/train_log.json"
        try:
            with open(log_file, "r") as f:
                full_log = json.load(f)
        except FileNotFoundError:
            full_log = {}
        
        # append the test log to the full log
        full_log['test_log'] = test_log

        # save the full log to a file
        with open(log_file, "w") as f:
            json.dump(full_log, f, indent=4)
            
        return test_log


if __name__ == "__main__":
    ########## parse arguments #########
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument("--nepoch", type=int, default=10)
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--lr", type=float, default=0.001)
    args = ap.parse_args()

    if args.outdir is None:
        
        try:
            slurm_jid = os.environ['SLURM_JOB_ID']
            slurm_jname = os.environ['SLURM_JOB_NAME']
            username = os.environ['USER']
            args.outdir = f"/scratch/slac/models/{username}.{slurm_jname}.{slurm_jid}"
        except KeyError:
            args.outdir = "./models"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # simple model
    model = BaselineCNN(num_classes=4, keep_prob=0.75)
    model.to(device)

    ########## split train set into train and val if validation set does not exist #########
    #split_train_val('./data/train_info.csv', './data/test_info.csv', './data/val_info.csv', seed=42)

    ########## dataset paths #########
    csv_train_file = './data/train_info.csv'
    csv_test_file = './data/test_info.csv'
    csv_val_file = './data/val_info.csv'

    # load the datasets
    train_dataset = ImageDataset(csv_train_file)
    test_dataset = ImageDataset(csv_test_file)
    val_dataset = ImageDataset(csv_val_file)

    train_factory = DataLoaderFactory(train_dataset, num_workers=4)
    train_factory.setSequentialSampler()
    test_factory = DataLoaderFactory(test_dataset, num_workers=4)
    test_factory.setSequentialSampler()
    val_factory = DataLoaderFactory(val_dataset, num_workers=4)
    val_factory.setSequentialSampler()

    train_loader = train_factory.outputDataLoader()
    test_loader = test_factory.outputDataLoader()
    val_loader = val_factory.outputDataLoader()

    criterion = nn.CrossEntropyLoss() # internally computes the softmax so no need for it. 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, min_lr=1e-6)
    
    
    ########## train the model #########
    train_log = fit(model, train_loader, val_loader, num_epochs=args.nepoch, 
                    optimizer=optimizer, criterion=criterion, device=device, 
                    lr_scheduler=lr_scheduler, outdir=args.outdir)
    #test_log = test(model, test_loader, criterion, device) # will save testing for later
