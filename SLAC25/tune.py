# script to test the best hyperparams with a simple training loop for simplicity

from SLAC25.models import ResNet
from SLAC25.network import ModelWrapper
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune.schedulers import HyperBandScheduler, ASHAScheduler # can use ASHAScheduler with OptunaSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bayesopt import BayesOptSearch

class HyperparameterTuner:
    def __init__(self, use_hyperband=False, use_optuna=False, use_bayesopt=False):
        # Define the search spaces
        self.search_space = {
            "hyperband": {
                "hidden_num": tune.choice([128, 256, 512, 1024]),
                "learning_rate": tune.loguniform(1e-4, 1e-1),
                "keep_prob": tune.uniform(0.5, 0.9),
                "beta1": tune.uniform(0.5, 0.9),
                "beta2": tune.uniform(0.5, 0.9),
            },
            "optuna": {
                "hidden_num": tune.choice([128, 256, 512, 1024]),
                "learning_rate": tune.loguniform(1e-4, 1e-1),
                "keep_prob": tune.uniform(0.5, 0.9),
                "beta1": tune.uniform(0.5, 0.9),
                "beta2": tune.uniform(0.5, 0.9),
            },
            "bayesopt": {
                "hidden_num": tune.uniform(256, 1024),
                "learning_rate": tune.loguniform(1e-4, 1e-1),
                "keep_prob": tune.uniform(0.5, 0.9),
                "beta1": tune.uniform(0.5, 0.9),
                "beta2": tune.uniform(0.5, 0.9),
            }
        }
        '''if use_hyperband:
            self.search_alg = HyperBandScheduler(
                time_attr="training_iteration",
                max_t=10,
                reduction_factor=3,
                metric="loss",
                mode="min"
            )
        elif use_optuna:
            self.search_alg = OptunaSearch(
                metric="loss",
                mode="min"
            )
        elif use_bayesopt:
            self.search_alg = BayesOptSearch(
                metric="loss",
                mode="min"
            )
        else:
            raise ValueError("No search algorithm selected")'''
        self.search_space = self.search_space['optuna']
        self.search_alg = OptunaSearch(
            metric="loss",
            mode="min"
        )
    
    def objective(self, config, checkpoint_dir=None):
        '''
        training loop for finding the best hyperparameters using Ray Tune Hyperband scheduler
        '''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training model with config: ", config)

        # load data and model
        model = ResNet(num_classes=4, keep_prob=config["keep_prob"], hidden_num=config["hidden_num"])
        model.transfer_learn_phase()
        wrapper = ModelWrapper(model_class=model, num_classes=4, keep_prob=config["keep_prob"], verbose=True, testmode=True)        
        train_loader, test_loader, val_loader = wrapper._prepareDataLoader(batch_size=16, testmode=True, max_imgs=5000, nwork=4)
        criterion = nn.CrossEntropyLoss()

        # Define optimizer
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config["learning_rate"], 
            betas=(config["beta1"], config["beta2"])
        )
        
        # Train the model
        print("Training model...")
        for epoch in range(10):
            # training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(val_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_acc = val_correct / val_total
            val_loss = val_loss / len(val_loader)

            tune.report({"loss": val_loss, "mean_accuracy": val_acc})
            print("Epoch {}/10, Loss: {:.4f}, Accuracy: {:.4f}".format(epoch+1, val_loss, val_acc))
    
    def run_tuning(self, num_samples=10, max_epochs=10):
        """
        Run the hyperparameter tuning process
        args:
            num_samples: number of hyperparameter configurations to try
        """
        tuner = tune.Tuner(
            self.objective,
            tune_config=tune.TuneConfig(
                search_alg=self.search_alg,
                num_samples=num_samples
            ),
            param_space=self.search_space,
            run_config=tune.RunConfig(
                name="optuna_test",
                verbose=1,
            )
        )
        results = tuner.fit()
        
        # Get and return the best result
        best_result = results.get_best_result(metric="loss", mode="min")
        return best_result

# Example usage:
if __name__ == "__main__":
    # add in argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_hyperband", type=bool, default=True)
    parser.add_argument("--use_optuna", type=bool, default=False)
    parser.add_argument("--use_bayesopt", type=bool, default=False)
    #parser.add_argument("--num_samples", type=int, default=10)
    #parser.add_argument("--max_epochs", type=int, default=10)
    args = parser.parse_args()

    # run the tuning process
    tuner = HyperparameterTuner(use_hyperband=args.use_hyperband, use_optuna=args.use_optuna, use_bayesopt=args.use_bayesopt)
    best_result = tuner.run_tuning()
    print("Best hyperparameters found were: ", best_result.config)