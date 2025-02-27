# experiment1.py
import collections
import gc
import os
import pickle
import sys

import numpy as np
import torch
from tqdm import tqdm
from abc import ABC, abstractmethod

sys.path.append(os.path.abspath(os.path.join('.')))
from helpers import compute_average_logmel
from models import AudioCNN
from classification.train import AdamEarlyStopTrainer
from constants import MODELS

from classification.TUT.tut_dataset import get_datasets_for_fold, get_folds

class BaseExperiment(ABC):
    def __init__(self, batch_size, cuda):
        self.batch_size = batch_size
        self.cuda = cuda
        self.device = torch.device("cuda" if cuda else "cpu")
        self.root_dir = '/Users/nkdem/Downloads/TUT-acoustic-scenes-2017-development' if not cuda else '/disk/scratch/s2203859/minf-1/TUT-acoustic-scenes-2017-development'
        self.tut_dir = '/Users/nkdem/Downloads/TUT-acoustic-scenes-2017-development-16k' if not cuda else '/disk/scratch/s2203859/minf-1/TUT-acoustic-scenes-2017-development-16k'
        self.tut_root_dir = '/Users/nkdem/Downloads/TUT-acoustic-scenes-2017-development' if not cuda else '/disk/scratch/s2203859/minf-1/TUT-acoustic-scenes-2017-development'
        self.folds = get_folds(root_dir=self.root_dir)
    def initialize_result_containers(self):
        return collections.OrderedDict(
            {
            'class_accuracies': {model: [] for model in MODELS.keys()},
            'total_accuracies': {model: [] for model in MODELS.keys()},
            'confusion_matrix_raw': {model: [] for model in MODELS.keys()},
            'trained_models': {model: [] for model in MODELS.keys()}, 
            'losses': {model: [] for model in MODELS.keys()},
            'duration': {model: [] for model in MODELS.keys()},
            'learning_rates': {model: [] for model in MODELS.keys()}
            }
        )
    def collect_model_results(self, test_loader, model, no_classes, env_to_int):
        model.eval()
        total = 0
        correct = 0
        confusion_matrix = np.zeros((no_classes, no_classes))
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", unit="batch"):
                pair, envs, base = batch
                logmels = compute_average_logmel([(pair[0][i], pair[1][i]) for i in range(len(pair[0]))], self.device)
                labels = torch.tensor([env_to_int[env] for env in envs], dtype=torch.long).to(self.device)
                outputs = model(logmels)
                _, predicted = torch.max(outputs.data, 1)
                total += len(predicted)
                correct += (predicted == labels).sum().item()
                for i in range(len(predicted)):
                    confusion_matrix[labels[i], predicted[i]] += 1
        classwise_accuracy = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
        total_accuracy = correct / total
        return classwise_accuracy, total_accuracy, confusion_matrix


class TUTBaselineExperiment(BaseExperiment):
    def __init__(self, num_epochs=1, batch_size=16, 
                 experiment_no=1, learning_rates=None, cuda=False):
        super().__init__(batch_size=batch_size, cuda=cuda)
        self.num_epochs = num_epochs
        self.exp_no = experiment_no
        self.learning_rates = learning_rates
        self.experiment_name = f"full_adam_{num_epochs}epochs_{batch_size}batchsize_TUT"
    
    def create_experiment_dir(self, experiment_name, exp_no):
        base_dir = os.path.join('models', experiment_name, f"exp{exp_no}")
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def run(self):
        # Training phase
        print(f"Starting experiment: {self.experiment_name}")
        print(f"Parameters: epochs={self.num_epochs}, batch_size={self.batch_size}")
        print(f"Learning rates: {self.learning_rates}")

        print(f"\nStarting experiment run {self.exp_no}...")
        base_dir = self.create_experiment_dir(self.experiment_name, self.exp_no)
        for fold in self.folds:
            # create a new fold directory
            os.makedirs(os.path.join(base_dir, f'{fold}'), exist_ok=True)
            print(f"\nStarting fold {fold}...")

            # In your DataLoader:
            train_dataset, test_dataset = get_datasets_for_fold(self.tut_root_dir, self.tut_dir, fold)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            # as a sanity check lets validate that test_loader is disjoint from train_loader
            added = set()
            for batch in tqdm(train_loader, desc="Checking disjointness of train and test loaders", unit="batch"):
                pair, envs, base = batch
                for b in base:
                    added.add(b)
            bases_repeated = set()
            for batch in tqdm(test_loader, desc="Checking disjointness of train and test loaders", unit="batch"):
                pair, envs, base = batch
                for b in base:
                    if b in added:
                        bases_repeated.add(b)
            assert len(bases_repeated) == 0, "Bases are repeated in train and test loaders"

            adam = AdamEarlyStopTrainer(
                cuda=self.cuda,
                base_dir=os.path.join(base_dir, f'{fold}'),
                train_loader=train_loader,
                num_epochs=self.num_epochs,
            )
            adam.train()

            print("\nTraining phase completed. Starting results collection and analysis...")

            # Initialize results containers
            results = self.initialize_result_containers()

            print(f"Collecting results from experiment run {self.exp_no}...")
            for model in MODELS.keys():
                cnn1_channels, cnn2_channels, fc_neurons = MODELS[model]
                cnn = AudioCNN(adam.num_of_classes, cnn1_channels, cnn2_channels, fc_neurons).to(self.device)
                model_path = os.path.join(base_dir, fold, model, 'model.pth')
                cnn.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device)) 
                classwise_accuracy, total_accuracy, confusion_matrix = self.collect_model_results(test_loader=test_loader, model=cnn, no_classes=adam.num_of_classes, env_to_int=adam.env_to_int)
                results['losses'][model] = adam.losses[model]
                results['duration'][model] =adam.durations[model]
                results['learning_rates'][model] = adam.learning_rates[model]
                results['class_accuracies'][model] = classwise_accuracy
                results['total_accuracies'][model] = total_accuracy
                results['confusion_matrix_raw'][model] = confusion_matrix
                results['trained_models'][model] = cnn

            # save results 
            with open(os.path.join(base_dir, f'results-{fold}.pkl'), 'wb') as f:
                pickle.dump(results, f)
            print(f"Results saved in {base_dir}")
            print(results)

    def __str__(self):
        """String representation of the experiment configuration"""
        return (f"Experiment1("
                f"num_epochs={self.num_epochs}, "
                f"batch_size={self.batch_size}, "
                f"learning_rates={self.learning_rates}, "
                f"cuda={self.cuda})")

    def get_experiment_config(self):
        """Return experiment configuration as a dictionary"""
        return {
            "experiment_type": "Experiment1",
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rates": self.learning_rates,
            "cuda": self.cuda,
            "experiment_name": self.experiment_name
        }

if __name__ == '__main__':
    # get command line arg for --experiment_no
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_no", type=int)
    parser.add_argument("--cuda", action='store_true', default=False)
    args = parser.parse_args()

    # if arg is not provided, default to 1
    # but warn 
    if args.experiment_no is None:
        print("No experiment number provided. Defaulting to 1.")
        experiment_no = 1
    else:
        experiment_no = args.experiment_no
    cuda = args.cuda
    # experiment_no = 1
    # cuda = False 
    experiment_no = experiment_no
    experiment = TUTBaselineExperiment(
        num_epochs=240,
        batch_size=32, 
        experiment_no=experiment_no,
        cuda=cuda
    )
    experiment.run()