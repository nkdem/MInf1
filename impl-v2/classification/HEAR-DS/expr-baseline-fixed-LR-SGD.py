import os
import pickle
import sys

import torch
import logging
import os
import numpy as np
import torch

from base_experiment import BaseExperiment
sys.path.append(os.path.abspath(os.path.join('.')))
from models import AudioCNN
from classification.train import FixedLRSGDTrainer
from constants import MODELS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedLR_SGD(BaseExperiment):
    def __init__(self, train_combined, test_combined, learning_rates: list,num_epochs=1, batch_size=16,
                 experiment_no=1, cuda=False):
        super().__init__(batch_size=32, cuda=cuda, 
                         train_combined=train_combined, test_combined=test_combined)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.exp_no = experiment_no
        self.learning_rates = learning_rates
        self.experiment_name = f"fixed-lr-sgd-{experiment_no}"

    def run(self):
        print(f"Starting experiment: {self.experiment_name}")
        print(f"Parameters: epochs={self.num_epochs}, batch_size={self.batch_size}")
        print(f"Learning rates: {self.learning_rates}")

        print(f"\nStarting experiment run {self.exp_no}...")
        base_dir = self.create_experiment_dir(self.experiment_name, self.exp_no)
        trainer = FixedLRSGDTrainer(
            cuda=self.cuda,
            base_dir=base_dir,
            train_loader=self.train_loader,
            num_epochs=self.num_epochs,
            learning_rates=self.learning_rates,
            change_lr_at_epoch=1
        )
        trainer.train()

        print("\nTraining phase completed. Starting results collection and analysis...")

        results = self.get_results(trainer=trainer, base_dir=base_dir)

        # save results 
        with open(os.path.join(base_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved in {base_dir}")

        # print accuracy
        for model in MODELS.keys():
            print(f"\nModel: {model}")
            print(f"Average total accuracy: {np.mean(results['total_accuracies'][model])}")

    def __str__(self):
        """String representation of the experiment configuration"""
        return (f"Experiment1("
                f"num_epochs={self.num_epochs}, "
                f"batch_size={self.batch_size}, "
                f"learning_rates={self.learning_rates}, "
                f"cuda={self.cuda})")

    def get_experiment_config(self):
        return {
            "experiment_type": "Adam Early Stop",
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rates": self.learning_rates,
            "cuda": self.cuda,
            "experiment_name": self.experiment_name
        }

if __name__ == '__main__':
    # get command line arg for --experiment_no
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--experiment_no", type=int)
    # parser.add_argument("--cuda", action='store_true', default=False)
    # args = parser.parse_args()

    # # if arg is not provided, default to 1
    # # but warn 
    # if args.experiment_no is None:
    #     print("No experiment number provided. Defaulting to 1.")
    #     experiment_no = 1
    # cuda = args.cuda
    experiment_no = 1
    cuda = False 
    experiment_no = experiment_no
    split_file = f'splits/split_{experiment_no - 1}.pkl' # 0-indexed
    with open(split_file, 'rb') as f:
        split = pickle.load(f)
        train_combined = split['train']
        test_combined = split['test']
        experiment = FixedLR_SGD(
            train_combined=train_combined,
            test_combined=test_combined,
            num_epochs=6, 
            batch_size=32, 
            experiment_no=experiment_no,
            cuda=cuda,
            learning_rates=[0.05, 0.01, 0.001, 0.0005, 0.0002, 0.0001]
        )
        experiment.run()