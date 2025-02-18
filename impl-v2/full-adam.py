# experiment1.py
import gc
import os
import pickle

import torch
from base_experiment import BaseExperiment
from models import AudioCNN
from train import AdamEarlyStopTrainer
from constants import MODELS

class FullAdam(BaseExperiment):
    def __init__(self, num_epochs=1, batch_size=16, 
                 experiment_no=1, learning_rates=None, cuda=False):
        self.heards_dir = '/Users/nkdem/Downloads/HEAR-DS' if not cuda else '/disk/scratch/s2203859/minf-1/HEAR-DS'
        self.speech_dir = '/Volumes/SSD/Datasets/CHiME3/CHiME3-Isolated-DEV/dt05_bth' if not cuda else '/disk/scratch/s2203859/minf-1/dt05_bth/'
        super().__init__(heards_dir=self.heards_dir, speech_dir=self.speech_dir, batch_size=32, cuda=cuda)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.exp_no = experiment_no
        self.learning_rates = learning_rates
        # self.experiment_name = f"full_adam_{num_epochs}epochs_{batch_size}batchsize"
        self.experiment_name = f"test"

    def run(self):
        # Training phase
        print(f"Starting experiment: {self.experiment_name}")
        print(f"Parameters: epochs={self.num_epochs}, batch_size={self.batch_size}")
        print(f"Learning rates: {self.learning_rates}")

        print(f"\nStarting experiment run {self.exp_no}...")
        base_dir = self.create_experiment_dir(self.experiment_name, self.exp_no)
        adam = AdamEarlyStopTrainer(
            cuda=self.cuda,
            base_dir=base_dir,
            train_loader=self.train_loader,
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
            model_path = os.path.join(base_dir, model, 'model.pth')
            cnn.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device)) 
            classwise_accuracy, total_accuracy, confusion_matrix = self.collect_model_results(test_loader=self.test_loader, model=cnn, no_classes=adam.num_of_classes, env_to_int=adam.env_to_int)
            results['losses'][model].append(adam.losses[model])
            results['duration'][model].append(adam.durations[model])
            results['learning_rates'][model].append(adam.learning_rates[model])
            results['class_accuracies'][model].append(classwise_accuracy)
            results['total_accuracies'][model].append(total_accuracy)
            results['confusion_matrix_raw'][model].append(confusion_matrix)
            results['trained_models'][model].append(cnn)

        # save results 
        with open(os.path.join(base_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved in {base_dir}")
        print(results)

        with open(os.path.join(base_dir, 'test_files.csv'), 'w') as f:
            for _, env, _, _, base, snr in self.test_loader:
                for e,b in zip(env,base):
                    f.write(f'{b[0]}, {e}{", " + " ".join(b[1]) if b[1] is not None else ""}\n')

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
    experiment = FullAdam(
        num_epochs=1, 
        batch_size=32, 
        experiment_no=experiment_no,
        cuda=cuda
    )
    experiment.run()