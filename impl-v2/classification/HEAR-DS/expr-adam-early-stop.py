# experiment1.py
import gc
import os
import pickle
import sys

import torch
import collections
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
from torch.utils.data import DataLoader, ConcatDataset
import seaborn as sns
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join('.')))
from models import AudioCNN
from classification.train import AdamEarlyStopTrainer
from constants import MODELS
from heards_dataset import BackgroundDataset, MixedAudioDataset, SpeechDataset, split_background_dataset, split_speech_dataset
from helpers import compute_average_logmel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseExperiment(ABC):
    def __init__(self, train_combined: ConcatDataset, test_combined: ConcatDataset, batch_size=16, cuda=False):
        def collate_fn(batch):
            audios, clean, environments, recsits, cut_ids, extra, snrs= zip(*batch)
            audios = []
            for i in range(len(batch)):
                if snrs[i] is None:
                    waveform_l, sr = sf.read(batch[i][0][0])
                    waveform_r, sr = sf.read(batch[i][0][1]) 
                    audios.append([waveform_l, waveform_r])
                else:
                    audios.append([batch[i][0][0], batch[i][0][1]])
            return audios, clean, environments, recsits, cut_ids, extra, snrs
        self.train_loader = DataLoader(train_combined, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_combined, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.cuda = cuda
        self.device = torch.device('cuda' if cuda else 'mps')
        
    def create_experiment_dir(self, experiment_name, i):
        base_dir = f'models/{experiment_name}/{i}'
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

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

    def validate_data_consistency(self, training_data, testing_data):
        for j in range(len(MODELS) - 1):
            assert training_data[j] == training_data[j + 1], 'Training data is not the same across models'
            assert testing_data[j] == testing_data[j + 1], 'Testing data is not the same across models'

    def collect_model_results(self, test_loader, model, no_classes, env_to_int):
        model.eval()
        total = 0
        correct = 0
        confusion_matrix = np.zeros((no_classes, no_classes))
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", unit="batch"):
                noisy, clean, environments, recsits, cut_id, extra, snr = batch
                logmels = compute_average_logmel(noisy, self.device)
                labels = torch.tensor([env_to_int[env] for env in environments], dtype=torch.long).to(self.device)
                outputs = model(logmels)
                _, predicted = torch.max(outputs.data, 1)
                total += len(predicted)
                correct += (predicted == labels).sum().item()
                for i in range(len(predicted)):
                    confusion_matrix[labels[i], predicted[i]] += 1
        classwise_accuracy = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
        total_accuracy = correct / total
        return classwise_accuracy, total_accuracy, confusion_matrix

    @abstractmethod
    def run(self):
        """
        This method should be implemented by each specific experiment class.
        It should contain the main experiment logic.
        """
        pass


class FullAdam(BaseExperiment):
    def __init__(self, train_combined, test_combined, num_epochs=1, batch_size=16,
                 experiment_no=1, learning_rates=None, cuda=False):
        super().__init__(batch_size=32, cuda=cuda, 
                         train_combined=train_combined, test_combined=test_combined)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.exp_no = experiment_no
        self.learning_rates = learning_rates
        self.experiment_name = f"test"

    def run(self):
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

        # print accuracy
        for model in MODELS.keys():
            print(f"\nModel: {model}")
            print(f"Average total accuracy: {np.mean(results['total_accuracies'][model])}")

        with open(os.path.join(base_dir, 'test_files.csv'), 'w') as f:
            for _, _, env, _, _, base, snr in self.test_loader:
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
        experiment = FullAdam(
            train_combined=train_combined,
            test_combined=test_combined,
            num_epochs=1, 
            batch_size=32, 
            experiment_no=experiment_no,
            cuda=cuda
        )
        experiment.run()