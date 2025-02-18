# base_experiment.py
import collections
import logging
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
# from pesq import pesq
from torch.utils.data import DataLoader
# from pystoi import stoi
import seaborn as sns
from abc import ABC, abstractmethod

import torch
from tqdm import tqdm

from constants import MODELS
from datasets import BackgroundDataset, MixedAudioDataset, SpeechDataset, split_background_dataset, split_speech_dataset
from helpers import compute_average_logmel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseExperiment(ABC):
    def __init__(self, heards_dir, speech_dir, batch_size=16, cuda=False):
        full_background_ds = BackgroundDataset(heards_dir)
        full_speech_ds = SpeechDataset(speech_dir)
        
        # Split the background dataset based on recsit.
        train_background_ds, test_background_ds, train_background_speech_ds, test_background_speech_ds = split_background_dataset(full_background_ds)
        
        # Split the speech dataset based on speaker.
        train_speech_ds, test_speech_ds = split_speech_dataset(full_speech_ds)
        
        logger.info(f"Train background samples: {len(train_background_ds)}; Test background samples: {len(test_background_ds)}")
        logger.info(f"Train speech samples: {len(train_speech_ds)}; Test speech samples: {len(test_speech_ds)}")
        
        # Create Mixed Audio Datasets for training and test sets.
        train_mixed_ds = MixedAudioDataset(train_background_speech_ds, train_speech_ds)
        test_mixed_ds = MixedAudioDataset(test_background_speech_ds, test_speech_ds)
        
        train_combined = torch.utils.data.ConcatDataset([train_background_ds, train_mixed_ds])
        test_combined = torch.utils.data.ConcatDataset([test_background_ds, test_mixed_ds])

        def collate_fn(batch):
            audios, environments, recsits, cut_ids, extra, snrs= zip(*batch)
            audios = []
            for i in range(len(batch)):
                if snrs[i] is None:
                    waveform_l, sr = sf.read(batch[i][0][0])
                    waveform_r, sr = sf.read(batch[i][0][1]) 
                    audios.append([waveform_l, waveform_r])
                else:
                    audios.append([batch[i][0][0], batch[i][0][1]])
            return audios, environments, recsits, cut_ids, extra, snrs
        self.train_loader = DataLoader(train_combined, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_combined, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.cuda = cuda
        self.device = torch.device('cuda' if cuda else 'mps')
        
    def create_experiment_dir(self, experiment_name, i):
        base_dir = f'models/{experiment_name}/{i}'
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def initialize_result_containers(self):
        # return {
        #     'confusion_matrices': {model: [] for model in MODELS.keys()},
        #     'accuracies': {model: [] for model in MODELS.keys()},
        #     'losses': {model: [] for model in MODELS.keys()},
        #     'training_times': {model: [] for model in MODELS.keys()}
        # }
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

    def collect_experiment_data(self, base_dir):
        training_data = [[] for _ in range(len(MODELS))]
        testing_data = [[] for _ in range(len(MODELS))]

        for j, (model, _) in enumerate(MODELS.items()):
            root_dir = os.path.join(base_dir, model)
            if os.path.exists(root_dir):
                train_data = os.path.join(root_dir, 'train_files.txt')
                test_data = os.path.join(root_dir, 'test_files.txt')
                if os.path.exists(train_data) and os.path.exists(test_data):
                    with open(train_data, 'r') as f:
                        training_data[j] = f.readlines()
                    with open(test_data, 'r') as f:
                        testing_data[j] = f.readlines()
        
        return training_data, testing_data
    
    def collect_model_results(self, test_loader, model, no_classes, env_to_int):
        model.eval()
        total = 0
        correct = 0
        confusion_matrix = np.zeros((no_classes, no_classes))
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", unit="batch"):
                audios, environments, recsits, cut_id, extra, snr = batch
                logmels = compute_average_logmel(audios, self.device)
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
