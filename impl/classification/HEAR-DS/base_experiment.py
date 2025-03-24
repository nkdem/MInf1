from abc import ABC, abstractmethod
import collections
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import soundfile as sf
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join('.')))
from models import AudioCNN
from classification.train import BaseTrainer
from constants import MODELS
from helpers import compute_average_logmel
class BaseExperiment(ABC):
    def __init__(self, train_combined: ConcatDataset, test_combined: ConcatDataset, batch_size=16, cuda=False):
        def collate_fn(batch):
            audios, clean, environments, recsits, cut_ids, snippets, extra, snrs= zip(*batch)
            audios = []
            for i in range(len(batch)):
                if snrs[i] is None:
                    waveform_l, sr = sf.read(batch[i][0][0])
                    waveform_r, sr = sf.read(batch[i][0][1]) 
                    audios.append([waveform_l, waveform_r])
                else:
                    audios.append([batch[i][0][0], batch[i][0][1]])
            return audios, clean, environments, recsits, cut_ids, snippets, extra, snrs
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

    
    def get_results(self, trainer: BaseTrainer, base_dir: str):
        results = self.initialize_result_containers()
        print(f"Collecting results from experiment run {self.exp_no}...")
        for model in MODELS.keys():
            cnn1_channels, cnn2_channels, fc_neurons = MODELS[model]
            cnn = AudioCNN(trainer.num_of_classes, cnn1_channels, cnn2_channels, fc_neurons).to(self.device)
            model_path = os.path.join(base_dir, model, 'model.pth')
            cnn.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device)) 
            classwise_accuracy, total_accuracy, confusion_matrix = self.collect_model_results(test_loader=self.test_loader, model=cnn, no_classes=trainer.num_of_classes, env_to_int=trainer.env_to_int)
            results['losses'][model].append(trainer.losses[model])
            results['duration'][model].append(trainer.durations[model])
            results['learning_rates'][model].append(trainer.learning_rates_used[model])
            results['class_accuracies'][model].append(classwise_accuracy)
            results['total_accuracies'][model].append(total_accuracy)
            results['confusion_matrix_raw'][model].append(confusion_matrix)
            results['trained_models'][model].append(cnn)
        return results

    def collect_model_results(self, test_loader, model, no_classes, env_to_int):
        model.eval()
        total = 0
        correct = 0
        confusion_matrix = np.zeros((no_classes, no_classes))
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", unit="batch"):
                noisy, clean, environments, recsits, cut_ids, snippets, extra, snrs = batch
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