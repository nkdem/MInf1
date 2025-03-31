from abc import ABC, abstractmethod
import collections
import logging
import os
import sys
import numpy as np
from sklearn.base import defaultdict
import torch
from torch.utils.data import DataLoader, ConcatDataset
import soundfile as sf
from tqdm import tqdm


sys.path.append(os.path.abspath(os.path.join('.')))
from models import AudioCNN
from classification.train import BaseTrainer, CachedFeaturesDataset
from constants import MODELS
from helpers import compute_average_logmel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class BaseExperiment(ABC):
    def __init__(self, train_combined: ConcatDataset, test_combined: ConcatDataset, batch_size=16, cuda=False):
        def collate_fn(batch):
            audios, clean, environments, recsits, cut_ids, snippets, extra, snrs= zip(*batch)
            if audios[0] is None:
                return None, None, environments, recsits, cut_ids, snippets, extra, snrs
            audios = []
            for i in range(len(batch)):
                if snrs[i] is None:
                    audios.append([batch[i][0][0], batch[i][0][1]])
                else:
                    audios.append([batch[i][0][0], batch[i][0][1]])
            return audios, clean, environments, recsits, cut_ids, snippets, extra, snrs
        self.train_loader = DataLoader(train_combined, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.test_loader = DataLoader(test_combined, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.cuda = cuda
        self.device = torch.device('cuda' if cuda else 'mps')
        self.feature_cache = {}
        
    def create_experiment_dir(self, experiment_name, i):
        base_dir = f'models/{experiment_name}/{i}'
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def initialize_result_containers(self):
        return collections.OrderedDict(
            {
            'trained_models': {model: [] for model in MODELS.keys()}, 
            'losses': {model: [] for model in MODELS.keys()},
            'duration': {model: [] for model in MODELS.keys()},
            'learning_rates': {model: [] for model in MODELS.keys()},
            'naive_class_accuracies': {model: [] for model in MODELS.keys()},
            'naive_total_accuracies': {model: [] for model in MODELS.keys()},
            'naive_confusion_matrix_raw': {model: [] for model in MODELS.keys()},
            # per snr metrics
            'per_snr_metrics': {model: defaultdict(list) for model in MODELS.keys()},
            'overall_metrics': {model: [] for model in MODELS.keys()}
            }
        )

    def validate_data_consistency(self, training_data, testing_data):
        for j in range(len(MODELS) - 1):
            assert training_data[j] == training_data[j + 1], 'Training data is not the same across models'
            assert testing_data[j] == testing_data[j + 1], 'Testing data is not the same across models'

    
    def get_results(self, base_dir: str, test_loader: DataLoader, num_of_classes: int, env_to_int: dict, losses: list, durations: list, learning_rates_used: list):
        results = self.initialize_result_containers()
        print(f"Collecting results from experiment run {self.exp_no}...")
        for model in MODELS.keys():
            cnn1_channels, cnn2_channels, fc_neurons = MODELS[model]
            cnn = AudioCNN(num_of_classes, cnn1_channels, cnn2_channels, fc_neurons).to(self.device)
            model_path = os.path.join(base_dir, model, 'model.pth')
            cnn.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device)) 
            model_results = self.collect_model_results(test_loader=test_loader, model=cnn, no_classes=num_of_classes, env_to_int=env_to_int)
            results['losses'][model].append(losses[model])
            results['duration'][model].append(durations[model])
            results['learning_rates'][model].append(learning_rates_used[model])
            results['naive_class_accuracies'][model].append(model_results['overall']['naive_classwise_accuracy'])
            results['naive_total_accuracies'][model].append(model_results['overall']['naive_total_accuracy'])
            results['naive_confusion_matrix_raw'][model].append(model_results['overall']['confusion_matrix'])
            results['per_snr_metrics'][model].update(model_results['per_snr'])
            results['overall_metrics'][model].append(model_results['overall'])

            results['trained_models'][model].append(cnn)
        return results
    
    # def get_accuracy_for_all_models(self, test_loader, env_to_int, base_dir):
    #     for model in MODELS.keys():
    #         num_of_classes = len(env_to_int)
    #         cnn1_channels, cnn2_channels, fc_neurons = MODELS[model]
    #         cnn = AudioCNN(num_of_classes, cnn1_channels, cnn2_channels, fc_neurons).to(self.device)
    #         model_path = os.path.join(base_dir, model, 'model.pth')
    #         cnn.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device)) 
    #         classwise_accuracy, total_accuracy, confusion_matrix = self.collect_model_results(test_loader=test_loader or self.test_loader, model=cnn, no_classes=num_of_classes, env_to_int=env_to_int)
    #         print(f"Model: {model}")
    #         print(f"Classwise accuracy: {classwise_accuracy}")
    #         print(f"Total accuracy: {total_accuracy}")
    #         print(f"Confusion matrix: {confusion_matrix}")

    def collect_model_results(self, test_loader, model, no_classes, env_to_int):
        model.eval()
        total = 0
        correct = 0
        confusion_matrix = np.zeros((no_classes, no_classes))
        
        # Initialize dictionaries to track per-SNR metrics
        snr_metrics = {}
        snr_confusion_matrices = {}
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", unit="batch"):
                if len(batch) == 3:
                    logmels, labels, snrs = batch
                    labels = labels.to(self.device)
                    snrs = snrs.to(self.device)
                else:
                    noisy, clean, environments, recsits, cut_ids, snippets, extra, snrs = batch
                    logmels = compute_average_logmel(noisy, self.device)
                    labels = torch.tensor([env_to_int[env] for env in environments], dtype=torch.long).to(self.device)
                
                outputs = model(logmels)
                _, predicted = torch.max(outputs.data, 1)
                
                # Update overall metrics
                total += len(predicted)
                correct += (predicted == labels).sum().item()
                for i in range(len(predicted)):
                    confusion_matrix[labels[i], predicted[i]] += 1
                
                # Update per-SNR metrics
                for i, snr in enumerate(snrs):
                    actual_snr = snr.item()
                    if actual_snr not in snr_metrics:
                        snr_metrics[actual_snr] = {'total': 0, 'correct': 0}
                        snr_confusion_matrices[actual_snr] = np.zeros((no_classes, no_classes))
                    
                    snr_metrics[actual_snr]['total'] += 1
                    if predicted[i] == labels[i]:
                        snr_metrics[actual_snr]['correct'] += 1
                    snr_confusion_matrices[actual_snr][labels[i], predicted[i]] += 1
        classwise_accuracy = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
        total_accuracy = correct / total
        
        # Calculate metrics per SNR level
        snr_metrics_detailed = {}
        for snr in snr_confusion_matrices:
            conf_matrix = snr_confusion_matrices[snr]
            
            # Calculate per-class metrics
            class_metrics = {}
            for i in range(no_classes):
                # True Positives
                tp = conf_matrix[i, i]
                # False Positives (sum of column minus diagonal)
                fp = np.sum(conf_matrix[:, i]) - tp
                # False Negatives (sum of row minus diagonal)
                fn = np.sum(conf_matrix[i, :]) - tp

                tn = np.sum(conf_matrix) - tp - fp - fn
                
                # Calculate precision, recall, and F1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics[i] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': (tp + tn) / (tp + fp + fn + tn)
                }
            
            # Calculate macro F1 score for this SNR level
            # for -100 ignore if f1 is 0
            macro_f1 = np.mean([metrics['f1'] for metrics in class_metrics.values()])
            macro_accuracy = np.mean([metrics['accuracy'] for metrics in class_metrics.values()])
            
            snr_metrics_detailed[snr] = {
                'class_metrics': class_metrics,
                'macro_f1': macro_f1,
                'confusion_matrix': conf_matrix,
                'macro_average_accuracy': macro_accuracy
            }
        
        # Calculate overall macro F1 score (average of all SNR macro F1 scores)
        
        return {
            'overall': {
                'macro_average_accuracy': np.mean([metrics['macro_average_accuracy'] for metrics in snr_metrics_detailed.values()]),
                'macro_average_f1': np.mean([metrics['macro_f1'] for metrics in snr_metrics_detailed.values()]),
                'confusion_matrix': confusion_matrix,
                'naive_classwise_accuracy': classwise_accuracy,
                'naive_total_accuracy': total_accuracy
            },
            'per_snr': snr_metrics_detailed
        }
    def precompute_test_logmels(self, test_loader, env_to_int):
        """Precompute logmels for test data and create a CachedFeaturesDataset"""
        logger.info("Precomputing logmels for test data...")
        self.snr_levels = [-21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21]
        
        # Precompute logmels for each SNR level
        for snr in self.snr_levels:
            # Set SNR for the dataset
            for dataset in test_loader.dataset.datasets:
                if hasattr(dataset, 'snr'):
                    dataset.snr = snr
                else:
                    method = getattr(dataset, 'set_snr', None)
                    if method is not None and callable(method):
                        method(snr)

            # Compute logmels for this SNR level
            for batch in tqdm(test_loader, desc=f"Precomputing logmels for SNR {snr}", unit="batch"):
                waveforms, _, envs, recsits, cuts, snippets, _, snrs = batch
                logmels = compute_average_logmel(waveforms, self.device)

                # Cache the features
                for env, recsit, cut, snippet, snr_val, logmel in zip(envs, recsits, cuts, snippets, snrs, logmels):
                    key = f"{env}_{recsit}_{cut}_{snippet}_{snr_val}"
                    self.feature_cache[key] = logmel
            torch.mps.empty_cache()

        # Create a new dataset with cached features
        cached_dataset = CachedFeaturesDataset(self.feature_cache, env_to_int)
        return DataLoader(
            cached_dataset,
            batch_size=self.batch_size,
            shuffle=False  # No shuffling for test data
        )

    @abstractmethod
    def run(self):
        """
        This method should be implemented by each specific experiment class.
        It should contain the main experiment logic.
        """
        pass