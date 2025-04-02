import copy
import torchaudio.transforms as T
import gc
import json
import math
import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.optim.sgd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sklearn.utils.class_weight import compute_class_weight

from constants import MODELS
from helpers import compute_average_logmel, get_truly_random_seed_through_os, seed_everything
from models import AudioCNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CachedFeaturesDataset(Dataset):
    def __init__(self, feature_cache, env_to_int):
        self.features = []
        self.labels = []
        self.snrs = [] # the corresponding snr for each feature - 100 if not applicable
        # Group features by environment to ensure we handle speech and non-speech correctly
        grouped_features = {}
        for key, feature in feature_cache.items():
            # key format is either:
            # "env_recsit_cut_snippet_snr" or
            # "SpeechIn_env_recsit_cut_snippet_snr"
            parts = key.split('_')
            if parts[0] == 'SpeechIn':
                env = f"SpeechIn_{parts[1]}"
                recsit, cut, snippet, snr = parts[2:]
            else:
                env, recsit, cut, snippet, snr = parts

            if env not in grouped_features:
                grouped_features[env] = []
            # For non-speech environments, SNR might be None
            snr_val = int(snr) if snr != 'None' else None
            grouped_features[env].append((snr_val, feature))
        
        # For each environment, add features appropriately
        for env, snr_features in grouped_features.items():
            # If it's a speech environment, add one feature per SNR
            if env.startswith('SpeechIn_'):
                # Sort by SNR to ensure consistent ordering
                # Filter out None SNRs just in case
                valid_snr_features = [(s, f) for s, f in snr_features if s is not None]
                valid_snr_features.sort(key=lambda x: x[0])
                for snr, feature in valid_snr_features:
                    self.features.append(feature)
                    self.labels.append(env_to_int[env])
                    self.snrs.append(snr)
            else:
                # For non-speech environments, just add one feature (SNR doesn't matter)
                for snr, feature in snr_features:
                    self.features.append(feature)
                    self.labels.append(env_to_int[env])
                    self.snrs.append(-100)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.snrs[idx]

class BaseTrainer:
    def __init__(self, base_dir: str, num_epochs:int, batch_size:int, train_loader: DataLoader, cuda=False, classes_train=None, augment = True):
        self.augment = augment
        self.base_dir = base_dir
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.cuda = cuda
        self.feature_cache = {}

        self.device = torch.device("mps" if not self.cuda else "cuda")
        logger.info(f"Using device: {self.device}")
        self.seed_val = get_truly_random_seed_through_os()
        seed_everything(self.seed_val)
        logger.info(f"Random seed: {self.seed_val}")

        # iterate over the dataset to get the number of classes
        self.envs = {}
        if classes_train is None:
            logger.info("Counting number of classes...")
            for batch in tqdm(self.train_loader, desc="Counting number of classes", unit="batch"):
                if len(batch) == 3:
                    # tut-ds
                    pair, env, base = batch

                for e, b in zip(env, base):
                    if e not in self.envs:
                        self.envs[e] = 0
                    else:
                        self.envs[e] += 1
        else:
            self.envs = classes_train
        self.num_of_classes = len(self.envs)
        logger.info(f"Number of classes: {self.num_of_classes}")
        self.env_to_int = {env: i for i, env in enumerate(self.envs.keys())}
        class_weights = compute_class_weight('balanced', classes=np.array(list(self.env_to_int.values())), y=np.array([self.env_to_int[env] for env in self.envs.keys() for _ in range(self.envs[env])]))

        # let's use inverse proportionality for the weights
        # total samples / samples per class
        # total_samples = sum(self.envs.values())
        # class_weights = [total_samples / self.envs[env] for env in self.envs.keys()]



        for env, weight in zip(self.env_to_int.keys(), class_weights):
            logger.info(f"Class {env} has weight {weight:.4f}")
        self.weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        
        self.models_to_train = MODELS

        self.metadata = {}
        self.losses = {model: [] for model in self.models_to_train.keys()}
        self.learning_rates_used = {model: [] for model in self.models_to_train.keys()}
        self.durations = {model: [] for model in self.models_to_train.keys()}

        with open(os.path.join(self.base_dir, "int_to_label.txt"), "w") as f:
            for int_label, label in self.env_to_int.items():
                f.write(f"{int_label} {label}\n")
        

        # lets precompute the logmels for the train loader
        self.precompute_logmels()
    
    def set_load_waveforms(self, load_waveforms: bool):
        for dataset in self.train_loader.dataset.datasets:
            if hasattr(dataset, 'load_waveforms'):
                dataset.load_waveforms = load_waveforms
            else:
                method = getattr(dataset, 'set_load_waveforms', None)
                if method is not None and callable(method):
                    method(load_waveforms)
    def set_snr(self, snr: int):
        for dataset in self.train_loader.dataset.datasets:
            if hasattr(dataset, 'snr'):
                dataset.snr = snr
            else:
                method = getattr(dataset, 'set_snr', None)
                if method is not None and callable(method):
                    method(snr)
                continue
                
    def precompute_logmels(self):
        self.snr_levels = [-21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21]
        
        # Always use random_snr approach for precomputation
        for snr in self.snr_levels:
            self.set_snr(snr)
            # counter = 0
            for batch in tqdm(self.train_loader, desc=f"Precomputing logmels for SNR {snr}", unit="batch"):
                if len(batch) == 8:
                    # HEAR-DS
                    waveforms, _ , envs, recsits, cuts, snippets, _, snrs = batch
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}. Expected 8.")
                logmels = self.compute_logmels(waveforms, envs, recsits, cuts, snippets, snrs)
                # counter += 1
                # if counter > 100:
                #     break
         
        self.set_load_waveforms(False)
        self.set_snr(None)
        
        # If not augmenting, create a new dataset with cached features
        if not self.augment:
            cached_dataset = CachedFeaturesDataset(self.feature_cache, self.env_to_int)
            self.train_loader = DataLoader(
                cached_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )

    def save_metadata(self, model, losses, start_time, end_time, learning_rates, extra_meta: dict):
        self.losses[model] = losses
        self.learning_rates_used[model] = learning_rates
        self.durations[model] = end_time - start_time
        self.meta_data = {
            "Seed": self.seed_val,
            "Batch size": self.batch_size,
            "Number of classes": self.num_of_classes,
            "Number of epochs": self.num_epochs,
        }
        self.meta_data.update(extra_meta)

    def prepare_directory(self, model_name):
        model_dir = os.path.join(self.base_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # self.save_file_lists(model_dir)
        return model_dir
    def compute_logmels(self, waveforms, envs, recsits, cuts, snippets, snrs): 
        keys = []
        for env, recsit, cut, snippet, snr in zip(envs, recsits, cuts, snippets, snrs):
            key = f"{env}_{recsit}_{cut}_{snippet}_{snr}"
            keys.append(key)
        
        cached_results = {}
        missing_keys = []
        for i, key in enumerate(keys):
            if key in self.feature_cache:
                cached_results[key] = self.feature_cache[key]
            else:
                missing_keys.append(i)
        
        if missing_keys:
            logger.debug(f"Missing {len(missing_keys)} features. Computing...")
            # missing_audio = [(waveforms[0][i],waveforms[1][i]) for i in missing_keys]         < for TUT 
            missing_audio = [waveforms[i] for i in missing_keys] 

            computed_logmel = compute_average_logmel(missing_audio, self.device)
            for idx, i in enumerate(missing_keys):
                result = computed_logmel[idx].view(1, 40, -1)
                cached_results[keys[i]] = result
                self.feature_cache[keys[i]] = result

        logmels = [cached_results[key] for key in keys]
        return torch.stack(logmels, dim=0)

    def train(self):
        raise NotImplementedError("Subclasses must implement this method!")
    

class FixedLRSGDTrainer(BaseTrainer):
    def __init__(self, base_dir, num_epochs, train_loader: DataLoader, learning_rates: list, change_lr_at_epoch: list, batch_size=32, cuda=False, classes_train=None, augment=True):
        super().__init__(base_dir=base_dir, num_epochs=num_epochs, batch_size=batch_size, cuda=cuda, train_loader=train_loader, classes_train=classes_train, augment=augment)
        self.learning_rates = learning_rates
        self.change_lr_at_epoch = change_lr_at_epoch

    def train(self):
        criterion = nn.CrossEntropyLoss(self.weights)
        for model_name, (cnn1_channels, cnn2_channels, fc_neurons) in self.models_to_train.items():
            losses = []
            learning_rates = []
            model_dir = self.prepare_directory(model_name)

            start_time = time.time()

            model = AudioCNN(self.num_of_classes, cnn1_channels, cnn2_channels, fc_neurons).to(self.device)
            initial_lr = self.learning_rates[0]
            optimiser = torch.optim.SGD(model.parameters(), lr=initial_lr)

            lr_chosen = 0
            for epoch in range(self.num_epochs):
                model.train()
                running_loss = 0.0
                if epoch != 0 and (epoch % self.change_lr_at_epoch) == 0:
                    lr_chosen += 1
                    optimiser = torch.optim.SGD(model.parameters(), lr=self.learning_rates[lr_chosen])

                pbar = tqdm(
                    self.train_loader,
                    desc=f"Training {model_name} [Epoch {epoch + 1}/{self.num_epochs}] [LR: {optimiser.param_groups[0]['lr']}]",
                    unit="batch"
                )
                for batch in pbar:
                    if len(batch) == 3:  # cached features case
                        logmels, actual_labels, snrs = batch
                        logmels = logmels.to(self.device)
                        actual_labels = actual_labels.to(self.device)
                        snrs = snrs.to(self.device)
                    else:  # augment case
                        waveforms, _, envs, recsits, cuts, snippets, _, snrs = batch
                        logmels = self.compute_logmels(waveforms, envs, recsits, cuts, snippets, snrs)
                        actual_labels = torch.tensor([self.env_to_int[env] for env in envs], dtype=torch.long).to(self.device)

                    outputs = model(logmels)
                    loss = criterion(outputs, actual_labels)

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == actual_labels).sum().item()
                    accuracy = 100 * correct / len(actual_labels)

                    # Update running loss and progress bar
                    running_loss += loss.item()
                    avg_loss = running_loss / (pbar.n + 1)  # Current average loss
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{avg_loss:.4f}',
                        'accuracy': f'{accuracy:.2f}%'
                    })

                epoch_loss = running_loss / len(self.train_loader)
                logger.info(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")
                losses.append(epoch_loss)
                learning_rates.append(optimiser.param_groups[0]['lr'])

            end_time = time.time()
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
            extra_meta = {
                "Final loss": epoch_loss,
                "Model parameters": str(model),
                "env_to_int": self.env_to_int,
            }
            self.save_metadata(model_name, losses, start_time, end_time, learning_rates, extra_meta)

            with open(os.path.join(model_dir, "metadata.json"), "w") as f:
                json.dump(self.meta_data, f, indent=4)
            print(f"Model {model_name} saved with fixed LR SGD.")
            torch.mps.empty_cache()
            
        self.feature_cache = {}
        # self.set_load_waveforms(True)
        # self.set_snr(None)



class AdamEarlyStopTrainer(BaseTrainer):
    def __init__(self, base_dir, num_epochs, train_loader: DataLoader, batch_size=32, cuda=False,
                 initial_lr=1e-3,  early_stop_threshold=1e-4, patience=5, classes_train=None, augment=True):
        super().__init__(base_dir=base_dir,num_epochs=num_epochs, batch_size=batch_size, cuda=cuda, train_loader=train_loader, classes_train=classes_train, augment=augment)
        self.initial_lr = initial_lr
        self.early_stop_threshold = early_stop_threshold
        self.patience = patience

    def train(self):
        criterion = nn.CrossEntropyLoss(self.weights)
        for model_name, (cnn1_channels, cnn2_channels, fc_neurons) in self.models_to_train.items():
            model_dir = self.prepare_directory(model_name)

            losses = []
            start_time = time.time()

            model = AudioCNN(self.num_of_classes, cnn1_channels, cnn2_channels, fc_neurons).to(self.device)
            optimiser = torch.optim.Adam(model.parameters(), lr=self.initial_lr)

            best_loss = float('inf')
            epochs_no_improve = 0
            learning_rates = []

            for epoch in range(self.num_epochs):
                model.train()
                running_loss = 0.0

                pbar = tqdm(
                    self.train_loader,
                    desc=f"Training {model_name} [Epoch {epoch + 1}/{self.num_epochs}] [LR: {optimiser.param_groups[0]['lr']}]",
                    unit="batch"
                )
                for batch in pbar:
                    # Determine the dataset type based on batch length
                    if len(batch) == 8:
                        # HEAR-DS
                        waveforms, _ , envs, recsits, cuts, snippets, _, snrs = batch
                    elif len(batch) == 3:
                        # TUT-DS
                        pair, envs, base = batch
                        waveforms = pair  # Assuming pair contains the audio data

                        # base is of the form a026_140_150.wav
                        # let's assume the format is: recsit_CUT_CUT.wav
                        recsits = [None] * len(envs)  # Placeholder for recsits
                        cuts = [None] * len(envs)  # Placeholder for cuts
                        for i, b in enumerate(base):
                            recsit, cut, rest = b.split("_")
                            # rest has CUT.wav, append cut with rest but without the wav
                            cut = f"{cut}_{rest[:-4]}"
                            recsits[i] = recsit
                            cuts[i] = cut
                        snrs = [None] * len(envs)  # Placeholder for snrs
                    else:
                        raise ValueError(f"Unexpected batch length: {len(batch)}. Expected 3 or 6.")

                    logmels = self.compute_logmels(waveforms, envs, recsits, cuts, snippets, snrs)

                    actual_labels = torch.tensor(np.array([self.env_to_int[env] for env in envs]), dtype=torch.long).to(self.device)
                    outputs = model(logmels)
                    loss = criterion(outputs, actual_labels)

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == actual_labels).sum().item()
                    accuracy = 100 * correct / len(actual_labels)

                    # Update running loss and progress bar
                    running_loss += loss.item()
                    avg_loss = running_loss / (pbar.n + 1)  # Current average loss
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'avg_loss': f'{avg_loss:.4f}',
                        'accuracy': f'{accuracy:.2f}%'
                    })

                epoch_loss = running_loss / len(self.train_loader)
                logger.info(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")
                losses.append(epoch_loss)
                learning_rates.append(optimiser.param_groups[0]['lr'])

                if best_loss - epoch_loss > self.early_stop_threshold:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    logger.info(f"Epoch {epoch + 1}: Loss improved to {epoch_loss:.4f}")
                else:
                    epochs_no_improve += 1
                    logger.info(f"Epoch {epoch + 1}: No significant improvement (best loss: {best_loss:.4f}), count: {epochs_no_improve}/{self.patience}")

                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                    break

            end_time = time.time()
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
            extra_meta = {
                "Final loss": epoch_loss,
                "Model parameters": str(model),
                "env_to_int": self.env_to_int,
            }
            self.save_metadata(model_name, losses, start_time, end_time, learning_rates, extra_meta)

            # save metadata to model_dior
            with open(os.path.join(model_dir, "metadata.json"), "w") as f:
                json.dump(self.meta_data, f, indent=4)
            print(f"Model {model_name} saved with Adam optimiser and early stopping.")
            torch.mps.empty_cache()

        self.feature_cache = {}