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

import torch
import sys 
import os 
sys.path.append(os.path.abspath(os.path.join('.')))
from helpers import compute_average_logmel, get_truly_random_seed_through_os, seed_everything
from models import CNNSpeechEnhancer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_pytorch_device(prefer_cuda: bool = True):
    """
    Returns an appropriate torch.device object. 
    By default, tries CUDA if available. Otherwise tries MPS (Apple Silicon),
    and falls back to CPU if neither is available.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif not prefer_cuda and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class CachedFeaturesDataset(Dataset):
    def __init__(self, feature_cache):
        self.noisy_features = []
        self.clean_features = []
        
        # Group features by environment to ensure we handle speech and non-speech correctly
        grouped_features = {}
        for key, (noisy, clean) in feature_cache.items():
            # key format is: "env_recsit_cut_snippet_snr"
            parts = key.split('_')
            env, recsit, cut, snippet, snr = parts
            
            if env not in grouped_features:
                grouped_features[env] = []
            # For non-speech environments, SNR might be None
            snr_val = int(snr) if snr != 'None' else None
            grouped_features[env].append((snr_val, noisy, clean))
        
        # For each environment, add features appropriately
        for env, snr_features in grouped_features.items():
            # Sort by SNR to ensure consistent ordering
            # Filter out None SNRs just in case
            valid_snr_features = [(s, n, c) for s, n, c in snr_features if s is not None]
            valid_snr_features.sort(key=lambda x: x[0])
            for _, noisy, clean in valid_snr_features:
                self.noisy_features.append(noisy)
                self.clean_features.append(clean)
    
    def __len__(self):
        return len(self.noisy_features)
    
    def __getitem__(self, idx):
        return self.noisy_features[idx], self.clean_features[idx]

class BaseTrainer:
    def __init__(self, base_dir: str, num_epochs:int, batch_size:int, train_loader: DataLoader, cuda=False):
        self.base_dir = base_dir
        self.train_loader = train_loader
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.cuda = cuda

        self.device = torch.device("mps" if not self.cuda else "cuda")
        logger.info(f"Using device: {self.device}")
        self.seed_val = get_truly_random_seed_through_os()
        seed_everything(self.seed_val)
        logger.info(f"Random seed: {self.seed_val}")

        self.metadata = {}
        self.losses = []
        self.learning_rates = []
        self.duration = 0
        self.feature_cache = {}
        self.noisy_mean = None
        self.noisy_std = None
        self.clean_mean = None
        self.clean_std = None

    def set_snr(self, snr: int):
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'snr'):
            dataset.snr = snr
        else:
            method = getattr(dataset, 'set_snr', None)
            if method is not None and callable(method):
                method(snr)
    def set_load_waveforms(self, load_waveforms: bool):
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'load_waveforms'):
            dataset.load_waveforms = load_waveforms
        else:
            method = getattr(dataset, 'set_load_waveforms', None)
            if method is not None and callable(method):
                method(load_waveforms)
    def precompute_logmels(self, augment=True):
        self.snr_levels = [-21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21]
        # self.snr_levels = [-6]
        
        # Always use random_snr approach for precomputation
        for snr in self.snr_levels:
            self.set_snr(snr)
            for batch in tqdm(self.train_loader, desc=f"Precomputing logmels for SNR {snr}", unit="batch"):
                (noisy_batch, clean_batch, envs, recsits, cuts, snippets, extras, snrs) = batch

                if len(noisy_batch) == 0:
                    continue

                keys = [f"{env}_{recsit}_{cut}_{snippet}_{snr}" for env, recsit, cut, snippet, snr in zip(envs, recsits, cuts, snippets, snrs)]

                for i, key in enumerate(keys):
                    if key not in self.feature_cache:
                        noisy_logmel = compute_average_logmel([noisy_batch[i]], self.device).view(1, 40, -1)
                        clean_logmel = compute_average_logmel([clean_batch[i]], self.device).view(1, 40, -1)
                        self.feature_cache[key] = (noisy_logmel, clean_logmel)
        
        self.set_snr(None)
        self.set_load_waveforms(False)
        self.set_snr(-6)
        
        # If not augmenting, create a new dataset with cached features
        if not augment:
            cached_dataset = CachedFeaturesDataset(self.feature_cache)
            self.train_loader = DataLoader(
                cached_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )

    def compute_mean_and_std(self, num_steps=500):
        """
        Compute the mean and standard deviation of logmels across the dataset using GPU acceleration.
        """
        # Initialize lists to store all mel bands
        all_clean = []
        all_noisy = []
        
        # Collect all mel bands from the feature cache
        for key, (noisy, clean) in self.feature_cache.items():
            # Remove batch dimension and move to device
            clean = clean.squeeze(0).to(self.device)
            noisy = noisy.squeeze(0).to(self.device)
            all_clean.append(clean)
            all_noisy.append(noisy)
        
        # Stack all mel bands into a single tensor
        # Shape: (num_samples, num_mel_bands, time_frames)
        all_clean = torch.stack(all_clean)
        all_noisy = torch.stack(all_noisy)
        
        # Compute means across all samples
        # Shape: (num_mel_bands, time_frames)
        self.clean_mean = all_clean.mean(dim=0)
        self.noisy_mean = all_noisy.mean(dim=0)
        
        # Compute standard deviations
        # Shape: (num_mel_bands, time_frames)
        self.clean_std = all_clean.std(dim=0)
        self.noisy_std = all_noisy.std(dim=0)
        
    def normalize_logmels(self, logmels, is_clean=False):
        """
        Normalize logmels using precomputed mean and std.
        """
        if is_clean:
            mean = self.clean_mean.to(logmels.device)
            std = self.clean_std.to(logmels.device)
        else:
            mean = self.noisy_mean.to(logmels.device)
            std = self.noisy_std.to(logmels.device)

        return (logmels - mean) / std

    def denormalize_logmels(self, normalized_logmels, is_clean=False):
        """
        Denormalize logmels using precomputed mean and std.
        """
        if is_clean:
            mean = self.clean_mean.to(normalized_logmels.device)
            std = self.clean_std.to(normalized_logmels.device)
        else:
            mean = self.noisy_mean.to(normalized_logmels.device)
            std = self.noisy_std.to(normalized_logmels.device)
        
        return (normalized_logmels * std) + mean

    def save_metadata(self, extra_meta: dict):
        self.meta_data = {
            "Seed": self.seed_val,
            "Batch size": self.batch_size,
            "Number of epochs": self.num_epochs,
        }
        self.meta_data.update(extra_meta)
        with open(os.path.join(self.base_dir, "metadata.json"), "w") as f:
            json.dump(self.meta_data, f, indent=4)
        with open(os.path.join(self.base_dir, "losses.json"), "w") as f:
            json.dump(self.losses, f, indent=4)
        with open(os.path.join(self.base_dir, "learning_rates.json"), "w") as f:
            json.dump(self.learning_rates, f, indent=4)
        with open(os.path.join(self.base_dir, "duration.json"), "w") as f:
            json.dump(self.duration, f, indent=4)

    def prepare_directory(self, model_name):
        model_dir = os.path.join(self.base_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        return model_dir

    def train(self):
        raise NotImplementedError("Subclasses must implement this method!")


class SpeechEnhanceAdamEarlyStopTrainer(BaseTrainer):
    def __init__(self, base_dir, num_epochs, train_loader: DataLoader, batch_size=32, cuda=False,
                 initial_lr=1e-3, early_stop_threshold=1e-4, patience=5, augment=True):
        super().__init__(base_dir=base_dir, num_epochs=num_epochs, batch_size=batch_size, cuda=cuda, train_loader=train_loader)
        self.initial_lr = initial_lr
        self.early_stop_threshold = early_stop_threshold
        self.patience = patience
        self.augment = augment

        # Precompute logmels and create cached dataset if not augmenting
        self.precompute_logmels(augment=augment)
        # Compute mean and std for normalization
        self.compute_mean_and_std()

    def train(self):
        criterion = nn.MSELoss()
        start_time = time.time()

        model = CNNSpeechEnhancer().to(self.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.initial_lr)

        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            running_loss = 0.0

            for batch in tqdm(
                self.train_loader,
                desc=f"[Epoch {epoch + 1}/{self.num_epochs}] [LR: {optimiser.param_groups[0]['lr']}]",
                unit="batch"
            ):
                if self.augment:
                    (noisy_batch, clean_batch, envs, recsits, cuts, snippets, extras, snrs) = batch
                    if len(noisy_batch) == 0:
                        continue

                    keys = [f"{env}_{recsit}_{cut}_{snippet}_{snr}" for env, recsit, cut, snippet, snr in zip(envs, recsits, cuts, snippets, snrs)]
                    noisy_logmels = torch.stack([self.feature_cache[key][0] for key in keys])
                    clean_logmels = torch.stack([self.feature_cache[key][1] for key in keys])
                else:
                    noisy_logmels, clean_logmels = batch
                
                # Normalize the logmels
                normalised_noisy_logmels = self.normalize_logmels(noisy_logmels)
                normalised_clean_logmels = self.normalize_logmels(clean_logmels, is_clean=True)

                # Ensure the input to the model has 40 channels
                assert normalised_noisy_logmels.shape[2] == 40, f"Expected 40 channels, but got {normalised_noisy_logmels.shape[1]}"

                # Reshape the input to match the expected 4D format
                # batch_size, channels, time_frames = normalised_noisy_logmels.shape
                # reshaped_input = normalised_noisy_logmels.view(batch_size, channels, 1, time_frames)

                outputs = model(normalised_noisy_logmels)
                loss = criterion(outputs, normalised_clean_logmels)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                running_loss += loss.item()

                # Clear some memory
                del noisy_logmels, clean_logmels, normalised_noisy_logmels, normalised_clean_logmels

            epoch_loss = running_loss / len(self.train_loader)
            logger.info(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")
            self.losses.append(epoch_loss)
            self.learning_rates.append(optimiser.param_groups[0]['lr'])

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
        torch.save(model.state_dict(), os.path.join(self.base_dir, 'model.pth'))
        self.duration = end_time - start_time
        extra_meta = {
            "Final loss": epoch_loss,
        }
        self.save_metadata(extra_meta)