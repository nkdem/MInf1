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
from torch.utils.data import DataLoader
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

        # iterate over the dataset to get the number of classes
        # with open(os.path.join(self.base_dir, "train_files.csv"), "w") as f:
        #     for (noisy_batch, clean_batch, envs, recsits, cuts, extras, snrs) in tqdm(self.train_loader):
        #         for base, speech_used in extras:
        #             f.write(f"{base},{speech_used}\n")
        self.metadata = {}
        self.losses = []
        self.learning_rates = []
        self.duration = 0
        

    def save_metadata(self, extra_meta: dict):
        self.meta_data = {
            "Seed": self.seed_val,
            "Batch size": self.batch_size,
            "Number of epochs": self.num_epochs,
        }
        self.meta_data.update(extra_meta)
        with open(os.path.join(self.base_dir, "metadata.json"), "w") as f:
            json.dump(self.meta_data, f, indent=4)
        #  save losses
        with open(os.path.join(self.base_dir, "losses.json"), "w") as f:
            json.dump(self.losses, f, indent=4)
        # save learning rates
        with open(os.path.join(self.base_dir, "learning_rates.json"), "w") as f:
            json.dump(self.learning_rates, f, indent=4)
        # save duration
        with open(os.path.join(self.base_dir, "duration.json"), "w") as f:
            json.dump(self.duration, f, indent=4)

    def prepare_directory(self, model_name):
        model_dir = os.path.join(self.base_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # self.save_file_lists(model_dir)
        return model_dir

    def train(self):
        raise NotImplementedError("Subclasses must implement this method!")


class SpeechEnhanceAdamEarlyStopTrainer(BaseTrainer):
    def __init__(self, base_dir, num_epochs, train_loader: DataLoader, batch_size=32, cuda=False,
                 initial_lr=1e-3, early_stop_threshold=1e-4, patience=5):
        super().__init__(base_dir=base_dir, num_epochs=num_epochs, batch_size=batch_size, cuda=cuda, train_loader=train_loader)
        self.initial_lr = initial_lr
        self.early_stop_threshold = early_stop_threshold
        self.patience = patience
        self.feature_cache = {}
        self.noisy_mean = None
        self.noisy_std = None
        self.clean_mean = None
        self.clean_std = None
        self.compute_mean_and_std()

    def compute_mean_and_std(self, num_steps=500):
        """
        Compute the mean and standard deviation of logmels across the dataset.
        Also, cache the logmels for later use.
        """
        all_noisy_logmels = []
        all_clean_logmels = []
        for batch in tqdm(self.train_loader, desc="Computing mean and std", unit="batch"):
            (noisy_batch, clean_batch, envs, recsits, cuts, snippets, extras, snrs) = batch

            if len(noisy_batch) == 0:
                continue

            keys = [f"{env}_{recsit}_{cut}_{snippet}_{snr}" for env, recsit, cut, snippet, snr in zip(envs, recsits, cuts, snippets, snrs)]

            for i, key in enumerate(keys):
                if key not in self.feature_cache:
                    noisy_logmel = compute_average_logmel([noisy_batch[i]], self.device).view(1, 40, -1)
                    clean_logmel = compute_average_logmel([clean_batch[i]], self.device).view(1, 40, -1)
                    self.feature_cache[key] = (noisy_logmel, clean_logmel)
                    all_noisy_logmels.append(noisy_logmel.squeeze(0))
                    all_clean_logmels.append(clean_logmel.squeeze(0))

            if len(all_noisy_logmels) >= num_steps * self.batch_size:
                break

        all_noisy_logmels = torch.cat(all_noisy_logmels, dim=1)
        all_clean_logmels = torch.cat(all_clean_logmels, dim=1)
        
        self.noisy_mean = all_noisy_logmels.mean(dim=1)
        self.noisy_std = all_noisy_logmels.std(dim=1)
        self.clean_mean = all_clean_logmels.mean(dim=1)
        self.clean_std = all_clean_logmels.std(dim=1)

    def normalize_logmels(self, logmels, is_clean=False):
        """
        Normalize logmels using precomputed mean and std.
        This function works for both single samples and batches.
        
        Args:
            logmels (torch.Tensor): Input logmel spectrogram(s). 
                                    Shape: (batch_size, 1, 40, n_frames), (batch_size, 40, n_frames), or (40, n_frames)
            is_clean (bool): Whether the input is clean or noisy logmels
        
        Returns:
            torch.Tensor: Normalized logmel spectrogram(s)
        """
        if is_clean:
            mean = self.clean_mean.to(logmels.device)
            std = self.clean_std.to(logmels.device)
        else:
            mean = self.noisy_mean.to(logmels.device)
            std = self.noisy_std.to(logmels.device)
        
        # Check if logmels is a 4D tensor
        if logmels.dim() == 4:  # (batch_size, 1, 40, n_frames)
            # Remove the extra dimension
            logmels = logmels.squeeze(1)
        
        # Check if logmels is a batch
        if logmels.dim() == 3:  # (batch_size, 40, n_frames)
            # Expand mean and std to match the batch dimension
            mean = mean.unsqueeze(0).unsqueeze(2)
            std = std.unsqueeze(0).unsqueeze(2)
        elif logmels.dim() == 2:  # (40, n_frames)
            mean = mean.unsqueeze(1)
            std = std.unsqueeze(1)
        else:
            raise ValueError(f"Expected logmels to have 2, 3, or 4 dimensions, but got {logmels.dim()}")

        return (logmels - mean) / std

    def denormalize_logmels(self, normalized_logmels, is_clean=False):
        """
        Denormalize logmels using precomputed mean and std.
        This function works for both single samples and batches.
        
        Args:
            normalized_logmels (torch.Tensor): Input normalized logmel spectrogram(s). 
                                               Shape: (batch_size, 1, 40, n_frames), (batch_size, 40, n_frames), or (40, n_frames)
            is_clean (bool): Whether the input is clean or noisy logmels
        
        Returns:
            torch.Tensor: Denormalized logmel spectrogram(s)
        """
        if is_clean:
            mean = self.clean_mean.to(normalized_logmels.device)
            std = self.clean_std.to(normalized_logmels.device)
        else:
            mean = self.noisy_mean.to(normalized_logmels.device)
            std = self.noisy_std.to(normalized_logmels.device)
        
        # Check if normalized_logmels is a 4D tensor
        if normalized_logmels.dim() == 4:  # (batch_size, 1, 40, n_frames)
            # Remove the extra dimension
            normalized_logmels = normalized_logmels.squeeze(1)
        
        # Check if normalized_logmels is a batch
        if normalized_logmels.dim() == 3:  # (batch_size, 40, n_frames)
            # Expand mean and std to match the batch dimension
            mean = mean.unsqueeze(0).unsqueeze(2)
            std = std.unsqueeze(0).unsqueeze(2)
        elif normalized_logmels.dim() == 2:  # (40, n_frames)
            mean = mean.unsqueeze(1)
            std = std.unsqueeze(1)
        else:
            raise ValueError(f"Expected normalized_logmels to have 2, 3, or 4 dimensions, but got {normalized_logmels.dim()}")

        return (normalized_logmels * std) + mean

    def compute_logmels(self, noisy_waveforms, clean_waveforms, envs, recsits, cuts, snippets, snrs):
        """
        Compute logmels for both noisy and clean waveforms, with caching.
        """
        keys = [f"{env}_{recsit}_{cut}_{snippet}_{snr}" for env, recsit, cut, snippet, snr in zip(envs, recsits, cuts, snippets, snrs)]
        
        cached_noisy = {}
        cached_clean = {}
        missing_keys = []
        
        for i, key in enumerate(keys):
            if key in self.feature_cache:
                cached_noisy[key] = self.feature_cache[key][0]
                cached_clean[key] = self.feature_cache[key][1]
            else:
                missing_keys.append(i)
        
        if missing_keys:
            logger.debug(f"Missing {len(missing_keys)} features. Computing...")
            # Build lists of audio pairs from the missing keys
            noisy_missing_audio = [noisy_waveforms[i] for i in missing_keys]
            clean_missing_audio = [clean_waveforms[i] for i in missing_keys]
            
            # Compute logmels for missing audio
            noisy_computed_logmel = compute_average_logmel(noisy_missing_audio, self.device)
            clean_computed_logmel = compute_average_logmel(clean_missing_audio, self.device)
            
            for idx, i in enumerate(missing_keys):
                noisy = noisy_computed_logmel[idx].view(1, 40, -1)
                clean = clean_computed_logmel[idx].view(1, 40, -1)
                cached_noisy[keys[i]] = noisy
                cached_clean[keys[i]] = clean
                self.feature_cache[keys[i]] = (noisy, clean)
        
        # Reorder results to match original order
        noisy_logmels = [cached_noisy[key] for key in keys]
        clean_logmels = [cached_clean[key] for key in keys]
        
        return torch.cat(noisy_logmels, dim=0), torch.cat(clean_logmels, dim=0)

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
                (noisy_batch, clean_batch, envs, recsits, cuts, snippets, extras, snrs) = batch

                if len(noisy_batch) == 0:
                    continue

                # Compute logmels using the new method
                noisy_logmels, clean_logmels = self.compute_logmels(
                    noisy_batch, clean_batch, envs, recsits, cuts, snippets, snrs
                )
                
                # Normalize the logmels
                normalised_noisy_logmels = self.normalize_logmels(noisy_logmels)
                normalised_clean_logmels = self.normalize_logmels(clean_logmels, is_clean=True)

                # Ensure the input to the model has 40 channels
                assert normalised_noisy_logmels.shape[1] == 40, f"Expected 40 channels, but got {normalised_noisy_logmels.shape[1]}"

                # Reshape the input to match the expected 4D format
                batch_size, channels, time_frames = normalised_noisy_logmels.shape
                reshaped_input = normalised_noisy_logmels.view(batch_size, channels, 1, time_frames)

                outputs = model(reshaped_input)
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