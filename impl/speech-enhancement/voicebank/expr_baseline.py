import argparse
import json
import os
import sys
import time
import numpy as np
from pesq import pesq
from pystoi import stoi
import torch
from tqdm import tqdm
from voicebank_dataset import get_loaders
import logging
import torch.nn as nn
import torch.optim
import torch.optim.sgd
from torch.utils.data import DataLoader
import soundfile as sf
sys.path.append(os.path.abspath(os.path.join('.')))
from helpers import compute_average_logmel, linear_to_waveform, logmel_to_linear
from models import CNNSpeechEnhancer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VoicebankAdamEarlyStopTrainer():
    def __init__(self, base_dir, num_epochs, train_loader: torch.utils.data.DataLoader, batch_size=32, cuda=False,
                 initial_lr=1e-3, early_stop_threshold=1e-4, patience=5):
        self.base_dir = base_dir
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.batch_size = batch_size
        self.cuda = cuda
        self.initial_lr = initial_lr
        self.early_stop_threshold = early_stop_threshold
        self.patience = patience
        self.feature_cache = {}
        self.noisy_mean = None
        self.noisy_std = None
        self.clean_mean = None
        self.clean_std = None
        self.device = torch.device("mps")
        self.losses = []
        self.learning_rates = []
        self.duration = None
        self.compute_mean_and_std()
    def compute_mean_and_std(self, num_steps=1000):
        """
        Compute the mean and standard deviation of logmels across the dataset.
        Also, cache the logmels for later use.
        """
        all_noisy_logmels = []
        all_clean_logmels = []
        for batch in tqdm(self.train_loader, desc="Computing mean and std", unit="batch"):
            (noisy_batch, clean_batch, base_name) = batch


            keys = [base_name[i] for i in range(len(base_name))]

            for i, key in enumerate(keys):
                if key not in self.feature_cache:
                    noisy_logmel = compute_average_logmel([noisy_batch[0][i], noisy_batch[1][i]], self.device).view(1, 40, -1)
                    clean_logmel = compute_average_logmel([clean_batch[0][i], clean_batch[1][i]], self.device).view(1, 40, -1)
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
        if logmels.dim() == 5: # (batch_size, 1, 1, 40, n_frames) 
            logmels = logmels.squeeze(1).squeeze(1)

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


    def save_metadata(self, extra_meta: dict):
        metadata = {
            'losses': self.losses,
            'learning_rates': self.learning_rates,
            'duration': self.duration,
            'extra_meta': extra_meta
        }
        with open(os.path.join(self.base_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)



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
                (noisy_batch, clean_batch, base_name) = batch

                if len(noisy_batch) == 0:
                    continue

                keys = [base_name[i] for i in range(len(base_name))]

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
                    # Build a list of audio pairs from the missing keys.

                    # noisy_missing audio is a tuple of two numpy arrays
                    noisy_missing_audio = (np.array([noisy_batch[0][i] for i in missing_keys]), np.array([noisy_batch[1][i] for i in missing_keys]))
                    noisy_computed_logmel = compute_average_logmel(noisy_missing_audio, self.device)
                    clean_missing_audio = (np.array([clean_batch[0][i] for i in missing_keys]), np.array([clean_batch[1][i] for i in missing_keys]))
                    clean_computed_logmel = compute_average_logmel(clean_missing_audio, self.device)

                    for idx, i in enumerate(missing_keys):
                        noisy = noisy_computed_logmel[idx].view(1, 40, -1)  # Shape: (1, 40, n_frames)
                        cached_noisy[keys[i]] = noisy  
                        clean = clean_computed_logmel[idx].view(1, 40, -1)  # Shape: (1, 40, n_frames)
                        cached_clean[keys[i]] = clean 

                        self.feature_cache[keys[i]] = (noisy, clean)

                noisy_logmels = [cached_noisy[key] for key in keys] # Reorder the results to match the original order
                noisy_logmels = torch.cat(noisy_logmels, dim=0)

                clean_logmels = [cached_clean[key] for key in keys] # Reorder the results to match the original order
                clean_logmels = torch.cat(clean_logmels, dim=0)
                
                # Normalize the logmels
                normalised_noisy_logmels = self.normalize_logmels(noisy_logmels)
                normalised_clean_logmels = self.normalize_logmels(clean_logmels, is_clean=True)

                # Ensure the input to the model has 40 channels
                assert normalised_noisy_logmels.shape[1] == 40, f"Expected 40 channels, but got {normalised_noisy_logmels.shape[1]}"

                # Reshape the input to match the expected 4D format
                batch_size, channels, time_frames = normalised_noisy_logmels.shape
                reshaped_input = normalised_noisy_logmels.view(batch_size, channels, 1, time_frames)  # Add a height dimension of 1

                outputs = model(reshaped_input)
                loss = criterion(outputs, normalised_clean_logmels)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                running_loss += loss.item()

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
class VoicebankExperiment():
    def __init__(self, experiment_no: int):
        self.train_loader, self.test_loader = get_loaders()
        self.batch_size = 32
        self.cuda = False
        self.experiment_no = experiment_no
        self.device = torch.device("mps")   

    def run(self):
        logger.info("Running Voicebank Experiment")
        base_dir = self.create_experiment_dir("voicebank", self.experiment_no)
        adam = VoicebankAdamEarlyStopTrainer(
            base_dir=base_dir,
            num_epochs=40,
            train_loader=self.train_loader,
            batch_size=self.batch_size,
            cuda=self.cuda,
        )

        adam.train()

        logger.info("Training phase completed. Starting results collection and analysis...")

        results = self.initialize_result_containers()
        results['duration'] = adam.duration
        results['learning_rates'] = adam.learning_rates
        results['losses'] = adam.losses

        model_path = os.path.join(base_dir, 'model.pth')
        cnn = CNNSpeechEnhancer().to(self.device)
        cnn.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))

        self.test(base_dir=base_dir, test_loader=self.test_loader, model=cnn, trainer=adam)


    def create_experiment_dir(self, experiment_name, i):
        base_dir = f'models/{experiment_name}/{i}'
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir)
    
    def initialize_result_containers(self):
        return {
            'duration': [],
            'learning_rates': [],
            'losses': []
        }
    
    def test(self, base_dir, test_loader: torch.utils.data.DataLoader, model: CNNSpeechEnhancer, trainer: VoicebankAdamEarlyStopTrainer):
        scores = {
            'before_pesq': {},
            'before_stoi': {},
            'after_pesq': {},
            'after_stoi': {}
        }
        env_scores = {}
        test_files = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", unit="batch"):
                (noisy_batch, clean_batch, base_name) = batch
                if len(noisy_batch) == 0:
                    continue
            
                noisy_computed_logmel = compute_average_logmel(noisy_batch, self.device, sample_rate=16000, n_fft=1024)

                # Normalize the input
                normalised_noisy_logmels = trainer.normalize_logmels(noisy_computed_logmel)

                # Reshape the input to match the expected 4D format (batch_size, channels, height, width)
                batch_size, channels, time_frames = normalised_noisy_logmels.shape
                reshaped_input = normalised_noisy_logmels.view(batch_size, channels, 1, time_frames)

                enhanced = model(reshaped_input)

                sample_rate = 16000
                inverted = logmel_to_linear(logmel_spectrogram=enhanced.squeeze(2), sample_rate=sample_rate, n_fft=1024, n_mels=40, device=torch.device('cpu'))
                hop_length = int(0.02 * sample_rate)
                win_length = int(0.04 * sample_rate)
                waveform = linear_to_waveform(linear_spectrogram_batch=inverted, sample_rate=sample_rate, n_fft=1024, device=self.device, hop_length=hop_length, win_length=win_length)

                # Denormalize the enhanced logmel
                denormalised = trainer.denormalize_logmels(enhanced.squeeze(2), is_clean=True)
                
                inverted_denormalised = logmel_to_linear(logmel_spectrogram=denormalised, sample_rate=sample_rate, n_fft=1024, n_mels=40, device=torch.device('cpu'))
                waveform_denormalised = linear_to_waveform(linear_spectrogram_batch=inverted_denormalised, sample_rate=sample_rate, n_fft=1024, device=self.device, hop_length=hop_length, win_length=win_length)

                # Calculate PESQ and STOI scores for both original and enhanced audio
                for i in range(batch_size):
                    noisy_audio = np.array(noisy_batch[0][i])
                    clean_audio = np.array(clean_batch[0][i])

                    # pad noisy with length of enhanced
                    if len(waveform_denormalised[i]) > len(noisy_audio[0]):
                        noisy_audio = np.pad(noisy_audio[0], (0, abs(len(waveform_denormalised[i]) - len(noisy_audio[0]))), mode='constant')
                        clean_audio = np.pad(clean_audio[0], (0, abs(len(waveform_denormalised[i]) - len(clean_audio[0]))), mode='constant')
                    else:
                        noisy_audio = noisy_audio[0][:len(waveform_denormalised[i])]
                        clean_audio = clean_audio[0][:len(waveform_denormalised[i])]

                    enhanced_audio = waveform_denormalised[i].cpu().numpy()

                    before_pesq = pesq(sample_rate, clean_audio, noisy_audio, 'wb')
                    after_pesq = pesq(sample_rate, clean_audio, enhanced_audio, 'wb')
                    before_stoi = stoi(clean_audio, noisy_audio, sample_rate, extended=False)
                    after_stoi = stoi(clean_audio, enhanced_audio, sample_rate, extended=False)

                    scores['before_pesq'][base_name[i]] = before_pesq
                    scores['before_stoi'][base_name[i]] = before_stoi
                    scores['after_pesq'][base_name[i]] = after_pesq
                    scores['after_stoi'][base_name[i]] = after_stoi
                
                # save the waveforms
                os.makedirs(os.path.join(base_dir, 'enhanced'), exist_ok=True)
                for i in range(batch_size):
                    os.makedirs(os.path.join(base_dir, 'enhanced', base_name[i]), exist_ok=True)
                    sf.write(os.path.join(base_dir, 'enhanced', base_name[i], f'{base_name[i]}_enhanced.wav'), waveform_denormalised[i].cpu().numpy(), sample_rate)
                    sf.write(os.path.join(base_dir, 'enhanced', base_name[i], f'original.wav'), noisy_batch[0][i][0].cpu().numpy(), sample_rate)



                
                
                
                
                
                
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    experiment_no = 1
    args = parser.parse_args()
    experiment = VoicebankExperiment(experiment_no)
    experiment.run()