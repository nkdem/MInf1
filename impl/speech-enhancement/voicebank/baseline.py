import argparse
import collections
import csv
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
        self.precompute_logmels()
        self.compute_mean_and_std()

    def precompute_logmels(self):
        """Precompute logmels for training data"""
        logger.info("Precomputing logmels for training data...")
        for batch in tqdm(self.train_loader, desc="Precomputing logmels", unit="batch"):
            (noisy_batch, clean_batch, base_name) = batch
            if len(noisy_batch) == 0:
                continue

            keys = [base_name[i] for i in range(len(base_name))]
            for i, key in enumerate(keys):
                if key not in self.feature_cache:
                    noisy_logmel = compute_average_logmel([noisy_batch[0][i], noisy_batch[1][i]], self.device).view(1, 40, -1)
                    clean_logmel = compute_average_logmel([clean_batch[0][i], clean_batch[1][i]], self.device).view(1, 40, -1)
                    self.feature_cache[key] = (noisy_logmel, clean_logmel)
            torch.mps.empty_cache()
        self.train_loader.dataset.load_waveforms = False
        

    def compute_mean_and_std(self):
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
                noisy_logmels = torch.stack([self.feature_cache[key][0] for key in keys])
                clean_logmels = torch.stack([self.feature_cache[key][1] for key in keys])
                
                # Normalize the logmels
                normalised_noisy_logmels = self.normalize_logmels(noisy_logmels)
                normalised_clean_logmels = self.normalize_logmels(clean_logmels, is_clean=True)

                # Ensure the input to the model has 40 channels
                assert normalised_noisy_logmels.shape[2] == 40, f"Expected 40 channels, but got {normalised_noisy_logmels.shape[1]}"

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

        self.feature_cache = None
        torch.mps.empty_cache()

class VoicebankExperiment():
    def __init__(self, experiment_no: int = 1, batch_size: int = 32, cuda: bool = False):
        self.experiment_no = experiment_no
        self.train_loader, self.test_loader = get_loaders()
        self.batch_size = batch_size
        self.cuda = cuda
        self.device = torch.device("mps" if not cuda else "cuda")
        logger.info("VoicebankExperiment initialized.")
    
    def create_experiment_dir(self, experiment_name, i):
        base_dir = f'models/{experiment_name}/{i}'
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def initialize_result_containers(self):
        return collections.OrderedDict(
            {
            'losses': [],
            'duration': 0,
            'learning_rates': [],
            }
        )
    def precompute_test_logmels(self, test_loader):
        """Precompute logmels for test data"""
        logger.info("Precomputing logmels for test data...")
        self.feature_cache = {}
        
        # Create base directory for saving wav files
        base_dir = os.path.join('models', 'voicebank', str(self.experiment_no))
        wav_dir = os.path.join(base_dir, 'wav_files')
        os.makedirs(wav_dir, exist_ok=True)
        
        # Compute logmels for test data
        for batch in tqdm(test_loader, desc="Precomputing logmels", unit="batch"):
            noisy_batch, clean_batch, base_name = batch
            if len(noisy_batch) == 0:
                continue
                
            noisy_logmels = compute_average_logmel(noisy_batch, self.device, sample_rate=16000, n_fft=1024)
            clean_logmels = compute_average_logmel(clean_batch, self.device, sample_rate=16000, n_fft=1024)

            # Cache the features and save wav files
            for i, (name, noisy_logmel, clean_logmel) in enumerate(zip(base_name, noisy_logmels, clean_logmels)):
                key = name
                
                # Save wav files
                name = name.split('.')[0]
                rec_dir = os.path.join(wav_dir, name)
                os.makedirs(rec_dir, exist_ok=True)
                
                # Save noisy wav files
                # rec_dir of the form 'models/voicebank/1/wav_files/p257_207.wav'
                # name of the form 'p257_207.wav'
                # so we need to strap the .wav first
                noisy_path = os.path.join(rec_dir, f'{name}')
                sample_L = np.array(noisy_batch[0][i])
                sample_R = np.array(noisy_batch[1][i])
                sf.write(f"{noisy_path}_L.wav", sample_L[0], 16000)
                sf.write(f"{noisy_path}_R.wav", sample_R[0], 16000)
                
                # Save clean wav files
                clean_path = os.path.join(rec_dir, f'{name}_clean')
                clean_L = np.array(clean_batch[0][i])
                clean_R = np.array(clean_batch[1][i])
                sf.write(f"{clean_path}_L.wav", clean_L[0], 16000)
                sf.write(f"{clean_path}_R.wav", clean_R[0], 16000)
                
                # Cache the features
                self.feature_cache[key] = {
                    'noisy_logmel': noisy_logmel,
                    'clean_logmel': clean_logmel,
                    'wav_paths': {
                        'noisy': noisy_path,
                        'clean': clean_path
                    }
                }

        # Create a new dataset with cached features
        class CachedFeaturesDataset(torch.utils.data.Dataset):
            def __init__(self, feature_cache):
                self.feature_cache = feature_cache
                self.keys = list(feature_cache.keys())

            def __len__(self):
                return len(self.keys)

            def __getitem__(self, idx):
                key = self.keys[idx]
                data = self.feature_cache[key]
                return data['noisy_logmel'], data['clean_logmel'], key

        cached_dataset = CachedFeaturesDataset(self.feature_cache)
        return DataLoader(
            cached_dataset,
            batch_size=1,
            shuffle=False  # No shuffling for test data
        )

    def run(self):
        logger.info("Running Voicebank Experiment")
        base_dir = self.create_experiment_dir("voicebank", self.experiment_no)
        adam = VoicebankAdamEarlyStopTrainer(
            base_dir=base_dir,
            num_epochs=2,
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

        # Precompute test logmels and create cached test dataset
        cached_test_loader = self.precompute_test_logmels(self.test_loader)
        
        # Use the cached test loader for results collection
        self.test(base_dir=base_dir, test_loader=cached_test_loader, model=cnn, trainer=adam)

    def test(self, base_dir, test_loader: DataLoader, model: CNNSpeechEnhancer, trainer: VoicebankAdamEarlyStopTrainer):
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
                noisy_logmels, clean_logmels, base_names = batch
                if len(noisy_logmels) == 0:
                    continue

                # Normalize the input
                normalised_noisy_logmels = trainer.normalize_logmels(noisy_logmels)
                normalised_noisy_logmels = normalised_noisy_logmels.squeeze(0)


                enhanced = model(normalised_noisy_logmels)


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
                for i, base_name in enumerate(base_names):
                    # Load noisy and clean audio from saved files
                    wav_paths = self.feature_cache[base_name]['wav_paths']
                    noisy_audio = sf.read(f"{wav_paths['noisy']}_L.wav")[0]
                    clean_audio = sf.read(f"{wav_paths['clean']}_L.wav")[0]
                    enhanced_audio = waveform_denormalised[i].cpu().numpy()
                    # enhanced audio might be shorter than clean audio
                    # only use clean and noisy audio until the shortest length
                    min_length = min(len(clean_audio), len(noisy_audio), len(enhanced_audio))
                    clean_audio = clean_audio[:min_length]
                    noisy_audio = noisy_audio[:min_length]
                    enhanced_audio = enhanced_audio[:min_length]

                    before_pesq = pesq(sample_rate, clean_audio, noisy_audio, 'wb')
                    after_pesq = pesq(sample_rate, clean_audio, enhanced_audio, 'wb')
                    before_stoi = stoi(clean_audio, noisy_audio, sample_rate, extended=False)
                    after_stoi = stoi(clean_audio, enhanced_audio, sample_rate, extended=False)

                    scores['before_pesq'][base_name] = before_pesq
                    scores['before_stoi'][base_name] = before_stoi
                    scores['after_pesq'][base_name] = after_pesq
                    scores['after_stoi'][base_name] = after_stoi
                
                # save the waveforms
                os.makedirs(os.path.join(base_dir, 'enhanced'), exist_ok=True)
                for i, base_name in enumerate(base_names):
                    os.makedirs(os.path.join(base_dir, 'enhanced', base_name), exist_ok=True)
                    sf.write(os.path.join(base_dir, 'enhanced', base_name, f'{base_name}_enhanced.wav'), waveform_denormalised[i].cpu().numpy(), sample_rate)
                    sf.write(os.path.join(base_dir, 'enhanced', base_name, f'original.wav'), noisy_audio, sample_rate)
                
            # get average pesq and stoi scores
            before_pesq_avg = np.mean(list(scores['before_pesq'].values()))
            before_stoi_avg = np.mean(list(scores['before_stoi'].values()))
            after_pesq_avg = np.mean(list(scores['after_pesq'].values()))
            after_stoi_avg = np.mean(list(scores['after_stoi'].values()))

            # print table
            print(f"PESQ: {before_pesq_avg:.2f} -> {after_pesq_avg:.2f}")
            print(f"STOI: {before_stoi_avg:.2f} -> {after_stoi_avg:.2f}")

            # save to csv
            with open(os.path.join(base_dir, 'results.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['PESQ', 'STOI'])
                writer.writerow([before_pesq_avg, before_stoi_avg])
                writer.writerow([after_pesq_avg, after_stoi_avg])

                
                
                
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_no", type=int)
    parser.add_argument("--cuda", action='store_true', default=False)
    args = parser.parse_args()

    # if arg is not provided, default to 1
    # but warn 
    if args.experiment_no is None:
        print("No experiment number provided. Defaulting to 1.")
        experiment_no = 1
    else:
        experiment_no = args.experiment_no
    cuda = args.cuda
    experiment = VoicebankExperiment(experiment_no=experiment_no, cuda=cuda)
    experiment.run()