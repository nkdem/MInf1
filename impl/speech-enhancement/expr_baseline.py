import collections
import csv
import logging
import os
import torch
import numpy as np
import pickle
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import soundfile as sf
import sys 
from pesq import pesq
from pystoi import stoi
from train_enhance import SpeechEnhanceAdamEarlyStopTrainer
sys.path.append(os.path.abspath(os.path.join('.')))
from heards_dataset import BackgroundDataset, MixedAudioDataset, SpeechDataset, split_background_dataset, split_speech_dataset
from models import CNNSpeechEnhancer
from helpers import compute_average_logmel, linear_to_waveform, logmel_to_linear


# pesq and stoi imports 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_snr(dataset, snr):
    if hasattr(dataset, 'snr'):
        dataset.snr = snr
    else:
        method = getattr(dataset, 'set_snr', None)
        if method is not None and callable(method):
            method(snr)
def set_load_waveforms(dataset, load_waveforms):
    if hasattr(dataset, 'load_waveforms'):
        dataset.load_waveforms = load_waveforms
    else:
        method = getattr(dataset, 'set_load_waveforms', None)
        if method is not None and callable(method):
            method(load_waveforms)

class BaseExperiment(ABC):
    def __init__(self, batch_size, cuda=False, train_loader=None, test_loader=None):
        self.batch_size = batch_size
        self.cuda = cuda
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.train_loader = train_loader
        self.test_loader = test_loader

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
    def setup_data_from_splits(self):
        """Set up data loaders using pre-generated splits"""
        split_file = f'splits/split_{self.experiment_no - 1}.pkl'  # 0-indexed
        
        if not os.path.exists(split_file):
            logger.warning(f"Split file {split_file} not found. Falling back to generating splits on the fly.")
            self.setup_data_from_scratch(self.heards_dir, self.speech_dir)
            return
            
        logger.info(f"Loading splits from {split_file}")
        with open(split_file, 'rb') as f:
            split = pickle.load(f)
            
            # Extract the mixed datasets from the splits
            train_mixed_ds = split['random_snr']['subsets']['train']['mixed']
            test_mixed_ds = split['random_snr']['subsets']['test']['mixed']
            
            def speech_enh_collate_fn(batch, ignore = True):
                noisy_list = []
                clean_list = []
                envs = []
                recs = []
                cut_ids = []
                snip_ids = []
                extras = []
                snrs = []

                for (noisy, clean, env, recsit, cut_id, snip_id, extra, snr) in batch:
                    if ignore and env in ["CocktailParty", "InterfereringSpeakers"]:
                        # Skip these examples during training since we don't have clean audio for them
                        continue
                    noisy_list.append(noisy)
                    clean_list.append(clean)
                    envs.append(env)
                    recs.append(recsit)
                    cut_ids.append(cut_id)
                    snip_ids.append(snip_id)
                    extras.append(extra)
                    snrs.append(snr)

                return noisy_list, clean_list, envs, recs, cut_ids, snip_ids, extras, snrs

            self.train_loader = DataLoader(
                train_mixed_ds, batch_size=self.batch_size, shuffle=True, collate_fn=speech_enh_collate_fn
            )
            self.test_loader = DataLoader(
                test_mixed_ds, batch_size=self.batch_size, shuffle=False, collate_fn=lambda x: speech_enh_collate_fn(x, ignore=True)
            )
    def setup_data_from_scratch(self, heards_dir, speech_dir):
        """Set up data loaders from scratch without using pre-generated splits"""
        full_background_ds = BackgroundDataset(heards_dir)
        full_speech_ds = SpeechDataset(speech_dir)

        train_background_ds, test_background_ds, train_background_speech_ds, test_background_speech_ds = split_background_dataset(full_background_ds)

        # combine train_background_ds and train_background_speech_ds
        train_background = torch.utils.data.ConcatDataset([train_background_ds, train_background_speech_ds])
        test_background = torch.utils.data.ConcatDataset([test_background_ds, test_background_speech_ds])

        train_speech_ds, test_speech_ds = split_speech_dataset(full_speech_ds)
        train_noisy_ds = MixedAudioDataset(train_background, train_speech_ds)
        test_noisy_ds = MixedAudioDataset(test_background, test_speech_ds)
        
        def speech_enh_collate_fn(batch, ignore = True):
            noisy_list = []
            clean_list = []
            envs = []
            recs = []
            cut_ids = []
            snip_ids = []
            extras = []
            snrs = []

            for (noisy, clean, env, recsit, cut_id, snip_id, extra, snr) in batch:
                if ignore and env in ["CocktailParty", "InterfereringSpeakers"]:
                    # Skip these examples during training since we don't have clean audio for them
                    continue
                noisy_list.append(noisy)
                clean_list.append(clean)
                envs.append(env)
                recs.append(recsit)
                cut_ids.append(cut_id)
                snip_ids.append(snip_id)
                extras.append(extra)
                snrs.append(snr)

            return noisy_list, clean_list, envs, recs, cut_ids, snip_ids, extras, snrs

        self.train_loader = DataLoader(
            train_noisy_ds, batch_size=self.batch_size, shuffle=True, collate_fn=speech_enh_collate_fn
        )
        self.test_loader = DataLoader(
            test_noisy_ds, batch_size=self.batch_size, shuffle=False, collate_fn=lambda x: speech_enh_collate_fn(x, ignore=False)
        )
        
    def precompute_test_logmels(self, test_loader):
        """Precompute logmels for test data across all SNR levels"""
        logger.info("Precomputing logmels for test data...")
        self.snr_levels = [-21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21]
        self.feature_cache = {}
        
        # Create base directory for saving wav files
        base_dir = os.path.join('models', 'speech_enhance_snr_all', str(self.experiment_no))
        wav_dir = os.path.join(base_dir, 'wav_files')
        os.makedirs(wav_dir, exist_ok=True)
        
        # Precompute logmels for each SNR level
        for snr in self.snr_levels:
            # Set SNR for the dataset
            dataset = self.test_loader.dataset
            set_snr(dataset, snr)
            set_load_waveforms(dataset, True)  # Ensure we load waveforms

            # Compute logmels for this SNR level
            for batch in tqdm(test_loader, desc=f"Precomputing logmels for SNR {snr}", unit="batch"):
                noisy, clean, envs, recs, cut_ids, snip_ids, extras, snrs = batch
                # TOOD: See why extras[1] could be empty
                if len(noisy) == 0:
                    continue
                noisy_logmels = compute_average_logmel(noisy, self.device)
                clean_logmels = compute_average_logmel(clean, self.device)

                for extra in extras:
                    if extra[1] is None:
                        print(f"Extra {extra} is None")
                
                # Cache the features and save wav files
                for i, (env, rec, cut_id, snip_id, snr_val, noisy_logmel, clean_logmel) in enumerate(zip(envs, recs, cut_ids, snip_ids, snrs, noisy_logmels, clean_logmels)):
                    key = f"{env}_{rec}_{cut_id}_{snip_id}_{snr_val}"
                    
                    # Save wav files
                    rec_dir = os.path.join(wav_dir, rec, str(snr_val))
                    os.makedirs(rec_dir, exist_ok=True)
                    
                    # Save noisy wav files
                    noisy_path = os.path.join(rec_dir, f'{cut_id}_{snip_id}')
                    sample_L = np.array(noisy[i][0])
                    sample_R = np.array(noisy[i][1])
                    sf.write(f"{noisy_path}_L.wav", sample_L, 16000)
                    sf.write(f"{noisy_path}_R.wav", sample_R, 16000)
                    
                    # Save clean wav files if available
                    if clean[i] is not None:
                        clean_path = os.path.join(rec_dir, f'{cut_id}_clean')
                        clean_L = np.array(clean[i][0])
                        clean_R = np.array(clean[i][1])
                        sf.write(f"{clean_path}_L.wav", clean_L, 16000)
                        sf.write(f"{clean_path}_R.wav", clean_R, 16000)
                    
                    # Cache the features
                    self.feature_cache[key] = {
                        'noisy_logmel': noisy_logmel,
                        'clean_logmel': clean_logmel,
                        'extra': extras[i],
                        'wav_paths': {
                            'noisy': noisy_path,
                            'clean': clean_path if clean[i] is not None else None
                        }
                    }
            torch.mps.empty_cache()
            set_load_waveforms(dataset, False)

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
                parts = key.split('_')
                if parts[0] == 'SpeechIn':
                    # Format: SpeechIn_env_recsit_cut_snippet_snr
                    env = f"SpeechIn_{parts[1]}"
                    rec = parts[2]
                    cut_id = parts[3]
                    snip_id = parts[4]
                    snr = parts[5]
                else:
                    # Format: env_recsit_cut_snippet_snr
                    env = parts[0]
                    rec = parts[1]
                    cut_id = parts[2]
                    snip_id = parts[3]
                    snr = parts[4]
                return data['noisy_logmel'], data['clean_logmel'], env, rec, cut_id, snip_id, data['extra'], float(snr)

        cached_dataset = CachedFeaturesDataset(self.feature_cache)
        return DataLoader(
            cached_dataset,
            batch_size=1,
            shuffle=False  # No shuffling for test data
        )

    def test(self, base_dir, test_loader: DataLoader, model: CNNSpeechEnhancer, trainer: SpeechEnhanceAdamEarlyStopTrainer):
        model.eval()

        # Initialize dictionaries to store scores
        scores = {
            'overall': {
                'before_pesq': {}, 'before_stoi': {}, 
                'after_pesq': {}, 'after_stoi': {}
            }
        }
        env_scores = {}
        test_files = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", unit="batch"):
                logmels, clean, envs, recs, cut_ids, snip_ids, extras, snrs = batch
                if len(logmels) == 0:
                    continue

                # Normalize the input
                normalised_noisy_logmels = trainer.normalize_logmels(logmels)

                enhanced = model(normalised_noisy_logmels)

                sample_rate = 16000
                inverted = logmel_to_linear(logmel_spectrogram=enhanced.squeeze(2), sample_rate=sample_rate, n_fft=1024, n_mels=40, device=self.device)
                hop_length = int(0.02 * sample_rate)
                win_length = int(0.04 * sample_rate)

                # Denormalize the enhanced logmel
                denormalised = trainer.denormalize_logmels(enhanced.squeeze(2), is_clean=True)
                
                inverted_denormalised = logmel_to_linear(logmel_spectrogram=denormalised, sample_rate=sample_rate, n_fft=1024, n_mels=40, device=self.device)
                waveform_denormalised = linear_to_waveform(linear_spectrogram_batch=inverted_denormalised, sample_rate=sample_rate, n_fft=1024, device=self.device, hop_length=hop_length, win_length=win_length)

                # Calculate PESQ and STOI scores for both original and enhanced audio
                for i, (cut_id, rec, snr, env, extra) in enumerate(zip(cut_ids, recs, snrs, envs, extras)):
                    if clean[i] is None:
                        continue
                    
                    # Load noisy and clean audio from saved files
                    key = f"{env}_{rec}_{cut_id}_{snip_ids[i]}_{int(snr)}"
                    wav_paths = self.feature_cache[key]['wav_paths']
                    noisy_audio = sf.read(f"{wav_paths['noisy']}_L.wav")[0]
                    clean_audio = sf.read(f"{wav_paths['clean']}_L.wav")[0]
                    enhanced_audio = waveform_denormalised[i].cpu().numpy()

                    before_pesq = pesq(sample_rate, clean_audio, noisy_audio, 'wb')
                    after_pesq = pesq(sample_rate, clean_audio, enhanced_audio, 'wb')
                    before_stoi = stoi(clean_audio, noisy_audio, sample_rate, extended=False)
                    after_stoi = stoi(clean_audio, enhanced_audio, sample_rate, extended=False)

                    snr_key = str(snr)
                    for d in [scores['overall'], scores.get(env, {'before_pesq': {}, 'before_stoi': {}, 'after_pesq': {}, 'after_stoi': {}})]:
                        if snr_key not in d['before_pesq']:
                            d['before_pesq'][snr_key] = []
                            d['before_stoi'][snr_key] = []
                            d['after_pesq'][snr_key] = []
                            d['after_stoi'][snr_key] = []
                        d['before_pesq'][snr_key].append(before_pesq)
                        d['before_stoi'][snr_key].append(before_stoi)
                        d['after_pesq'][snr_key].append(after_pesq)
                        d['after_stoi'][snr_key].append(after_stoi)

                    if env not in env_scores:
                        env_scores[env] = {'before_pesq': {}, 'before_stoi': {}, 'after_pesq': {}, 'after_stoi': {}}
                    env_scores[env]['before_pesq'][snr_key] = env_scores[env]['before_pesq'].get(snr_key, []) + [before_pesq]
                    env_scores[env]['before_stoi'][snr_key] = env_scores[env]['before_stoi'].get(snr_key, []) + [before_stoi]
                    env_scores[env]['after_pesq'][snr_key] = env_scores[env]['after_pesq'].get(snr_key, []) + [after_pesq]
                    env_scores[env]['after_stoi'][snr_key] = env_scores[env]['after_stoi'].get(snr_key, []) + [after_stoi]

                    # Add to test_files list

                # Save the enhanced audio
                os.makedirs(os.path.join(base_dir, 'enhanced'), exist_ok=True)
                for i, (rec, cut_id, snip_id, snr, extra) in enumerate(zip(recs, cut_ids, snip_ids, snrs, extras)):
                    # create a folder for the rec
                    save_path = os.path.join(base_dir, 'enhanced', rec, str(int(snr)), f'{cut_id}_{snip_id}_enhanced.wav')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    sf.write(save_path, waveform_denormalised[i].cpu().numpy(), 16000)

        # Function to print table
        def print_table(title, data):
            print(f"\n{title}")
            print("SNR\tBefore Enhancement\t\tAfter Enhancement")
            print("\tPESQ\tSTOI\t\tPESQ\tSTOI")
            for row in data:
                print(f"{row[0]}\t{row[1]}\t{row[2]}\t\t{row[3]}\t{row[4]}")

        # Print and save overall table
        overall_table_data = []
        for snr in sorted(set(scores['overall']['before_pesq'].keys()) - {'Unknown'}) + ['Unknown']:
            before_pesq_avg = np.mean(scores['overall']['before_pesq'].get(snr, []))
            before_stoi_avg = np.mean(scores['overall']['before_stoi'].get(snr, []))
            after_pesq_avg = np.mean(scores['overall']['after_pesq'].get(snr, []))
            after_stoi_avg = np.mean(scores['overall']['after_stoi'].get(snr, []))
            overall_table_data.append([
                snr,
                f"{before_pesq_avg:.2f}",
                f"{before_stoi_avg:.2f}",
                f"{after_pesq_avg:.2f}",
                f"{after_stoi_avg:.2f}"
            ])
        print_table("Overall Results", overall_table_data)

        # Save overall results to CSV
        save_path = os.path.join(base_dir, 'results.csv')
        with open(save_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['SNR', 'Before_PESQ', 'Before_STOI', 'After_PESQ', 'After_STOI'])
            for row in overall_table_data:
                writer.writerow(row)

        # Print and save environment-specific tables
        for env, env_data in env_scores.items():
            env_table_data = []
            for snr in sorted(set(env_data['before_pesq'].keys()) - {'Unknown'}) + ['Unknown']:
                before_pesq_avg = np.mean(env_data['before_pesq'].get(snr, []))
                before_stoi_avg = np.mean(env_data['before_stoi'].get(snr, []))
                after_pesq_avg = np.mean(env_data['after_pesq'].get(snr, []))
                after_stoi_avg = np.mean(env_data['after_stoi'].get(snr, []))
                env_table_data.append([
                    snr,
                    f"{before_pesq_avg:.2f}",
                    f"{before_stoi_avg:.2f}",
                    f"{after_pesq_avg:.2f}",
                    f"{after_stoi_avg:.2f}"
                ])
            print_table(f"Results for Environment: {env}", env_table_data)

            # Save environment-specific results to CSV
            with open(os.path.join(base_dir, f'{env}_results.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['SNR', 'Before_PESQ', 'Before_STOI', 'After_PESQ', 'After_STOI'])
                for row in env_table_data:
                    writer.writerow(row)

    @abstractmethod
    def run(self):
        pass

class SpeechEnhancementExperiment(BaseExperiment):
    """
    An example experiment class that deals only with speech enhancement data:
    - We assume your MixedAudioDataset now returns (noisy, clean, possibly-other-stuff).
    - We create train / test DataLoaders directly from speech or mixed-speech data.
    """
    def __init__(
    self,
    batch_size: int = 16,
    cuda: bool = False,
    experiment_no: int = 1,
    use_splits: bool = True,
    augment: bool = True,
    ):
        self.heards_dir = '/Users/nkdem/Downloads/HEAR-DS' if not cuda else '/disk/scratch/s2203859/minf-1/HEAR-DS'
        self.speech_dir = '/Volumes/SSD/Datasets/CHiME3/CHiME3-Isolated-DEV/dt05_bth' if not cuda else '/disk/scratch/s2203859/minf-1/dt05_bth/'
        self.experiment_no = experiment_no
        self.use_splits = use_splits
        self.augment = augment
        
        # Initialize with empty loaders first
        super().__init__(batch_size=batch_size, cuda=cuda)
        
        if use_splits:
            self.setup_data_from_splits()
        else:
            self.setup_data_from_scratch(self.heards_dir, self.speech_dir)
            
        logger.info("SpeechEnhancementExperiment initialized.")

    def run(self):
        """
        Example run method: you might do a training loop here, or call
        out to some "train_model" function, evaluate, etc.
        """
        logger.info("Running speech enhancement experiment...")
        base_dir = self.create_experiment_dir("speech_enhance_snr_all", self.experiment_no)
        adam = SpeechEnhanceAdamEarlyStopTrainer(
            base_dir=base_dir,
            num_epochs=100,
            train_loader=self.train_loader,
            batch_size=self.batch_size,
            cuda=self.cuda,
            initial_lr=1e-3,
            early_stop_threshold=1e-4,
            patience=5,
            augment=self.augment,
        )

        adam.train()
        
        logger.info("Training phase completed. Starting results collection and analysis...")

        model_path = os.path.join(base_dir, 'model.pth')
        cnn = CNNSpeechEnhancer().to(self.device)
        cnn.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))

        # Precompute test logmels and create cached test dataset
        cached_test_loader = self.precompute_test_logmels(self.test_loader)
        
        # Use the cached test loader for results collection
        self.test(base_dir=base_dir, test_loader=cached_test_loader, model=cnn, trainer=adam)


if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_no", type=int)
    parser.add_argument("--cuda", action='store_true', default=False)
    parser.add_argument("--no_splits", action='store_true', default=False, 
                        help="If set, don't use pre-generated splits")
    parser.add_argument("--no_augment", action='store_true', default=False,
                        help="If set, don't use data augmentation and use cached features")
    args = parser.parse_args()

    # if arg is not provided, default to 1
    # but warn 
    if args.experiment_no is None:
        print("No experiment number provided. Defaulting to 1.")
        experiment_no = 1
    else:
        experiment_no = args.experiment_no
    cuda = args.cuda
    use_splits = not args.no_splits
    augment = not args.no_augment
    exp = SpeechEnhancementExperiment(batch_size=16, cuda=cuda, experiment_no=experiment_no, use_splits=use_splits, augment=augment)
    exp.run()
    logger.info("Done.")