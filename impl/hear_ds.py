import os
import logging
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import torchaudio
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HEARDS(Dataset):
    """
    Args:
        root_dir (str): Root directory of dataset 
    """
    def __init__(self, root_dir, audio_files=None, int_to_label=None, feature_cache=None, cuda=False, augmentation=False):
        self.device = torch.device("mps" if not cuda else "cuda")
        self.augmentation = augmentation
        logger.info(f"Augmentation: {augmentation}")
        logger.info(f"Using device: {self.device}")
        self.root_dir = root_dir
        if audio_files is None:
            self.audio_files = self._get_all_audio_files()
        else:
            self.audio_files = audio_files
        self.feature_dir = os.path.join(self.root_dir, 'features')
        os.makedirs(self.feature_dir, exist_ok=True)
        self.train_indices = []
        self.test_indices = []
        
        self.snr_levels = [-21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21]
        if self.augmentation:
            self.speech_lookup = self._initialize_speech_lookup()


        # Initialize a cache for features
        if feature_cache is not None:
            logger.info('Using provided feature cache')
        self.feature_cache = feature_cache if feature_cache is not None else {}

        # Initialize label mappings
        if int_to_label is None:
            self.label_to_int = {}
            self.int_to_label = {}
            for (_, _, label, _) in self.audio_files:
                if label not in self.label_to_int:
                    self.label_to_int[label] = len(self.label_to_int)
                    self.int_to_label[self.label_to_int[label]] = label
        else:
            self.int_to_label = int_to_label
            self.label_to_int = {v: k for k, v in int_to_label.items()}

    def _initialize_speech_lookup(self):
        speech_lookup = {}
        if self.augmentation:
            for idx, (pairs, recsit, label, snr) in enumerate(self.audio_files):
                if '_speech' in label:
                    key = (label, recsit)
                    if key not in speech_lookup:
                        speech_lookup[key] = {}
                    speech_lookup[key][snr] = pairs
        return speech_lookup

    def _get_all_audio_files(self):
        audio_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.wav') or file.endswith('.wav.zst'):
                    root_split = root.split('/')
                    relative_diff = len(root_split) - len(self.root_dir.split('/'))
                    environment = root_split[-relative_diff]
                    
                    snr_level = None
                    if relative_diff == 3:
                        environment += '_speech'
                        if self.augmentation:
                            snr_level = root_split[-1]
                            
                    recsit = file.split('_')[1]
                    file_path = os.path.join(root, file)
                    
                    if '_L' in file:
                        pair_file = file.replace('_L', '_R')
                        pair_file_path = os.path.join(root, pair_file)
                        if os.path.exists(pair_file_path):
                            if self.augmentation:
                                audio_files.append(([file_path, pair_file_path], recsit, environment, snr_level))
                            else:
                                audio_files.append(([file_path, pair_file_path], recsit, environment, None))
        return audio_files
    
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        if self.augmentation:
            pairs, recsit, label, _ = self.audio_files[idx]
            if '_speech' in label:
                key = (label, recsit)
                if key in self.speech_lookup:
                    random_snr = random.choice(self.snr_levels)
                    str_snr = str(random_snr)
                    if str_snr in self.speech_lookup[key]:
                        pairs = self.speech_lookup[key][str_snr]
        else:
            pairs, recsit, label, _ = self.audio_files[idx]
            
        logmel = self._get_mfcc(pairs[0], pairs[1])
        label_tensor = torch.tensor(self.label_to_int[label], dtype=torch.long).to(self.device)
        return pairs, logmel, label_tensor
    
    def _basename(self, pairs):
        depth = len(pairs[0].split('/')) - len(self.root_dir.split('/'))
        if depth == 4:
            snr = pairs[0].split('/')[-2]
            return os.path.basename(pairs[0]).split('.wav')[0] + f'_{snr}'
        return os.path.basename(pairs[0]).split('.wav')[0]

    def _get_mfcc(self, file_L, file_R, n_mels=40, n_fft=1024, hop=20):
        # Create a unique key for caching based on the file paths
        basename = self._basename([file_L, file_R])
        cache_key = f'{basename}'

        # Check if the feature is already in cache
        if cache_key in self.feature_cache:
            # logger.debug(f'Loading cached MFCC for {basename}')
            return self.feature_cache[cache_key]
        # check if the feature is already computed and saved
        feature_file = os.path.join(self.feature_dir, f'{basename}.pt')
        if os.path.exists(feature_file):
            # logger.debug(f'Loading saved MFCC for {basename} and saving to cache')
            self.feature_cache[cache_key] = torch.load(feature_file, weights_only=True)
            return self.feature_cache[cache_key]
        
        logger.info(f'Computing MFCC for {basename}')
        waveform_L, sample_rate_L = torchaudio.load(file_L)
        waveform_R, sample_rate_R = torchaudio.load(file_R)

        if sample_rate_L != sample_rate_R:
            logger.error(f'Sample rates do not match: left channel: {sample_rate_L}, right channel: {sample_rate_R}')
            return None

        hop_length = int(sample_rate_L * hop / 1000)  # convert ms to samples

        mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate_L,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        mel_L = mel_transform(waveform_L)
        mel_R = mel_transform(waveform_R)

        logmel_L = 20 * torch.log10(mel_L + 1e-10).to(self.device)
        logmel_R = 20 * torch.log10(mel_R + 1e-10).to(self.device)

        logmel_mean = (logmel_L + logmel_R) / 2
        logmel_mean = logmel_mean.view(1, n_mels, -1)  # Shape will be (1, n_mels, n_frames)

        # Save computed features to cache
        self.feature_cache[cache_key] = logmel_mean

        return logmel_mean
    
    def split_dataset(self, size=1):
        self.train_indices = []
        self.test_indices = []
        envs = {}
        
        # Group by environment and recsit
        for i, (pairs, recsit, label, _) in enumerate(self.audio_files):
            if label not in envs:
                envs[label] = {}
            if recsit not in envs[label]:
                envs[label][recsit] = []
            envs[label][recsit].append(i)
        
        
        # for label in envs:
        #     for recsit in envs[label]:
        #         envs[label][recsit] = envs[label][recsit][:200]

        logger.info(f'Found {len(envs)} environments')
        
        for label in envs:
            logger.info(f'Environment: {label} has {len(envs[label])} recsits')
            shuffled_recsits = list(envs[label].keys())
            random.shuffle(shuffled_recsits)
            test_recsit = shuffled_recsits[0]  # Take just one recsit for testing
            train_recsits = shuffled_recsits[1:]  # All other recsits go to training
            
            logger.info(f'Train recsits: {train_recsits}')
            logger.info(f'Test recsit: {test_recsit}')

            # Collect indices
            for recsit in train_recsits:
                self.train_indices.extend(envs[label][recsit])
            self.test_indices.extend(envs[label][test_recsit])

            logger.info(f'Environment: {label} has {len(envs[label][test_recsit])} test samples')
        
        logger.info(f'Total train set size: {len(self.train_indices)}')
        logger.info(f'Total test set size: {len(self.test_indices)}')


    def get_train_data(self):
        return [(self.audio_files[i]) for i in self.train_indices]

    def get_test_data(self):
        return [(self.audio_files[i]) for i in self.test_indices]
    
    def get_audio_file(self, idx):
        return self.audio_files[idx]
        
    def get_num_classes(self):
        return len(set([label for _, _, label, _ in self.audio_files]))
    
    def get_weights(self):
        labels = np.array([self.label_to_int[label] for label in self.label_to_int])
        class_weights = compute_class_weight('balanced', classes=labels, y=np.array([self.label_to_int[label] for _, _, label,_ in self.audio_files]))
        return torch.tensor(class_weights, dtype=torch.float).to(self.device)

if __name__ == '__main__':
    dataset = HEARDS('/Users/nkdem/Downloads/HEAR-DS')
    print(len(dataset))
    print(dataset[0])  # Access the first audio file and its label
