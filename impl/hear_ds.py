import os
import logging
import random
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torchaudio
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
logger.info(f"Using device: {device}")

class HEARDS(Dataset):
    """
    Args:
        root_dir (str): Root directory of dataset 
    """
    def __init__(self, root_dir, audio_files=None, int_to_label=None, feature_cache=None):
        self.root_dir = root_dir
        if audio_files is None:
            self.audio_files = self._get_all_audio_files()
        else:
            self.audio_files = audio_files
        self.feature_dir = os.path.join(self.root_dir, 'features')
        os.makedirs(self.feature_dir, exist_ok=True)
        self.train_indices = []
        self.test_indices = []
        
        # Initialize a cache for features
        self.feature_cache = feature_cache if feature_cache is not None else {}

        # Initialize label mappings
        if int_to_label is None:
            self.label_to_int = {}
            self.int_to_label = {}
            for (_, _, label) in self.audio_files:
                if label not in self.label_to_int:
                    self.label_to_int[label] = len(self.label_to_int)
                    self.int_to_label[self.label_to_int[label]] = label
        else:
            self.int_to_label = int_to_label
            self.label_to_int = {v: k for k, v in int_to_label.items()}

    def _get_all_audio_files(self):
        audio_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.wav'):
                    root_split = root.split('/')
                    relative_diff = len(root_split) - len(self.root_dir.split('/'))
                    environment = root_split[-relative_diff]

                    if relative_diff == 3:
                        environment += '_speech'
                        # get SNR 
                        # SNR = root_split[-1]
                        # if SNR != '0':
                        #     continue
                    recsit = file.split('_')[1]  # Assuming RECSIT is extracted from the filename

                    file_path = os.path.join(root, file)

                    if '_L' in file:
                        logger.debug(f'Found left channel file: {file_path}')
                        pair_file = file.replace('_L', '_R')
                        pair_file_path = os.path.join(root, pair_file)
                        if not os.path.exists(pair_file_path):
                            logger.warning(f'Pair file not found for left channel file: {file_path}')
                        else:
                            logger.debug(f'Found right channel file: {pair_file_path}')
                            audio_files.append(([file_path, pair_file_path], recsit, environment))
        return audio_files
    
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        pairs, recsit, label = self.audio_files[idx]
        logmel = self._get_mfcc(pairs[0], pairs[1])
        label_tensor = torch.tensor(self.label_to_int[label], dtype=torch.long).to(device)
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

        logmel_L = 20 * torch.log10(mel_L + 1e-10).to(device)
        logmel_R = 20 * torch.log10(mel_R + 1e-10).to(device)

        logmel_mean = (logmel_L + logmel_R) / 2
        logmel_mean = logmel_mean.view(1, n_mels, -1)  # Shape will be (1, n_mels, n_frames)

        # Save computed features to cache
        self.feature_cache[cache_key] = logmel_mean

        return logmel_mean
    
    def split_dataset(self, size=1):
        self.train_indices = []
        self.test_indices = []
        envs = {}
        for i, (pairs, recsit, label) in enumerate(self.audio_files):
            if label not in envs:
                envs[label] = {}
            if recsit not in envs[label]:
                envs[label][recsit] = []
            envs[label][recsit].append(i)
        logger.info(f'Found {len(envs)} environments')
        for label in envs:
            logger.info(f'Environment: {label} has {len(envs[label])} recsits')
            shuffled_recsits = list(envs[label].keys())
            random.shuffle(shuffled_recsits)
            test_recsits = shuffled_recsits[:len(shuffled_recsits) // 2]
            train_recsits = shuffled_recsits[len(shuffled_recsits) // 2:]
            logger.info(f'Train recsits: {train_recsits}')
            logger.info(f'Test recsits: {test_recsits}')

            train_indices = []
            test_indices = []
            for recsit in train_recsits:
                train_indices.extend(envs[label][recsit][:int(size * len(envs[label][recsit]))])
            for recsit in test_recsits:
                test_indices.extend(envs[label][recsit])
            self.train_indices.extend(train_indices)
            self.test_indices.extend(test_indices)

            logger.info(f'Environment: {label} has {len(train_indices)} train samples and {len(test_indices)} test samples')
        logger.info(f'Train set size: {len(self.train_indices)}')
        logger.info(f'Test set size: {len(self.test_indices)}')

    def get_train_data(self):
        train = []
        for i in self.train_indices:
            pairs, recsit, label = self.audio_files[i]
            train.append((pairs, recsit, label))
        return train

    def get_test_data(self):
        test = []
        for i in self.test_indices:
            pairs, recsit, label = self.audio_files[i]
            test.append((pairs, recsit, label))
        return test
    
    def get_audio_file(self, idx):
        return self.audio_files[idx]
        
    def get_num_classes(self):
        return len(set([label for _, _, label in self.audio_files]))
    
    def get_weights(self):
        labels = np.array([self.label_to_int[label] for label in self.label_to_int])
        class_weights = compute_class_weight('balanced', classes=labels, y=np.array([self.label_to_int[label] for _, _, label in self.audio_files]))
        return torch.tensor(class_weights, dtype=torch.float).to(device)


if __name__ == '__main__':
    dataset = HEARDS('/Users/nkdem/Downloads/HEAR-DS')
    print(len(dataset))
    print(dataset[0])  # Access the first audio file and its label
