import os
import logging
import soundfile as sf
import random
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ORIGINAL_SAMPLE_RATE = 48000
DOWN_SAMPLE_TO_SAMPLE_RATE = 16000
TARGET_LENGTH = 32768

dir_map = {
    'train': ('train/noisy', 'train/clean'),
    'test': ('test/noisy', 'test/clean')
}

class VoicebankDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.audio_files = self._get_audio_files()
        self.resampler = torchaudio.transforms.Resample(ORIGINAL_SAMPLE_RATE, DOWN_SAMPLE_TO_SAMPLE_RATE)
        self.load_waveforms = True  # Default to True for backward compatibility
        self.feature_cache = {}

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        base_name = self.audio_files[idx][0].split('/')[-1]
        if idx in self.feature_cache:
            return self.feature_cache[idx]

        noisy_file_path, clean_file_path = self.audio_files[idx]
            
        if not self.load_waveforms:
            return np.zeros(TARGET_LENGTH), np.zeros(TARGET_LENGTH), base_name
            
        noisy_audio, sample_rate = torchaudio.load(noisy_file_path)
        clean_audio, sample_rate = torchaudio.load(clean_file_path)
        
        # Resample to 16kHz
        noisy_audio = self.resampler(noisy_audio)
        clean_audio = self.resampler(clean_audio)
        
        # if number of samples is less than TARGET_LENGTH, pad with zeros
        if noisy_audio.shape[1] < TARGET_LENGTH:
            noisy_audio = torch.nn.functional.pad(noisy_audio, (0, TARGET_LENGTH - noisy_audio.shape[1]))
            clean_audio = torch.nn.functional.pad(clean_audio, (0, TARGET_LENGTH - clean_audio.shape[1]))
        # if number of samples is greater than TARGET_LENGTH, trim
        elif noisy_audio.shape[1] > TARGET_LENGTH:
            noisy_audio = noisy_audio[:, :TARGET_LENGTH]
            clean_audio = clean_audio[:, :TARGET_LENGTH]
        # return (noisy_audio, noisy_audio), (clean_audio, clean_audio), base_name # my cnn model assumes averaging logmels across channels so we need to return a tuple of two tensors
        self.feature_cache[idx] = (noisy_audio, clean_audio, base_name)
        return noisy_audio, clean_audio, base_name # my cnn model assumes averaging logmels across channels so we need to return a tuple of two tensors
    
    def _get_audio_files(self):
        audio_files = []
        for file_name in os.listdir(os.path.join(self.root_dir, dir_map[self.split][0])):
            noisy_file_path = os.path.join(self.root_dir, dir_map[self.split][0], file_name)
            clean_file_path = os.path.join(self.root_dir, dir_map[self.split][1], file_name)
            audio_files.append((noisy_file_path, clean_file_path))
        return audio_files
    

def get_loaders(batch_size=4, cuda =False):
    ROOT_DIR = '/Users/nkdem/Downloads/VOICEBANK' if not cuda else '/workspace/DS_10283_2791'

    train_dataset = VoicebankDataset(os.path.join(ROOT_DIR), 'train')
    test_dataset = VoicebankDataset(os.path.join(ROOT_DIR), 'test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, test_loader
    
    

if __name__ == "__main__":
    train_loader, test_loader = get_loaders()

    for batch in train_loader:
        print(batch)
        break