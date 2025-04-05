import logging
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import os
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
    def __init__(self, root_dir, split='train', n_fft=320, win_length=320, hop_length=160):
        self.root_dir = root_dir
        self.split = split
        self.audio_files = self._get_audio_files()
        # self.load_waveforms = True 
        self.feature_cache = {}
        self.train = True 
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        noisy, clean, base_name = None, None, None
        if idx in self.feature_cache:
            noisy, clean, base_name = self.feature_cache[idx]
        else:
            base_name = self.audio_files[idx][0].split('/')[-1]
            noisy_file_path, clean_file_path = self.audio_files[idx]
            noisy, _ = librosa.load(noisy_file_path, sr=DOWN_SAMPLE_TO_SAMPLE_RATE)
            clean, _ = librosa.load(clean_file_path, sr=DOWN_SAMPLE_TO_SAMPLE_RATE)
            
            # if number of samples is less than TARGET_LENGTH, pad with zeros
            noisy = librosa.util.fix_length(noisy, size=TARGET_LENGTH)
            clean = librosa.util.fix_length(clean, size=TARGET_LENGTH)
            self.feature_cache[idx] = (noisy, clean, base_name)
        noisy_mag, noisy_phase = librosa.magphase(librosa.stft(noisy, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)) # [F, T]
        clean_mag, clean_phase = librosa.magphase(librosa.stft(clean, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)) # [F, T]
        # lets transpose so the shape is [T, F]
        noisy_mag = noisy_mag.T
        clean_mag = clean_mag.T
        noisy_phase = noisy_phase.T
        clean_phase = clean_phase.T

        if self.train:
            return (noisy_mag, noisy_phase), (clean_mag, clean_phase), base_name
        else:
            # return the wav files as well
            return (noisy_mag, noisy_phase), (clean_mag, clean_phase), base_name, (noisy, clean)
    def _get_audio_files(self):
        audio_files = []
        for file_name in os.listdir(os.path.join(self.root_dir, dir_map[self.split][0])):
            noisy_file_path = os.path.join(self.root_dir, dir_map[self.split][0], file_name)
            clean_file_path = os.path.join(self.root_dir, dir_map[self.split][1], file_name)
            audio_files.append((noisy_file_path, clean_file_path))
        return audio_files
    

def get_loaders(batch_size=4, cuda =False):
    ROOT_DIR = '/Users/nkdem/Downloads/VOICEBANK' if not cuda else '/workspace/VOICEBANK'

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