import os
import logging
import soundfile as sf
import random
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np

from helpers import mix_signals  # assumes mix_signals(signal, background, snr) is provided

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_LENGTH = 160000  # (10 seconds at 16kHz)
SAMPLE_RATE = 16000

class TUTDataset(Dataset):
    def __init__(self, root_dir, audio_files=None):
        self.root_dir = root_dir
        self.labels = self._get_labels()
        if audio_files is None:
            self.audio_files = self._get_audio_files()
        else:
            # self.audio_files = [(audio_file, self.labels[f'{os.path.basename(audio_file)}']) for audio_file in audio_files]
            self.audio_files = [] 
            for audio_file in audio_files:
                base = os.path.basename(audio_file)
                label = self.labels.get(f'{base}', 'unknown')
                # check if it exists
                if os.path.exists(audio_file):
                    self.audio_files.append((audio_file, label))
    
    def _get_audio_files(self): 
        audio_files = []
        for file_name in os.listdir(os.path.join(self.root_dir, 'audio')):
            file_path = os.path.join(self.root_dir,  file_name)
            if os.path.isfile(file_path) and file_path.endswith('.wav'):
                basename = os.path.basename(file_path)
                label = self.labels.get(f'{basename}', 'unknown')
                audio_files.append((file_path, label))
        return audio_files
    def _get_labels(self):
        # there's a file called /Users/nkdem/Downloads/TUT/TUT-acoustic-scenes-2017-development 13/meta.txt 
        # that contains the labels for each audio file
        # audio/b020_90_100.wav	beach	b020
        with open(os.path.join(self.root_dir, 'meta.txt'), 'r') as f:
            labels = {}
            for line in f:
                parts = line.strip().split('\t')
                labels[parts[0].split('/')[1]] = parts[1]
            return labels


    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path,label = self.audio_files[idx]
        audio, sample_rate = torchaudio.load(audio_path)
        
        # Check if the audio is stereo
        if audio.shape[0] == 2:
            left_channel = audio[0]
            right_channel = audio[1]
        else:
            # If mono, duplicate the channel
            left_channel = audio[0]
            right_channel = audio[0]
        
        # Here you might want to preprocess the audio, e.g., resample, normalize, etc.
        # For simplicity, we're just returning the raw channels
        
        base = os.path.basename(audio_path)
        return (left_channel, right_channel), label, base

def get_folds(root_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    folds = {}
    evaluation_setup_dir = os.path.join(root_dir, 'evaluation_setup')
    for fold in range(1, 5):
        for set_type in ['train', 'evaluate']:
            file_name = f'fold{fold}_{set_type}.txt'
            file_path = os.path.join(evaluation_setup_dir, file_name)
            with open(file_path, 'r') as f:
                fold_data = [line.strip().split('\t') for line in f if line.strip()]
                fold_data = list(map(lambda x: (x[0].split('/')[1], x[1]), fold_data))
                folds.setdefault(f'fold{fold}', {}).update({set_type: fold_data})
    return folds

def get_datasets_for_fold(root_dir: str, down_sampled_dir: str, fold: str) -> Tuple[TUTDataset, TUTDataset]:
    folds = get_folds(root_dir)
    train_files = []
    test_files = []

    # Collect training files from all other folds
    train_files.extend([os.path.join(down_sampled_dir, file[0]) for file in folds[fold]['train']])
    
    # Collect test files from the specified fold
    test_files = [os.path.join(down_sampled_dir, file[0]) for file in folds[fold]['evaluate']]

    # Create datasets
    train_dataset = TUTDataset(root_dir=root_dir, audio_files=train_files)

    test_dataset = TUTDataset(root_dir=root_dir, audio_files=test_files)

    return train_dataset, test_dataset

if __name__ == '__main__':
    # Create a dataset object
    root_dir = '/Users/nkdem/Downloads/TUT-acoustic-scenes-2017-development'
    dataset = TUTDataset(root_dir)
    print(dataset.audio_files[:5])

    # Get folds
    folds = get_folds(root_dir)
    for fold, sets in folds.items():
        print(f"\n{fold}:")
        for set_type, data in sets.items():
            print(f"  {set_type}: {len(data)} items")

    # Example of how you might use the folds for training and testing
    for fold in folds:
        train_dataset, test_dataset = get_datasets_for_fold(root_dir, fold)
        print(f"\nUsing {fold} for testing:")
        print(f"  Test set size: {len(test_dataset)}")
        print(f"  Train set size: {len(train_dataset)}")
        # print first 5 labels
        print(f"  First 5 labels: {test_dataset[0]}")