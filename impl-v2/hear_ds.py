import os
import logging
import soundfile as sf
import random
from typing import Dict
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import torchaudio
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from helpers import mix_signals

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_LENGTH = 160000 # at sampling rate of 16kHz, this is 10 seconds
SAMPLE_RATE = 16000

def ensure_valid_lengths_with_speech(speech, background):
    """Modified version to handle concatenated speech"""
    # If background is longer than target_length, trim it
    if len(background) > TARGET_LENGTH:
        start = random.randint(0, len(background) - TARGET_LENGTH)
        background = background[start:start + TARGET_LENGTH]
    
    # If background is shorter than target_length, pad it
    elif len(background) < TARGET_LENGTH:
        padding = TARGET_LENGTH - len(background)
        background = np.pad(background, (0, padding), 'wrap')
    
    # Pad speech if shorter than target_length
    if len(speech) < TARGET_LENGTH:
        padding = TARGET_LENGTH - len(speech)
        speech = np.pad(speech, (0, padding), 'constant')
    return speech, background

class SpeechSample:
    def __init__(self, speech_files_used, waveforms):
        self.speech_files_used = speech_files_used
        self.waveforms = waveforms
    # make serializable
    def __getstate__(self):
        return self.speech_files_used, self.waveforms
    def __setstate__(self, state):
        self.speech_files_used, self.waveforms = state

class HEARDS(Dataset):
    """
    Args:
        root_dir (str): Root directory of dataset 
    """
    def __init__(self, root_dir, chime_dir, audio_files=None, int_to_label=None, feature_cache=None, cuda=False):
        self.chime_dir = chime_dir
        self.device = torch.device("mps" if not cuda else "cuda")
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

        # keep track of which background samples are used for each speech sample

        self.weights = None
        self.num_classes = None

        speech_files = [f for f in os.listdir(chime_dir) if f.endswith('CH0.wav')]
        self.speakers = list(set([f.split('_')[0] for f in speech_files]))
        
        # create map of speakers to their speech files
        self.speaker_to_files = {}
        for speaker in self.speakers:
            self.speaker_to_files[speaker] = [f for f in speech_files if f.startswith(speaker) and f.endswith('CH0.wav')] # only use CH0 channel

        # assert that here is at least 2 speakers
        assert len(self.speakers) >= 2, "There must be at least 2 speakers so that we can split the dataset into train and test sets"

        
        self.snr_levels = [-21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21]
        # self.snr_levels = [0]

        # Initialize a cache for features
        provided_cache = False
        if feature_cache is not None and len(feature_cache.keys()) > 0:
            logger.info('Using provided feature cache')
            provided_cache = True
        self.feature_cache = feature_cache if provided_cache else {}

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
        
        self.speech_mapping: Dict[str, Dict[str, SpeechSample]] = {}
        self.speech_samples = set() # keep track of which speech samples are used (recsit, cut)

    def _get_all_audio_files(self):
        audio_files = []
        env_file_count = {}
        valid_dirs = {'All'}

        for root, dirs, files in os.walk(self.root_dir):
            # Get the immediate parent directory name
            current_dir = os.path.basename(root)
            parent_dir = os.path.basename(os.path.dirname(root))

            # Skip if neither the current nor parent directory is in valid_dirs
            if current_dir not in valid_dirs and parent_dir not in valid_dirs:
                continue

            for file in files:
                if file.endswith('.wav') or file.endswith('.wav.zst'):
                    root_split = root.split('/')
                    relative_diff = len(root_split) - len(self.root_dir.split('/'))
                    environment = root_split[-relative_diff]

                    # if env_file_count.get(environment, 0) >= 100:
                    #     continue

                    # Get the recsit identifier
                    recsit = file.split('_')[1]
                    file_path = os.path.join(root, file)

                    if '_L' in file:
                        pair_file = file.replace('_L', '_R')
                        pair_file_path = os.path.join(root, pair_file)
                        if os.path.exists(pair_file_path):
                            audio_files.append(([file_path, pair_file_path], recsit, environment))
                            # env_file_count[environment] = env_file_count.get(environment, 0) + 1

        return audio_files

    
    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        return self.audio_files[idx]
    
    def _basename(self, pairs):
        # depth = len(pairs[0].split('/')) - len(self.root_dir.split('/'))
        return os.path.basename(pairs[0]).split('.wav')[0]

    def get_mel(self, pair, label, train:bool):
        waveforms = []
        snr = None
        basename = self._basename(pair)
        recsit, cut = basename.split('_')[1], basename.split('_')[2]
        if f'{recsit}_{cut}' in self.speech_samples:
            snr = random.choice(self.snr_levels)
            waveforms = self.get_speech(pair, snr, train).waveforms
        else:
            for file in pair:
                waveform, _ = torchaudio.load(file)
                waveforms.append(waveform)
        key = f'{basename}_{label}_{snr}' if snr is not None else f'{basename}_{label}'
        hop_length = int(0.02 * SAMPLE_RATE) # 20ms
        win_length = int(0.04 * SAMPLE_RATE) # 40ms
        n_mels = 40
        n_fft = 1024
        return self._get_mel(waveforms[0], waveforms[1], key, SAMPLE_RATE, win_length, hop_length, n_mels, n_fft)


    def _get_mel(self, waveform_L, waveform_R, key, sample_rate, win_length,hop_length, n_mels, n_fft):
        # Create a unique key for caching based on the file paths
        # Check if the feature is already in cache
        if key in self.feature_cache:
            return self.feature_cache[key]
        
        logger.debug(f'Computing MFCC for {key}')
        mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            # power=1.0
        ).to(self.device)
        # convert waveform to tensor
        waveform_L = torch.tensor(waveform_L, dtype=torch.float32).to(self.device)
        waveform_R = torch.tensor(waveform_R, dtype=torch.float32).to(self.device)
        
        mel_L = mel_transform(waveform_L)
        mel_R = mel_transform(waveform_R)

        logmel_L = 20 * torch.log10(mel_L + 1e-10).to(self.device)
        logmel_R = 20 * torch.log10(mel_R + 1e-10).to(self.device)

        logmel_mean = (logmel_L + logmel_R) / 2
        logmel_mean = logmel_mean.view(1, n_mels, -1)  # Shape will be (1, n_mels, n_frames)

        # Save computed features to cache
        self.feature_cache[key] = logmel_mean

        return logmel_mean
    
    def is_pair_speech(self, pair):
        basename = pair.split('/')[-1].split('.wav')[0].split('_')
        try:
            environment_no, recsit, cut = basename[0], basename[1], basename[2]
            environment = self.int_to_label[int(environment_no)]
            if f'{recsit}_{cut}' in self.speech_samples:
                return True
            else:
                return False
        except:
            # cocktail party and interfering speakers are not speech
            return False 


    
    def split_dataset(self, size=1):
        self.train_indices = []
        self.test_indices = []
        envs = {}
        
        for i, (pairs, recsit, label) in enumerate(self.audio_files):
            if label not in envs:
                envs[label] = {}
            if recsit not in envs[label]:
                envs[label][recsit] = {}
            cut = pairs[0].split('/')[-1].split('_')[2]
            if cut not in envs[label][recsit]:
                envs[label][recsit][cut] = []
            envs[label][recsit][cut].append(i)
        
        male_speakers = [s for s in self.speakers if s.startswith('M')]
        female_speakers = [s for s in self.speakers if s.startswith('F')]
        
        # shuffle the speakers
        random.shuffle(male_speakers)
        random.shuffle(female_speakers)

        self.test_speakers = [] 
        self.test_speakers.append(female_speakers.pop())
        self.test_speakers.append(male_speakers.pop())

        self.train_speakers = []
        self.train_speakers.extend(male_speakers)
        self.train_speakers.extend(female_speakers)

        logger.info(f'Test speakers: {self.test_speakers}')
        logger.info(f'Train speakers: {self.train_speakers}')

        logger.info(f'Found {len(envs)} environments')
        logger.info(f'Creating _speech counterparts for each environment (except InterferingSpeakers and CocktailParty)')
    
        # we now add new environments by mixing speech 
        # so if there's WindTurbulence and InVehicle, then we create WindTurbulence_Speech and InVehicle_Speech
        environments = list(envs.keys())
        environments.remove('InterfereringSpeakers')
        environments.remove('CocktailParty')
        for env in environments:
            envs[env + '_speech'] = {}
            # update int_to_label and label_to_int
            highest_int = max(self.int_to_label.keys())
            if f'{env}_speech' not in self.label_to_int:
                self.label_to_int[f'{env}_speech'] = len(self.label_to_int)
                self.int_to_label[self.label_to_int[f'{env}_speech']] = f'{env}_speech'
        
        # we now need to 'steal' some of the data from the original environments and these will be background data used for speech mixing 
        # we will do this by taking half of the cuts from each recsit and adding them to the new environment (making sure to remove them from the original environment)
        for env in environments:
            for recsit in envs[env]:
                cuts = list(envs[env][recsit].keys())
                random.shuffle(cuts)
                for i, cut in enumerate(cuts):
                    if i % 2 == 0:
                        envs[env + '_speech'][recsit] = envs[env + '_speech'].get(recsit, {})
                        envs[env + '_speech'][recsit][cut] = envs[env][recsit][cut]
                        del envs[env][recsit][cut]
                        self.speech_samples.add(f'{recsit}_{cut}')
        
        for label in envs:
            logger.info(f'{label}: {len(envs[label])} recsits')
            shuffled_recsits = list(envs[label].keys())
            random.shuffle(shuffled_recsits)
            test_recsit = [shuffled_recsits[0]]
            train_recsits = shuffled_recsits[1:]
            logger.info(f'Train recsits: {len(train_recsits)}')
            logger.info(f'Test recsits: {len(test_recsit)}')

            for recsit in train_recsits:
                for cut in envs[label][recsit]:
                    self.train_indices.extend(envs[label][recsit][cut])
            for test_recsit in test_recsit:
                for cut in envs[label][test_recsit]:
                    self.test_indices.extend(envs[label][test_recsit][cut])
        
        logger.info(f'Total train set size: {len(self.train_indices)}')
        logger.info(f'Total test set size: {len(self.test_indices)}')

        labels = np.array([self.label_to_int[label] for label in self.label_to_int])
        class_weights = compute_class_weight('balanced', classes=labels, y=np.array([self.label_to_int[label] for label in envs for recsit in envs[label] for cut in envs[label][recsit]]))
        self.weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        self.num_classes = len(self.label_to_int)
    
    def get_speech(self, pair, snr, train:bool):
        # check if we have the speech in the cache
        basename = self._basename(pair)
        if snr in self.speech_mapping and basename in self.speech_mapping[snr]:
            return self.speech_mapping[snr][basename]
        else:
            required_length = int(0.75 * TARGET_LENGTH)
            concatenated_speech = np.array([])
            used_files = []

            if train:
                speaker = random.choice(self.train_speakers)
            else:
                speaker = random.choice(self.test_speakers)
            while len(concatenated_speech) < required_length:
                # pick a random speaker 
                # now from the speaker, pick a random file
                speech_files = self.speaker_to_files[speaker]
                speech_file = random.choice(speech_files)
                speech, sr = sf.read(os.path.join(self.chime_dir, speech_file))

                concatenated_speech = np.concatenate((concatenated_speech, speech))
                used_files.append(speech_file)
            if len(concatenated_speech) > TARGET_LENGTH:
                concatenated_speech = concatenated_speech[:TARGET_LENGTH]
            
            background_l, bg_sr_l = sf.read(pair[0])
            background_r, bg_sr_r = sf.read(pair[1])

            speech_l, background_l_chunk = ensure_valid_lengths_with_speech(concatenated_speech, background_l)
            speech_r, background_r_chunk = ensure_valid_lengths_with_speech(concatenated_speech, background_r)

            mixed_l = mix_signals(speech_l, background_l_chunk, snr)
            mixed_r = mix_signals(speech_r, background_r_chunk, snr)

            speech_sample = SpeechSample(used_files, [mixed_l, mixed_r])
            
            if snr not in self.speech_mapping:
                self.speech_mapping[snr] = {}
            self.speech_mapping[snr][basename] = speech_sample  

            return speech_sample

    def get_train_data(self):
        return [(self.audio_files[i]) for i in self.train_indices]

    def get_test_data(self):
        return [(self.audio_files[i]) for i in self.test_indices]
    
    def get_audio_file(self, idx):
        return self.audio_files[idx]
        
    def get_num_classes(self):
        return self.num_classes
    
    # def get_weights(self):
    #     labels = np.array([self.label_to_int[label] for label in self.label_to_int])
    #     class_weights = compute_class_weight('balanced', classes=labels, y=np.array([self.label_to_int[label] for _, _, label in self.audio_files]))
    #     return torch.tensor(class_weights, dtype=torch.float).to(self.device)

if __name__ == '__main__':
    dataset = HEARDS('/Users/nkdem/Downloads/HEAR-DS',chime_dir = '/Volumes/SSD/Datasets/CHiME3/CHiME3-Isolated-DEV/dt05_bth')
    dataset.split_dataset()
