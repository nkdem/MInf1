from itertools import groupby
import os
import logging
import soundfile as sf
import random
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
import tqdm

from helpers import mix_signals  # assumes mix_signals(signal, background, snr) is provided

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_LENGTH = 160000  # (10 seconds at 16kHz)
SAMPLE_RATE = 16000



def ensure_valid_lengths_with_speech(speech: np.ndarray, background: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process background: if longer than TARGET_LENGTH, randomly trim it; if shorter, pad it by wrapping.
    Process speech: if longer than TARGET_LENGTH, randomly trim it; if shorter, pad it with zeros.
    """
    # Process background (wrap padding)
    if len(background) > TARGET_LENGTH:
        start = random.randint(0, len(background) - TARGET_LENGTH)
        background = background[start:start + TARGET_LENGTH]
    elif len(background) < TARGET_LENGTH:
        padding = TARGET_LENGTH - len(background)
        background = np.pad(background, (0, padding), mode='wrap')

    # Process speech (random segment if possible)
    if len(speech) > TARGET_LENGTH:
        start = random.randint(0, len(speech) - TARGET_LENGTH)
        speech = speech[start:start + TARGET_LENGTH]
    elif len(speech) < TARGET_LENGTH:
        padding = TARGET_LENGTH - len(speech)
        speech = np.pad(speech, (0, padding), mode='constant', constant_values=0)

    return speech, background


def base_name(file_name: str) -> str:
    """
    Returns the base name of a file without the extension.
    So 03_106_42_001_ITC_L_16kHz.wav -> 03_106_42_001_ITC 
    """
    splits = file_name.split('_')[0:5]
    return '_'.join(splits)



###############################################################################
# 1. BackgroundDataset -- loads background file pairs and records the recsit.
###############################################################################
class BackgroundDataset(Dataset):
    """
    Loads background samples from a given root_dir.
    Expects the samples to be stored as .wav (or .wav.zst) files in recsit-specific directories.
    For a pair, one file contains "_L" and the other "_R".
    __getitem__ returns a tuple (file_pair, recsit, label) where label=0 indicates background-only.
    """
    def __init__(self, root_dir: str, max_samples_per_env: int = 10000, files_to_use: List[Tuple[List[str]]] = None):
        self.root_dir = root_dir
        self.load_waveforms = True
        if files_to_use is None:
            self.audio_files = self._get_all_audio_files(max_samples_per_env)
        else:
            self.audio_files = []
            for file_pair in files_to_use:
                # file is of the form '03_106_42_001_ITC_L_16kHz.wav'
                #  <ENV_ID>_<REC_ID>_<CUT_ID>_<SNIP_ID>_<TRACKNAME>_<SAMPLERATE>.wav
                splits = file_pair[0].split('_')
                environment = os.path.basename(os.path.dirname(os.path.dirname(file_pair[0])))
                recsit = splits[1]
                cut_id = splits[2]
                snip_id = splits[3]
                self.audio_files.append((file_pair, environment, recsit, cut_id, snip_id))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx: int):
        file_pair, environment, recsit, cut_id, snip_id = self.audio_files[idx]
        basename = base_name(os.path.basename(file_pair[0]))
        if self.load_waveforms:
            waveform_l, _ = sf.read(file_pair[0])
            waveform_r, _ = sf.read(file_pair[1])
        else:
            return None, None, environment, recsit, cut_id, snip_id, (basename, None), None # last element is SNR level, which is not used for background samples
        return [waveform_l, waveform_r], [], environment, recsit, cut_id, snip_id, (basename, None), None # last element is SNR level, which is not used for background samples

    def _get_all_audio_files(self, max_samples_per_env: int) -> List[Tuple[List[str], str, str, str, str]]:
        """
        Walk root_dir and collect a list of (file_pair, recsit) tuples. The recsit is assumed to be the name of the
        immediate parent directory.
        Only up to max_samples_per_env samples per recsit are loaded.
        """
        audio_files = []
        recsit_file_count: Dict[str, int] = {}

        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.wav') or file.endswith('.wav.zst'):
                    if '_L' in file:
                        pair_file = file.replace('_L', '_R')
                        file_path = os.path.join(root, file)
                        pair_file_path = os.path.join(root, pair_file)
                        if os.path.exists(pair_file_path):
                            # file is of the form '03_106_42_001_ITC_L_16kHz.wav'
                            #  <ENV_ID>_<REC_ID>_<CUT_ID>_<SNIP_ID>_<TRACKNAME>_<SAMPLERATE>.wav
                            splits = file.split('_')
                            environment = os.path.basename(os.path.dirname(root))
                            recsit = splits[1]
                            cut_id = splits[2]
                            snip_id = splits[3]
                            count = recsit_file_count.get(recsit, 0)
                            if count >= max_samples_per_env:
                                continue
                            audio_files.append(([file_path, pair_file_path], environment, recsit, cut_id, snip_id))
                            recsit_file_count[recsit] = count

        logger.info(f"Loaded {len(audio_files)} background file pairs from {self.root_dir}")
        return audio_files


###############################################################################
# 2. SpeechDataset -- loads CHiME-style speech files and extracts speaker information.
###############################################################################
class SpeechDataset(Dataset):
    """
    Loads speech audio files from a CHiME folder.
    Expects file names like "M03_052C010R_BTH.CH5.wav" or "F01_22GC010X_BTH.CH0.wav".
    The speaker is extracted from the first three characters of the filename (e.g. "M03", "F01").
    __getitem__ returns a tuple (speech_waveform, speaker) where speech_waveform is a mono numpy array trimmed or padded
    to TARGET_LENGTH.
    """
    def __init__(self, chime_dir: str, files_list: List[str] = None):
        self.chime_dir = chime_dir
        if files_list is None:
            self.speech_files = sorted([os.path.join(chime_dir, f) for f in os.listdir(chime_dir) if f.endswith('.wav')])
        else:
            self.speech_files = files_list

        if not self.speech_files:
            raise ValueError(f"No speech files found in {chime_dir}")

    def __len__(self):
        return len(self.speech_files)

    def _extract_speaker(self, filepath: str) -> str:
        """
        Extracts the speaker id from the file name.
        Assumes the speaker id is the first three characters of the file name (e.g. "M03" or "F01").
        """
        filename = os.path.basename(filepath)
        return filename[:3].upper()

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        speech_file = self.speech_files[idx]
        speech, sr = sf.read(speech_file)
        if speech.ndim > 1:
            speech = speech[:, 0]  # convert to mono
        if len(speech) > TARGET_LENGTH:
            start = random.randint(0, len(speech) - TARGET_LENGTH)
            speech = speech[start:start + TARGET_LENGTH]
        # we don't need to cover case where speech is shorter than TARGET_LENGTH as it will conflict with the mixed dataset where it assumes all speech is TARGET_LENGTH
        speaker = self._extract_speaker(speech_file)
        return speech, speaker


###############################################################################
# 3. MixedAudioDataset -- mixes background with speech.
###############################################################################
class MixedAudioDataset(Dataset):
    """
    For each item, a background audio pair is combined with a speech sample.
    Uses ensure_valid_lengths_with_speech to trim/pad and mix_signals to combine.
    Returns a tuple (mixed_audio_pair, label) where label=1 indicates speech is present.
    """
    def __init__(self, background_dataset: BackgroundDataset, speech_dataset: SpeechDataset, snr_levels: List[int] = None, channel = None, fixed_snr: bool = False, speech_cache: Dict = None):
        self.background_dataset = background_dataset
        self.speech_dataset = speech_dataset
        self.channel = channel
        self.fixed_snr = fixed_snr
        self.load_waveforms = True
        self.snr = None
        if snr_levels is None:
            self.snr_levels = [-21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21]
        else:
            self.snr_levels = snr_levels
        # Cache for speech metadata (speaker and file indices), keyed by (background_idx, snr)
        self.speech_cache = {}
        if speech_cache is None:
            self._prepopulate_speech_cache()
        else:
            self.speech_cache = speech_cache

    def _prepopulate_speech_cache(self):
        """
        Pre-populates the speech cache for all possible (background_idx, snr) combinations.
        This ensures deterministic behavior across different runs.
        """
        logger.info("Pre-populating speech cache...")
        
        for snr in tqdm.tqdm(self.snr_levels):
            for background_idx in tqdm.tqdm(range(len(self.background_dataset))):
                cache_key = (background_idx, snr)
                if cache_key not in self.speech_cache:
                    # This will automatically populate the cache for this key
                    self._get_combined_speech(cache_key)
        
        logger.info(f"Speech cache populated with {len(self.speech_cache)} combinations")

    def _get_combined_speech(self, cache_key: Tuple[int, int]) -> Tuple[np.ndarray, List[str]]:
        """
        Concatenates speech samples from self.speech_dataset until the accumulated speech 
        reaches at least 75% of TARGET_LENGTH. If cached metadata exists for this (background_idx, snr),
        uses the same files and speaker as before.
        
        Args:
            cache_key: Tuple of (background_idx, snr)
        """
        # Check if we have cached metadata for this (background_idx, snr) combination
        if cache_key in self.speech_cache:
            speech_indices = self.speech_cache[cache_key]
            concatenated_speech = np.array([], dtype=np.float32)
            samples_used = []
            for idx in speech_indices:
                speech, _ = self.speech_dataset[idx]
                concatenated_speech = np.concatenate((concatenated_speech, speech))
                samples_used.append(self.speech_dataset.speech_files[idx])
            
            if len(concatenated_speech) > TARGET_LENGTH:
                concatenated_speech = concatenated_speech[:TARGET_LENGTH]
            elif len(concatenated_speech) < TARGET_LENGTH:
                padding = TARGET_LENGTH - len(concatenated_speech)
                concatenated_speech = np.pad(concatenated_speech, (0, padding), mode='constant')
            return concatenated_speech, samples_used

        required_length = int(0.75 * TARGET_LENGTH)
        concatenated_speech = np.array([], dtype=np.float32)
        samples_used = []
        speech_indices = []  # Store indices for caching
        speaker_used = None
        while len(concatenated_speech) < required_length:
            if speaker_used is None:
                idx = random.randint(0, len(self.speech_dataset) - 1)
                speech, speaker = self.speech_dataset[idx]
                speaker_used = speaker
            else:
                idx = random.randint(0, len(self.speech_dataset) - 1)
                speech, speaker = self.speech_dataset[idx]
                while speaker != speaker_used:
                    idx = random.randint(0, len(self.speech_dataset) - 1)
                    speech, speaker = self.speech_dataset[idx]
            concatenated_speech = np.concatenate((concatenated_speech, speech))
            samples_used.append(self.speech_dataset.speech_files[idx])
            speech_indices.append(idx)
        
        if len(concatenated_speech) > TARGET_LENGTH:
            concatenated_speech = concatenated_speech[:TARGET_LENGTH]
        elif len(concatenated_speech) < TARGET_LENGTH:
            padding = TARGET_LENGTH - len(concatenated_speech)
            concatenated_speech = np.pad(concatenated_speech, (0, padding), mode='constant')
        
        # Cache the indices used for this (background_idx, snr) combination
        self.speech_cache[cache_key] = speech_indices
        return concatenated_speech, samples_used

    def __len__(self):
        if self.fixed_snr:
            return len(self.background_dataset) * len(self.snr_levels)
        return len(self.background_dataset)

    def __getitem__(self, idx: int):
        """
        Returns:
        - mixed_waveforms (list of length 2, e.g. [mixed_left, mixed_right])
        - clean_waveforms (list of length 2, e.g. [clean_left, clean_right])
        - environment, recsit, cut_jkid, (basename, speech_used), snr
        """
        if self.fixed_snr:
            # In fixed SNR mode, map the idx to the correct background sample and SNR level
            background_idx = idx // len(self.snr_levels)
            snr_idx = idx % len(self.snr_levels)
            snr = self.snr_levels[snr_idx]
            
            if background_idx >= len(self.background_dataset):
                raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}")
        else:
            # In random SNR mode, idx represents background_idx directly
            background_idx = idx
            snr = random.choice(self.snr_levels) if self.snr is None else self.snr

        file_pair, environment, recsit, cut_id, snip_id = self.background_dataset.audio_files[background_idx]
        basename = base_name(os.path.basename(file_pair[0]))
        if not self.load_waveforms:
            return (None, None, f"SpeechIn_{environment}", recsit, cut_id, snip_id, (basename, None), snr)

        # Load background channels
        background_l, _sr1 = sf.read(file_pair[0])
        background_r, _sr2 = sf.read(file_pair[1])
        if background_l.ndim > 1:
            background_l = background_l[:, 0]
        if background_r.ndim > 1:
            background_r = background_r[:, 0]

        # If environment already has speech, skip mixing and treat as background-only.
        # e.g. "CocktailParty", "InterfereringSpeakers"
        if environment in ["CocktailParty", "InterfereringSpeakers"]:
            return [background_l, background_r], None, environment, recsit, cut_id, snip_id, (basename, None), 0

        # Create cache key from background_idx and snr
        cache_key = (background_idx, snr)
        combined_speech, speech_used = self._get_combined_speech(cache_key)
        # TOOD: The speech uses all channels possible when itg should be using only one channel to select

        # For stereo "clean", you might replicate the same speech on both channels if desired:
        speech_l, bg_chunk_l = ensure_valid_lengths_with_speech(combined_speech, background_l)
        speech_r, bg_chunk_r = ensure_valid_lengths_with_speech(combined_speech, background_r)

        # The "clean" waveforms are what we'd feed to the MSE as the target
        clean_l, clean_r = speech_l.copy(), speech_r.copy()

        # Now mix them at the chosen snr
        mixed_l = mix_signals(speech_l, bg_chunk_l, snr)
        mixed_r = mix_signals(speech_r, bg_chunk_r, snr)

        environment = f"SpeechIn_{environment}"

        # Return: (noisy, clean, env, recsit, cut_id, snip_id, extra_info, snr)
        return (
            [mixed_l, mixed_r] if self.channel == None else [mixed_l, mixed_l] if self.channel == 'L' else [mixed_r, mixed_r],                # noisy mixture
            [clean_l, clean_r] if self.channel == None else [clean_l, clean_l] if self.channel == 'L' else [clean_r, clean_r],                # clean speech
            environment, 
            recsit, 
            cut_id,
            snip_id,
            (basename, speech_used),
            snr
        )

class DuplicatedMixedAudioDataset(Dataset):
    """
    Wrapper around MixedAudioDataset that duplicates each sample for each SNR level.
    This ensures that in fixed SNR mode, we have the same number of background-only samples
    as we have mixed samples (one for each SNR level).
    """
    def __init__(self, mixed_audio_dataset: MixedAudioDataset):
        self.mixed_audio_dataset = mixed_audio_dataset
    
    def set_snr(self, snr: int):
        self.mixed_audio_dataset.snr = snr
    
    def set_load_waveforms(self, load_waveforms: bool):
        self.mixed_audio_dataset.load_waveforms = load_waveforms

    def __len__(self):
        return len(self.mixed_audio_dataset)

    def __getitem__(self, idx):
        # Map the expanded index back to the original mixed audio dataset index
        original_idx = idx // len(self.mixed_audio_dataset.snr_levels)
        return self.mixed_audio_dataset[original_idx]

###############################################################################
# Helper functions for splitting datasets based on recsit and speaker.
###############################################################################
def split_background_dataset(background_ds: BackgroundDataset) -> Tuple[BackgroundDataset, BackgroundDataset, BackgroundDataset, BackgroundDataset]:
    """
    Groups background samples by recsit. For each environment, one recsit is chosen at random for the test set;
    the rest for the training set. In addition, we split each recsit's cuts into two disjoint sets:
      - background_cuts: used for background-only samples.
      - background_speech_cuts: used for the mixed set. NOTE: if the environment is "CocktailParty" or "InterfereringSpeakers",
        it will only be used for background-only (i.e. will not be included in the mixed set).
    Returns four BackgroundDatasets: train_background_ds, test_background_ds,
    train_background_speech_ds, test_background_speech_ds.
    """
    # Group files first. Here, groups is organized as:
    # { environment: { recsit: { cut_id: file_pair_list } } }
    groups: Dict[str, Dict[str, Dict[str, List[List[str]]]]] = {}
    for file_pair, environment, recsit, cut_id, _ in background_ds.audio_files:
        groups.setdefault(environment, {}).setdefault(recsit, {}).setdefault(cut_id, []).append(file_pair)

    all_environments = list(groups.keys())
    train_background_cuts = []
    train_background_speech_cuts = [] 
    test_background_cuts = []
    test_background_speech_cuts = []
    
    # Environments that already contain speech and should not be used in the mixed set:
    excluded_for_mixing = {"CocktailParty", "InterfereringSpeakers"}
    
    for env in all_environments:
        recsits = list(groups[env].keys())
        # pick one recsit at random for the test set.
        test_recsit = random.choice(recsits)
        for recsit, cuts in groups[env].items():
            # We need to handle each cut_id, which is a key.
            cut_ids = list(cuts.keys())
            random.shuffle(cut_ids)
            # Split cut_ids into two halves.
            if env == 'InterfereringSpeakers':
                # only has 1 cut
                background_cuts = cut_ids
                background_speech_cuts = []
            else:
                background_cuts = cut_ids[: len(cut_ids) // 2]
                background_speech_cuts = cut_ids[len(cut_ids) // 2 :]
            if recsit == test_recsit:
                for cut in background_cuts:
                    test_background_cuts.extend(groups[env][recsit][cut])
                # Only add to the mixed set if not in the excluded environments.
                if env not in excluded_for_mixing:
                    for cut in background_speech_cuts:
                        test_background_speech_cuts.extend(groups[env][recsit][cut])
            else:
                for cut in background_cuts:
                    train_background_cuts.extend(groups[env][recsit][cut])
                if env not in excluded_for_mixing:
                    for cut in background_speech_cuts:
                        train_background_speech_cuts.extend(groups[env][recsit][cut])
    
    logger.info(f"Selected {len(test_background_cuts)} test background cuts")
    logger.info(f"Selected {len(test_background_speech_cuts)} test background speech cuts (for mixing)")
    logger.info(f"Selected {len(train_background_cuts)} train background cuts")
    logger.info(f"Selected {len(train_background_speech_cuts)} train background speech cuts (for mixing)")
    
    train_background_ds = BackgroundDataset(background_ds.root_dir, files_to_use=train_background_cuts)
    test_background_ds = BackgroundDataset(background_ds.root_dir, files_to_use=test_background_cuts)
    train_background_speech_ds = BackgroundDataset(background_ds.root_dir, files_to_use=train_background_speech_cuts)
    test_background_speech_ds = BackgroundDataset(background_ds.root_dir, files_to_use=test_background_speech_cuts)
    
    return train_background_ds, test_background_ds, train_background_speech_ds, test_background_speech_ds

def split_speech_dataset(speech_ds: SpeechDataset) -> Tuple[SpeechDataset, SpeechDataset]:
    groups: Dict[str, List[str]] = {}
    for filepath in speech_ds.speech_files:
        # filter only CH0
        if 'CH0' not in filepath:
            continue
        speaker = speech_ds._extract_speaker(filepath)
        groups.setdefault(speaker, []).append(filepath)
    all_speakers = list(groups.keys())

    male_speakers = [s for s in all_speakers if s.startswith('M')]
    female_speakers = [s for s in all_speakers if s.startswith('F')]
    # shuffle
    random.shuffle(male_speakers)
    random.shuffle(female_speakers)


    test_speakers = []
    test_speakers.append(male_speakers.pop())
    test_speakers.append(female_speakers.pop())

    logger.info(f"Selected test speakers for speech: {test_speakers}")

    train_speakers = []
    train_speakers.extend(male_speakers)
    train_speakers.extend(female_speakers)
    logger.info(f"Selected train speakers for speech: {train_speakers}")

    train_files = []
    test_files = []
    for speaker, files in groups.items():
        if speaker in test_speakers:
            test_files.extend(files)
        else:
            train_files.extend(files)
    return SpeechDataset(speech_ds.chime_dir, files_list=train_files), SpeechDataset(speech_ds.chime_dir, files_list=test_files)


# ###############################################################################
# # Main function that performs splitting and creates DataLoaders.
# ###############################################################################
# def main():
#     # Replace these paths with your actual directories.
#     ROOT_DIR = '/Users/nkdem/Downloads/HEAR-DS'
#     CHIME_DIR = '/Volumes/SSD/Datasets/CHiME3/CHiME3-Isolated-DEV/dt05_bth'
#     CACHE_DIR = 'cache'  # Directory to store speech caches
    
#     # Create the full datasets.
#     full_background_ds = BackgroundDataset(ROOT_DIR)
#     full_speech_ds = SpeechDataset(CHIME_DIR)
    
#     # Split the background dataset based on recsit.
#     train_background_ds, test_background_ds, train_background_speech_ds, test_background_speech_ds = split_background_dataset(full_background_ds)
    
#     # Split the speech dataset based on speaker.
#     train_speech_ds, test_speech_ds = split_speech_dataset(full_speech_ds)
    
#     logger.info(f"Train background samples: {len(train_background_ds)}; Test background samples: {len(test_background_ds)}")
#     logger.info(f"Train speech samples: {len(train_speech_ds)}; Test speech samples: {len(test_speech_ds)}")
    
#     # Create Mixed Audio Datasets for training and test sets with caching
#     os.makedirs(CACHE_DIR, exist_ok=True)
#     train_cache_path = os.path.join(CACHE_DIR, 'train_speech_cache.pkl')
#     test_cache_path = os.path.join(CACHE_DIR, 'test_speech_cache.pkl')
    
#     train_mixed_ds = MixedAudioDataset(
#         train_background_speech_ds, 
#         train_speech_ds,
#         cache_path=train_cache_path
#     )
#     test_mixed_ds = MixedAudioDataset(
#         test_background_speech_ds, 
#         test_speech_ds,
#         cache_path=test_cache_path
#     )
    
#     train_combined = torch.utils.data.ConcatDataset([train_background_ds, train_mixed_ds])
#     test_combined = torch.utils.data.ConcatDataset([test_background_ds, test_mixed_ds])

#     def collate_fn(batch):
#         mixed_audios, environments, recsits, cut_ids, labels = zip(*batch)
#         mixed_audios = [[sf.read(mixed_audios[i][0]), sf.read(mixed_audios[i][1])] if labels[i] == 0 else mixed_audios[i] for i in range(len(mixed_audios))]
#         return mixed_audios, environments, recsits, cut_ids, labels
#     train_loader = DataLoader(train_combined, batch_size=32, shuffle=True, collate_fn=collate_fn)
#     test_loader = DataLoader(test_combined, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
#     # Demonstrate by iterating over one batch from each loader.
#     count = 0
#     for batch in train_loader:
#         mixed_audios, environment, recsit, cut_id, labels = batch
#         logger.info(f"Train Mixed Audio Batch Labels: {labels}")
#     for batch in test_loader:
#         mixed_audios, environment, recsit, cut_id, labels = batch
#         logger.info(f"Test Mixed Audio Batch Labels: {labels}")

# if __name__ == '__main__':
#     main()
