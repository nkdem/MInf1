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
                self.audio_files.append((file_pair, environment, recsit, cut_id))

    def _get_all_audio_files(self, max_samples_per_env: int) -> List[Tuple[List[str], str, str, str]]:
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
                            count = recsit_file_count.get(recsit, 0)
                            if count >= max_samples_per_env:
                                continue
                            audio_files.append(([file_path, pair_file_path], environment, recsit, cut_id))
                            recsit_file_count[recsit] = count + 1

        logger.info(f"Loaded {len(audio_files)} background file pairs from {self.root_dir}")
        return audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx: int):
        file_pair, environment, recsit, cut_id = self.audio_files[idx]
        waveform_l, _ = torchaudio.load(file_pair[0])
        waveform_r, _ = torchaudio.load(file_pair[1])
        basename = base_name(os.path.basename(file_pair[0]))
        return file_pair, environment, recsit, cut_id, (basename, None), None # last element is SNR level, which is not used for background samples


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
        elif len(speech) < TARGET_LENGTH:
            padding = TARGET_LENGTH - len(speech)
            speech = np.pad(speech, (0, padding), mode='constant')
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
    def __init__(self, background_dataset: BackgroundDataset, speech_dataset: SpeechDataset, snr_levels: List[int] = None):
        self.background_dataset = background_dataset
        self.speech_dataset = speech_dataset
        if snr_levels is None:
            self.snr_levels = [-21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21]
        else:
            self.snr_levels = snr_levels

    def __len__(self):
        return len(self.background_dataset)

    def _get_combined_speech(self) -> np.ndarray:
        """
        Concatenates randomly selected speech samples from self.speech_dataset until 
        the accumulated speech reaches at least 75% of TARGET_LENGTH.
        If the concatenated speech ends up longer than TARGET_LENGTH, then it is truncated.
        """
        required_length = int(0.75 * TARGET_LENGTH)
        concatenated_speech = np.array([], dtype=np.float32)
        samples_used = []
        while len(concatenated_speech) < required_length:
            idx = random.randint(0, len(self.speech_dataset) - 1)
            speech, _ = self.speech_dataset[idx]
            concatenated_speech = np.concatenate((concatenated_speech, speech))
            samples_used.append(self.speech_dataset.speech_files[idx])
        if len(concatenated_speech) > TARGET_LENGTH:
            concatenated_speech = concatenated_speech[:TARGET_LENGTH]
        elif len(concatenated_speech) < TARGET_LENGTH:
            padding = TARGET_LENGTH - len(concatenated_speech)
            concatenated_speech = np.pad(concatenated_speech, (0, padding), mode='constant')
        return concatenated_speech, samples_used

    def __getitem__(self, idx: int):
        """
        Returns:
        - mixed_waveforms (list of length 2, e.g. [mixed_left, mixed_right])
        - clean_waveforms (list of length 2, e.g. [clean_left, clean_right])
        - environment, recsit, cut_id, (basename, speech_used), snr
        """
        file_pair, environment, recsit, cut_id, _, _ = self.background_dataset[idx]

        # Load background channels
        background_l, _sr1 = sf.read(file_pair[0])
        background_r, _sr2 = sf.read(file_pair[1])
        if background_l.ndim > 1:
            background_l = background_l[:, 0]
        if background_r.ndim > 1:
            background_r = background_r[:, 0]

        basename = base_name(os.path.basename(file_pair[0]))
        # If environment already has speech, skip mixing and treat as background-only.
        # e.g. "CocktailParty", "InterfereringSpeakers"
        if environment in ["CocktailParty", "InterfereringSpeakers"]:
            # We return the background as if "no speech" => label 0 or 1 — up to your pipeline
            # For consistency, let's keep it "0 = no speech" or "1 = has speech".
            # But you originally used "return [bgL, bgR], environment, recsit, ... label=1"
            # We'll assume label=0 means background-only, label=1 means speech present
            # so maybe we do label=0 here, or preserve your earlier usage.
            # We'll do label=0 to indicate no added speech.
            return [background_l, background_r], None, environment, recsit, cut_id, (basename, None), 0

        # Otherwise, we do the usual mixing
        snr = random.choice(self.snr_levels)
        combined_speech, speech_used = self._get_combined_speech()

        # For stereo "clean", you might replicate the same speech on both channels if desired:
        speech_l, bg_chunk_l = ensure_valid_lengths_with_speech(combined_speech, background_l)
        speech_r, bg_chunk_r = ensure_valid_lengths_with_speech(combined_speech, background_r)

        # The “clean” waveforms are what we’d feed to the MSE as the target
        clean_l, clean_r = speech_l.copy(), speech_r.copy()

        # Now mix them at the chosen snr
        mixed_l = mix_signals(speech_l, bg_chunk_l, snr)
        mixed_r = mix_signals(speech_r, bg_chunk_r, snr)

        environment = f"SpeechIn_{environment}"

        # Return: (noisy, clean, env, recsit, cut_id, extra_info, label=1)
        # Because we intentionally added speech:
        return (
            [mixed_l, mixed_r],                # noisy mixture
            [clean_l, clean_r],                # clean speech
            environment, 
            recsit, 
            cut_id, 
            (basename, speech_used),
            snr
        )

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
    for file_pair, environment, recsit, cut_id in background_ds.audio_files:
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


###############################################################################
# Main function that performs splitting and creates DataLoaders.
###############################################################################
def main():
    # Replace these paths with your actual directories.
    ROOT_DIR = '/Users/nkdem/Downloads/HEAR-DS'
    CHIME_DIR = '/Volumes/SSD/Datasets/CHiME3/CHiME3-Isolated-DEV/dt05_bth'
    
    # Create the full datasets.
    full_background_ds = BackgroundDataset(ROOT_DIR)
    full_speech_ds = SpeechDataset(CHIME_DIR)
    
    # Split the background dataset based on recsit.
    train_background_ds, test_background_ds, train_background_speech_ds, test_background_speech_ds = split_background_dataset(full_background_ds)
    
    # Split the speech dataset based on speaker.
    train_speech_ds, test_speech_ds = split_speech_dataset(full_speech_ds)
    
    logger.info(f"Train background samples: {len(train_background_ds)}; Test background samples: {len(test_background_ds)}")
    logger.info(f"Train speech samples: {len(train_speech_ds)}; Test speech samples: {len(test_speech_ds)}")
    
    # Create Mixed Audio Datasets for training and test sets.
    train_mixed_ds = MixedAudioDataset(train_background_speech_ds, train_speech_ds)
    test_mixed_ds = MixedAudioDataset(test_background_speech_ds, test_speech_ds)
    
    train_combined = torch.utils.data.ConcatDataset([train_background_ds, train_mixed_ds])
    test_combined = torch.utils.data.ConcatDataset([test_background_ds, test_mixed_ds])

    def collate_fn(batch):
        mixed_audios, environments, recsits, cut_ids, labels = zip(*batch)
        mixed_audios = [[sf.read(mixed_audios[i][0]), sf.read(mixed_audios[i][1])] if labels[i] == 0 else mixed_audios[i] for i in range(len(mixed_audios))]
        return mixed_audios, environments, recsits, cut_ids, labels
    train_loader = DataLoader(train_combined, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_combined, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # Demonstrate by iterating over one batch from each loader.
    count = 0
    for batch in train_loader:
        mixed_audios, environment, recsit, cut_id, labels = batch
        logger.info(f"Train Mixed Audio Batch Labels: {labels}")
    for batch in test_loader:
        mixed_audios, environment, recsit, cut_id, labels = batch
        logger.info(f"Test Mixed Audio Batch Labels: {labels}")

if __name__ == '__main__':
    main()
