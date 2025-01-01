import os
import logging
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
import torchaudio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


"""
https://www.hz-ol.de/en/hear-ds.html
10.1109/ICASSP40776.2020.9053611:
Hearing Aid Research Data Set for Acoustic Environment Recognition https://ieeexplore.ieee.org/document/9053611
(Andreas Hüwel, Dr. Kamil Adiloğlu and Dr. Jörg-Hendrik Bach), published at ICASSP2020
"""
class HEARDS(Dataset):
    """
    Args:
        root_dir (str): Root directory of dataset 
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.audio_files = self._get_all_audio_files()
        self.feature_dir = os.path.join(self.root_dir, 'features')
        try:
            os.makedirs(self.feature_dir, exist_ok=True)
        except OSError as e:
            logger.error(f'Error creating feature directory: {e}')
    """
    Returns:
        A list of tuples where each tuple contains the paths to the left and right channel audio files and the corresponding label (environment in the context of HEARDS)
    """
    def _get_all_audio_files(self):
        audio_files = []
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.wav'):
                    # Extract the environment from the directory structure
                    root_split = root.split('/')
                    relative_diff = len(root_split) - len(self.root_dir.split('/'))
                    environment = root_split[-relative_diff]

                    if relative_diff == 3:
                        # contains speech samples from various SNR levels
                        # lets append environment with _speech
                        environment = environment + '_speech'
                    recsit = file.split('_')[1]  # Assuming RECSIT is extracted from the filename

                    # Create the full file path
                    file_path = os.path.join(root, file)

                    # Add the file path and its corresponding label (environment) to the list
                    # audio_files.append((file_path, environment))

                    # Log if the file is a left or right channel
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
        return pairs, label  
    
    def _basename(self, pairs):
        # get everything before .wav 
        # check the depth of the file
        depth = len(pairs[0].split('/')) - len(self.root_dir.split('/'))
        if depth == 4:
            # contains speech samples from various SNR levels
            # get the directory above name
            snr = pairs[0].split('/')[-2]
            return os.path.basename(pairs[0]).split('.wav')[0] + f'_{snr}'
        return os.path.basename(pairs[0]).split('.wav')[0]

    """
    Args:
        file_L (str): Path to the left channel audio file
        file_R (str): Path to the right channel audio file
        n_mels (int): Number of mel bands to generate
        n_fft (int): Number of FFT points
        hop (int): Hop length in ms
    """
    def _get_mfcc(self, file_L, file_R, n_mels=40, n_fft=1024, hop=20):
        # check if the feature has already been computed
        basename = self._basename([file_L, file_R])
        feature_file = os.path.join(self.feature_dir, f'{basename}.pt')
        if os.path.exists(feature_file):
            return torch.load(feature_file)
        else:
            logger.info(f'Computing MFCC for {basename}')
            waveform_L, sample_rate_L = torchaudio.load(file_L)
            waveform_R, sample_rate_R = torchaudio.load(file_R)

            if sample_rate_L != sample_rate_R:
                logger.error(f'Sample rates do not match: left channel: {sample_rate_L}, right channel: {sample_rate_R}')
                return None


            hop_length = int(sample_rate_L * hop / 1000) # convert ms to samples

            mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate_L,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )
            
            mel_L = mel_transform(waveform_L)
            mel_R = mel_transform(waveform_R)

            logmel_L = 20 * torch.log10(mel_L + 1e-10)
            logmel_R = 20 * torch.log10(mel_R + 1e-10)

            logmel_mean = (logmel_L + logmel_R) / 2

            logmel_mean = logmel_mean.view(1, n_mels, -1) # Shape will be (1, n_mels, n_frames)

            torch.save(logmel_mean, feature_file)

            return logmel_mean
    
    def get_MFCC(self, idx):
        pairs, recsit, label = self.audio_files[idx]
        return self._get_mfcc(pairs[0], pairs[1])


if __name__ == '__main__':
    dataset = HEARDS('/Users/nkdem/Downloads/HEAR-DS')
    print(len(dataset))
    print(dataset[0])  # Access the first audio file and its label