import os
import logging
from torch.utils.data import Dataset

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
        return pairs, recsit, label  


# Example usage
# dataset = HEARDS('/path/to/dataset')
# print(len(dataset))
# print(dataset[0])  # Access the first audio file and its label

if __name__ == '__main__':
    dataset = HEARDS('/Users/nkdem/Downloads/HEAR-DS')
    print(len(dataset))
    print(dataset[0])  # Access the first audio file and its label