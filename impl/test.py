import os

import torch
from tqdm import tqdm
from constants import MODELS
from hear_ds import HEARDS
import logging
from torch.utils.data import DataLoader

from models import AudioCNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
logger.info(f"Using device: {device}")

if __name__ == '__main__':
    dataset = HEARDS('/Users/nkdem/Downloads/HEAR-DS')
    num_classes = dataset.get_num_classes()
    models_to_test = MODELS
    for model_name, (cnn1_channels, cnn2_channels, fc_neurons) in models_to_test.items():
        if os.path.exists(model_name):
            logger.info(f"Testing model {model_name}")
            logger.info(f"Model parameters: {cnn1_channels}, {cnn2_channels}, {fc_neurons}")
            logger.info("Reading test data")
            DIR = os.path.join(os.getcwd(), model_name, 'test_files.txt')
            with open(DIR, 'r') as f:
                test_files = f.readlines()
                test_files = [line.strip().split() for line in test_files]
                test_files = [line[0][2:-2] for line in test_files]
                test_data = []
                for i in tqdm(range(len(dataset)), desc="Extracting test data"):
                    audio_file = dataset.get_audio_file(i)
                    pair = audio_file[0]
                    if pair[0] in test_files:
                        test_data.append(audio_file)
                if len(test_data) != len(test_files):
                    logger.warning("Some test data is missing")
                
                logger.info(f"No. of test data: {len(test_data)}")

                # read int_to_label mapping
                DIR = os.path.join(os.getcwd(), model_name, 'int_to_label.txt')
                with open(DIR, 'r') as f:
                    int_to_label = f.readlines()
                    int_to_label = [line.strip().split() for line in int_to_label]
                    int_to_label = {int(line[0]): line[1] for line in int_to_label}
                    logger.info(f"int_to_label mapping: {int_to_label}")


                test_dataset = HEARDS('/Users/nkdem/Downloads/HEAR-DS', test_data, int_to_label)
                test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

                correct = 0 
                total = 0

                logger.info("Loading model")
                model = AudioCNN(num_classes, cnn1_channels, cnn2_channels, fc_neurons).to(device)
                model_path = os.path.join(model_name, 'model.pth')
                model.load_state_dict(torch.load(model_path))

                logger.info("Testing model")
                with torch.no_grad():
                    for _, logmels, labels in tqdm(test_loader, desc="Testing", unit="batch"):
                        logmels = logmels.to(device)
                        labels = labels.to(device)

                        outputs = model(logmels)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted.to(device) == labels).sum().item()
                accuracy = correct / total
                logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Testing model {model_name} is done")
                
        else:
            logger.warning(f"Model {model_name} does not exist")
