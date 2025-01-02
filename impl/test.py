import os

import torch
from tqdm import tqdm
from constants import MODELS
from hear_ds import HEARDS
import logging
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
                
                logger.debug(f"No. of test data: {len(test_data)}")

                # read int_to_label mapping
                DIR = os.path.join(os.getcwd(), model_name, 'int_to_label.txt')
                with open(DIR, 'r') as f:
                    int_to_label = f.readlines()
                    int_to_label = [line.strip().split() for line in int_to_label]
                    int_to_label = {int(line[0]): line[1] for line in int_to_label}
                    logger.debug(f"int_to_label mapping: {int_to_label}")


                test_dataset = HEARDS('/Users/nkdem/Downloads/HEAR-DS', test_data, int_to_label)
                test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

                correct = 0 
                total = 0
                
                # store confusion matrix
                confusion_matrix = torch.zeros(num_classes, num_classes).to(device)

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
                        for t, p in zip(labels.view(-1), predicted.view(-1)):
                            confusion_matrix[t.long(), p.long()] += 1
                accuracy = correct / total
                accuracy = accuracy * 100
                logger.info(f"Accuracy: {accuracy}%")
            logger.info(f"Testing model {model_name} is done")
            
            # plot confusion matrix
# plot confusion matrix
            confusion_matrix = confusion_matrix.cpu().numpy()
            plt.figure(figsize=(10, 10))

            # Create the heatmap with centered annotations
            sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', 
                        annot_kws={"size": 12, "va": "center", "ha": "center"})  # Adjust the size and alignment

            # Set x and y ticks before saving the figure
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.xticks(np.arange(num_classes) + 0.5, [int_to_label[i] for i in range(num_classes)], rotation=90)  # Shift x ticks
            plt.yticks(np.arange(num_classes) + 0.5, [int_to_label[i] for i in range(num_classes)], rotation=0)  # Shift y ticks
            plt.title(f"Confusion matrix of {model_name} (Accuracy: {accuracy:.2f}%)")

            # Save the figure after setting the ticks
            plt.savefig(f"{model_name}/confusion_matrix.png")
            plt.close() 

        else:
            logger.warning(f"Model {model_name} does not exist")
