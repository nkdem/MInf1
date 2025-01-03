import os

import torch
from tqdm import tqdm
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

def test(dataset: HEARDS, base_dir,model_name, cnn1_channels, cnn2_channels, fc_neurons):
    num_classes = dataset.get_num_classes()
    DIR = os.path.join(base_dir)
    if os.path.exists(DIR):
        logger.info(f"Testing model {model_name}")
        logger.info(f"Model parameters: {cnn1_channels}, {cnn2_channels}, {fc_neurons}")
        logger.info("Reading test data")
        with open(os.path.join(DIR, 'test_files.txt'), 'r') as f:
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
            with open(os.path.join(DIR, 'int_to_label.txt'), 'r') as f:
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
            model_path = os.path.join(DIR, 'model.pth')
            model.load_state_dict(torch.load(model_path, weights_only=True))

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
        

        # Normalize the confusion matrix
        confusion_matrix = confusion_matrix.cpu().numpy()
        confusion_matrix_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

        # Create figure with specific size ratio and extra space at bottom
        plt.figure(figsize=(12, 10))  # Made figure taller to accommodate labels

        # Create the heatmap
        sns.heatmap(confusion_matrix_normalized, 
                    annot=False,
                    cmap='YlOrBr_r',
                    vmin=0.0,
                    vmax=1.0,
                    square=False,
                    linewidths=0.5,
                    linecolor='black',
                    cbar_kws={
                        'label': '',
                        'orientation': 'horizontal',
                        'pad': 0.2,
                        'ticks': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    })

        # Add x-axis labels with better formatting
        plt.xticks(np.arange(num_classes) + 0.5, [int_to_label[i] for i in range(num_classes)], 
                rotation=45,  # Changed to 45 degrees
                ha='right',   # Align the labels
                rotation_mode='anchor')  # Better rotation alignment


        # Adjust y-axis labels
        plt.yticks(np.arange(num_classes) + 0.5, [int_to_label[i] for i in range(num_classes)], rotation=0)
        plt.ylabel('True Scene')

        # Set title
        plt.title(f"Normalised Confusion Matrix of {model_name} (Accuracy: {accuracy:.2f}%)")

        # Adjust layout with more bottom space
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)  # Add extra space at bottom for labels

        # Move colorbar down
        ax = plt.gca()
        ax.set_xlabel('Estimated Scene')
        colorbar = ax.collections[0].colorbar
        colorbar.ax.set_position([colorbar.ax.get_position().x0, 
                                colorbar.ax.get_position().y0 - 0.05,
                                colorbar.ax.get_position().width,
                                colorbar.ax.get_position().height])

        # Save the figure
        plt.savefig(f"{DIR}/confusion_matrix.png", 
                    bbox_inches='tight', 
                    dpi=300)
        plt.close()

    else:
        logger.warning(f"Model {model_name} does not exist")