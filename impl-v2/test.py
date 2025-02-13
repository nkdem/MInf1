import os
import time

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


def test(dataset: HEARDS, base_dir, model_name, cnn1_channels, cnn2_channels, fc_neurons, samples_per_class=None, cuda=False):
    """
    Test the model on the dataset.
    
    Args:
        dataset (HEARDS): The dataset object
        base_dir (str): Base directory path
        model_name (str): Name of the model
        cnn1_channels (int): Number of channels in first CNN layer
        cnn2_channels (int): Number of channels in second CNN layer
        fc_neurons (int): Number of neurons in fully connected layer
        samples_per_class (int, optional): Number of samples to test per class. 
                                         If None, uses all available samples.
    """
    device = torch.device("mps" if not cuda else "cuda")
    logger.info(f"Using device: {device}")
    num_classes = dataset.get_num_classes()
    DIR = os.path.join(base_dir)
    if os.path.exists(DIR):
        logger.info(f"Testing model {model_name}")
        logger.info(f"Model parameters: {cnn1_channels}, {cnn2_channels}, {fc_neurons}")
        logger.info(f"Samples per class: {'all' if samples_per_class is None else samples_per_class}")
        logger.info("Reading test data")
        
        with open(os.path.join(DIR, 'test_files.txt'), 'r') as f:
            test_files = f.readlines()
            test_files = [line.strip().split() for line in test_files]
            test_files = [line[0][2:-2] for line in test_files]
            
            # test_files = [file.replace('/home/s2203859/HEAR-DS', '/Users/nkdem/Downloads/HEAR-DS') for file in test_files]
            with open(os.path.join(DIR, 'int_to_label.txt'), 'r') as f:
                int_to_label = f.readlines()
                int_to_label = [line.strip().split() for line in int_to_label]
                int_to_label = {int(line[0]): line[1] for line in int_to_label}
                logger.debug(f"int_to_label mapping: {int_to_label}")
            label_to_int = {v: k for k, v in int_to_label.items()}

            # Initialize counters for each class
            class_counts = {i: 0 for i in range(num_classes)}
            test_data = []
            
            # Collect samples
            for i in tqdm(range(len(dataset)), desc="Extracting test data"):
                audio_file = dataset.get_audio_file(i)
                pair = audio_file[0]
                label = label_to_int[audio_file[2]]
                
                if pair[0] in test_files:
                    if samples_per_class is None or class_counts[label] < samples_per_class:
                        test_data.append(audio_file)
                        class_counts[label] += 1
                    
                    # Check if we have enough samples for all classes
                    if samples_per_class is not None and all(count >= samples_per_class for count in class_counts.values()):
                        break
            
            logger.debug(f"No. of test data: {len(test_data)}")
            logger.debug(f"Samples per class: {class_counts}")

            test_dataset = HEARDS(dataset.root_dir, test_data, int_to_label, feature_cache=dataset.feature_cache, cuda=cuda)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

            correct = 0 
            total = 0
            
            # store confusion matrix
            confusion_matrix = torch.zeros(num_classes, num_classes).to(device)

            logger.info("Loading model")
            model = AudioCNN(num_classes, cnn1_channels, cnn2_channels, fc_neurons).to(device)
            model_path = os.path.join(DIR, 'model.pth')
            model.load_state_dict((torch.load(model_path, weights_only=True)))

            model.eval()
            logger.info("Testing model")
            with torch.no_grad():
                for audio, logmels, labels in tqdm(test_loader, desc="Testing", unit="batch"):
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
        sampling_info = f"({samples_per_class} samples/class)" if samples_per_class is not None else "(all samples)"
        plt.title(f"Normalised Confusion Matrix of {model_name} {sampling_info}\n(Accuracy: {accuracy:.2f}%)")

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

        return confusion_matrix, accuracy

    else:
        logger.warning(f"Model {model_name} does not exist")
