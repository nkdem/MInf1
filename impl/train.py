import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim.sgd
from torch.utils.data import DataLoader
import logging

from tqdm import tqdm

from constants import MODELS
from hear_ds import HEARDS
from helpers import get_truly_random_seed_through_os, seed_everything
from models import AudioCNN 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def loss_fn(weights, outputs, targets):
    return nn.CrossEntropyLoss(weights)(outputs, targets)

def train(dataset: HEARDS, base_dir,num_epochs, batch_size, max_lr=None, learning_rates = None, cuda=False):
    device = torch.device("mps" if not cuda else "cuda")
    logger.info(f"Using device: {device}")
    number = get_truly_random_seed_through_os()
    seed_everything(number)
    logger.info(f"Random seed: {number}")

    dataset.split_dataset()

    train_data = dataset.get_train_data()
    test_data = dataset.get_test_data()

    root_dir = '/Users/nkdem/Downloads/HEAR-DS' if not cuda else '/home/s2203859/minf-1/dataset'
    train_dataset = HEARDS(root_dir, train_data, feature_cache=dataset.feature_cache, cuda=cuda)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_of_classes = dataset.get_num_classes()
    models_to_train = MODELS

    weights = train_dataset.get_weights()

    # create directory to save the model
    for model_name, (cnn1_channels, cnn2_channels, fc_neurons) in models_to_train.items():
        if not os.path.exists(os.path.join(base_dir, model_name)):
            os.makedirs(os.path.join(base_dir, model_name))
    for model_name, (cnn1_channels, cnn2_channels, fc_neurons) in models_to_train.items():
        # store the losses at each epoch
        losses = []
        # store how long it took to train the model (get start time)
        start_time = time.time()

        model = AudioCNN(num_of_classes, cnn1_channels, cnn2_channels, fc_neurons).to(device)

        DIR_TO_SAVE = os.path.join(base_dir, model_name)

        # save the paths of audio files used for training and testing
        with open(f'{DIR_TO_SAVE}/train_files.txt', 'w') as f:
            for audio_file in train_data:
                f.write(f'{audio_file[0]}\n')
        with open(f'{DIR_TO_SAVE}/test_files.txt', 'w') as f:
            for audio_file in test_data:
                f.write(f'{audio_file[0]}\n')
        with open(f'{DIR_TO_SAVE}/int_to_label.txt', 'w') as f:
            for int_label, label in dataset.int_to_label.items():
                f.write(f'{int_label} {label}\n')

        # metadata

        if max_lr is not None:
            optimiser = torch.optim.SGD(model.parameters(), lr=max_lr)
            learning_rates = []
            learning_rates.append(max_lr)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimiser,
                max_lr=max_lr,
                epochs=num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=1e4
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model.train() # Set the model to training mode
            running_loss = 0.0
            for epoch in range(num_epochs):
                for _, logmel, labels in tqdm(train_loader, desc=f'Training {model_name} [Epoch {epoch + 1}/{num_epochs}] [LR: {optimiser.param_groups[0]["lr"]}]', unit='batch'):
                    logmel, labels = logmel.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs = model(logmel)
                    loss = loss_fn(weights, outputs, labels)
                    
                    # Backward pass and optimization
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    scheduler.step()
                    
                    running_loss += loss.item()
                epoch_loss = running_loss / len(train_loader)
                logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
                losses.append(epoch_loss)
                learning_rates.append(optimiser.param_groups[0]['lr'])
        elif learning_rates is not None:
            print("Learning rates are provided")
            optimiser = torch.optim.SGD(model.parameters(), lr=learning_rates[0])
            lr_change_epoch = math.ceil(num_epochs / (len(learning_rates)))
            for epoch in range(num_epochs):
                # change optimiser param group lr if epoch is divisible by lr_change_epoch
                if ((epoch + 1) % lr_change_epoch == 0) and (epoch + 1 != num_epochs):
                    index = (epoch + 1) // lr_change_epoch
                    optimiser.param_groups[0]['lr'] = learning_rates[index] 
                    logger.info(f"Learning rate changed to {optimiser.param_groups[0]['lr']}")
                model.train() # Set the model to training mode
                running_loss = 0.0
                for _, logmel, labels in tqdm(train_loader, desc=f'Training {model_name} [Epoch {epoch + 1}/{num_epochs}] [LR: {optimiser.param_groups[0]["lr"]}]', unit='batch'):
                    logmel, labels = logmel.to(device), labels.to(device)
                    
                    # Forward pass
                    outputs = model(logmel)
                    loss = loss_fn(weights, outputs, labels)
                    
                    # Backward pass and optimization
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    
                    running_loss += loss.item()
                epoch_loss = running_loss / len(train_loader)
                logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
                losses.append(epoch_loss)
        


        end = time.time()
        torch.save(model.state_dict(), f'{DIR_TO_SAVE}/model.pth')
        # save losses
        with open(f'{DIR_TO_SAVE}/losses.txt', 'w') as f:
            for loss in losses:
                f.write(f'{loss}\n')
        with open(f'{DIR_TO_SAVE}/duration.txt', 'w') as f:
            f.write(f'{end - start_time}\n')
        with open(f'{DIR_TO_SAVE}/metadata.txt', 'w') as f:
            f.write(f'Number of classes: {num_of_classes}\n')
            f.write(f'Number of epochs: {num_epochs}\n')
            f.write(f'Initial learning rate: {max_lr}\n')
            f.write(f'Learning rates: {learning_rates}\n')
            f.write(f'Weights: {weights}\n')
            f.write(f'Model parameters: {model}\n')

        print(f"Model {model_name} saved")