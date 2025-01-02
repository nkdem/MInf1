import os
import time
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

# local machine is a M4 chip, and uni cluster are cuda machines
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
logger.info(f"Using device: {device}")

def loss_fn(weights, outputs, targets):
    return nn.CrossEntropyLoss(weights)(outputs, targets)

if __name__ == '__main__':

    number = get_truly_random_seed_through_os()
    seed_everything(number)
    logger.info(f"Random seed: {number}")

    dataset = HEARDS('/Users/nkdem/Downloads/HEAR-DS')
    dataset.split_dataset()

    train_data = dataset.get_train_data()
    test_data = dataset.get_test_data()

    train_dataset = HEARDS('/Users/nkdem/Downloads/HEAR-DS', train_data)
    test_dataset = HEARDS('/Users/nkdem/Downloads/HEAR-DS', test_data)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    num_of_classes = dataset.get_num_classes()
    models_to_train = MODELS

    weights = train_dataset.get_weights()

    # create directory to save the model
    for model_name, (cnn1_channels, cnn2_channels, fc_neurons) in models_to_train.items():
        if not os.path.exists(model_name):
            os.makedirs(model_name)
    for model_name, (cnn1_channels, cnn2_channels, fc_neurons) in models_to_train.items():
        model = AudioCNN(num_of_classes, cnn1_channels, cnn2_channels, fc_neurons).to(device)
        initial_lr = 0.05
        optimiser = torch.optim.SGD(model.parameters(), lr=initial_lr)

        learning_rates = [0.05, 0.01, 0.001, 0.0005, 0.0002, 0.0001]
        lr_change_epoch = 40

        DIR_TO_SAVE = model_name

        # save the paths of audio files used for training and testing
        with open(f'{DIR_TO_SAVE}/train_files.txt', 'w') as f:
            for audio_file in train_data:
                f.write(f'{audio_file[0]}\n')
        with open(f'{DIR_TO_SAVE}/test_files.txt', 'w') as f:
            for audio_file in test_data:
                f.write(f'{audio_file[0]}\n')

        # save the int_to_label mapping
        with open(f'{DIR_TO_SAVE}/int_to_label.txt', 'w') as f:
            for int_label, label in dataset.int_to_label.items():
                f.write(f'{int_label} {label}\n')
        
        num_epochs = 280
        for epoch in range(num_epochs):
            if epoch > 0 and epoch % lr_change_epoch == 0:
                if (epoch // lr_change_epoch) - 1 < len(learning_rates):
                    new_lr = learning_rates[(epoch // lr_change_epoch) - 1]
                    for param_group in optimiser.param_groups:
                        param_group['lr'] = new_lr
                    print(f"Learning rate changed to: {new_lr}")
            model.train() # Set the model to training mode
            running_loss = 0.0
            for _,logmel, labels in tqdm(train_loader, desc=f'Training {model_name} [Epoch {epoch + 1}/{num_epochs}]', unit='batch'):
                logmel,labels = logmel.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(logmel)
                loss = loss_fn(weights, outputs, labels)

                # Backward pass and optimization
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                running_loss += loss.item()
            print(f"Training loss: {running_loss / len(train_loader)}")


        torch.save(model.state_dict(), f'{DIR_TO_SAVE}/model.pth')
        print(f"Model {model_name} saved")

                

