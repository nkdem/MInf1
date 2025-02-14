import copy
import gc
import json
import math
import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.optim.sgd
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import MODELS
from hear_ds import HEARDS 
from helpers import get_truly_random_seed_through_os, seed_everything
from models import AudioCNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def loss_fn(weights, outputs, targets):
    return nn.CrossEntropyLoss(weights)(outputs, targets)


class BaseTrainer:
    def __init__(self, root_dir, dataset: HEARDS, base_dir, num_epochs, batch_size, cuda=False, split=True):
        self.root_dir = root_dir
        self.dataset = dataset
        self.base_dir = base_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.cuda = cuda
        self.split = split

        self.device = torch.device("mps" if not self.cuda else "cuda")
        logger.info(f"Using device: {self.device}")
        self.seed_val = get_truly_random_seed_through_os()
        seed_everything(self.seed_val)
        logger.info(f"Random seed: {self.seed_val}")

        if self.split:
            self.dataset.split_dataset()
        self.train_data = self.dataset.get_train_data()
        self.test_data = self.dataset.get_test_data()
        self.num_of_classes = self.dataset.get_num_classes()
        self.models_to_train = MODELS

    def save_file_lists(self, DIR_TO_SAVE):
        # Save training metadata such as audio file paths and label maps.
        with open(os.path.join(DIR_TO_SAVE, "train_files.txt"), "w") as f:
            for audio_file in self.train_data:
                f.write(f"{audio_file[0]}\n")
        with open(os.path.join(DIR_TO_SAVE, "test_files.txt"), "w") as f:
            for audio_file in self.test_data:
                f.write(f"{audio_file[0]}\n")
        with open(os.path.join(DIR_TO_SAVE, "int_to_label.txt"), "w") as f:
            for int_label, label in self.dataset.int_to_label.items():
                f.write(f"{int_label} {label}\n")

    def save_speech_metadata(self, DIR_TO_SAVE, dataset):
        # save dataset.speech_mapping and dataset.speech_samples
        # the first one is a dictionary of snr keys, which itself is a dictionary of base background samples, and the value is a SpeechSample class which contains speech_files used
        # save a json file 
        # {
        #   "background": {
                # speech_files_used
            # }
        # }
        with open(os.path.join(DIR_TO_SAVE, "speech_metadata.json"), "w") as f:
            samples = {}
            for snr in dataset.speech_mapping:
                for base in dataset.speech_mapping[snr]:
                    samples[base] = dataset.speech_mapping[snr][base].speech_files_used
            json.dump(samples, f)


    def save_metadata(self, DIR_TO_SAVE, losses, start_time, end_time, learning_rates, extra_meta: dict):
        # Save losses, duration, and extra metadata information.
        with open(os.path.join(DIR_TO_SAVE, "losses.txt"), "w") as f:
            for loss_val in losses:
                f.write(f"{loss_val}\n")
        with open(os.path.join(DIR_TO_SAVE, "learning_rates.txt"), "w") as f:
            for lr in learning_rates:
                f.write(f"{lr}\n")
        with open(os.path.join(DIR_TO_SAVE, "duration.txt"), "w") as f:
            f.write(f"{end_time - start_time}\n")
        meta_data = {
            "Seed": self.seed_val,
            "Batch size": self.batch_size,
            "Number of classes": self.num_of_classes,
            "Number of epochs": self.num_epochs,
        }
        meta_data.update(extra_meta)
        with open(os.path.join(DIR_TO_SAVE, "metadata.txt"), "w") as f:
            for k, v in meta_data.items():
                f.write(f"{k}: {v}\n")

    def prepare_directory(self, model_name):
        model_dir = os.path.join(self.base_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.save_file_lists(model_dir)
        return model_dir

    def train(self):
        raise NotImplementedError("Subclasses must implement this method!")


class AdamEarlyStopTrainer(BaseTrainer):
    def __init__(self, root_dir, dataset: HEARDS, base_dir, num_epochs, batch_size,
                 initial_lr=1e-3, cuda=False, split=True, early_stop_threshold=1e-4, patience=5):
        super().__init__(root_dir, dataset, base_dir, num_epochs, batch_size, cuda, split)
        self.initial_lr = initial_lr
        self.early_stop_threshold = early_stop_threshold
        self.patience = patience

    def train(self):
        for model_name, (cnn1_channels, cnn2_channels, fc_neurons) in self.models_to_train.items():
            model_dir = self.prepare_directory(model_name)

            # copy dataset
            train_dataset = copy.deepcopy(self.dataset)
            train_dataset.audio_files = self.train_data

            # train_dataset = HEARDS(
            #     root_dir=self.root_dir, 
            #     audio_files=self.train_data, 
            #     feature_cache=self.dataset.feature_cache,
            #     cuda=self.cuda, 
            # )
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            losses = []
            start_time = time.time()

            model = AudioCNN(self.num_of_classes, cnn1_channels, cnn2_channels, fc_neurons).to(self.device)
            optimiser = torch.optim.Adam(model.parameters(), lr=self.initial_lr)

            # Early stopping variables
            best_loss = float('inf')
            epochs_no_improve = 0
            learning_rates = [self.initial_lr]  # To track the learning rate changes

            for epoch in range(self.num_epochs):
                model.train()
                running_loss = 0.0

                for batch in tqdm(
                    train_loader,
                    desc=f"Training {model_name} [Epoch {epoch + 1}/{self.num_epochs}] [LR: {optimiser.param_groups[0]['lr']}]",
                    unit="batch"
                ):
                    learning_rates.append(optimiser.param_groups[0]['lr'])
                    pairs, _, labels = batch
                        
                        # Build the full batch
                    num_samples = len(pairs[0])

                    samples = []
                    for i in range(num_samples):
                        # Build the sample paths for both channels
                        sample_paths = [pairs[0][i], pairs[1][i]]
                        # Get the mel spectrogram (expected to be a tensor of shape [1, 40, 501])
                        sample = train_dataset.get_mel(sample_paths, labels[i], train=True)
                        samples.append(sample)

                    # Stack all samples to form a batch with shape: [batch_size, 1, 40, 501]
                    logmels = torch.stack(samples, dim=0)

                    actual_labels = []
                    for i in range(num_samples):
                        label = labels[i]
                        if label == 'InterfereringSpeakers' or label == 'CocktailParty':
                            actual_labels.append(label)
                        else:
                            if train_dataset.is_pair_speech(pairs[0][i]):
                                actual_labels.append(f'{label}_speech')
                            else:
                                actual_labels.append(label)
                    labels_to_int = [train_dataset.label_to_int[label] for label in actual_labels]
                    labels = torch.tensor(labels_to_int, dtype=torch.long)
                    labels = labels.to(self.device)

                    # Get model outputs; now logmels has the correct batch size of 32
                    outputs = model(logmels)

                    # Compute loss with batch size matching
                    loss = loss_fn(train_dataset.weights, outputs, labels)

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                    running_loss += loss.item()

                epoch_loss = running_loss / len(train_loader)
                logger.info(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")
                losses.append(epoch_loss)

                # Early stopping logic
                if best_loss - epoch_loss > self.early_stop_threshold:
                    best_loss = epoch_loss
                    epochs_no_improve = 0
                    logger.info(f"Epoch {epoch + 1}: Loss improved to {epoch_loss:.4f}")
                else:
                    epochs_no_improve += 1
                    logger.info(f"Epoch {epoch + 1}: No significant improvement (best loss: {best_loss:.4f}), count: {epochs_no_improve}/{self.patience}")

                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                    break

            end_time = time.time()
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
            extra_meta = {
                "Final loss": epoch_loss,
                "Model parameters": str(model),
            }
            self.save_metadata(model_dir, losses, start_time, end_time, learning_rates, extra_meta)
            self.save_speech_metadata(model_dir, train_dataset)

            # Clean-up
            self.dataset.feature_cache = train_dataset.feature_cache
            del train_loader
            if hasattr(train_dataset, 'feature_cache'):
                train_dataset.feature_cache = None
            del train_dataset
            if self.cuda:
                torch.cuda.empty_cache()
            gc.collect()

            print(f"Model {model_name} saved with Adam optimiser and early stopping.")


class OneCycleTrainer(BaseTrainer):
    def __init__(self, root_dir, dataset: HEARDS, base_dir, num_epochs, batch_size,
                 max_lr, cuda=False, split=True):
        super().__init__(root_dir, dataset, base_dir, num_epochs, batch_size, cuda, split)
        self.max_lr = max_lr

    def train(self):
        for model_name, (cnn1_channels, cnn2_channels, fc_neurons) in self.models_to_train.items():
            model_dir = self.prepare_directory(model_name)

            train_dataset = HEARDS(
                self.root_dir, 
                self.train_data, 
                feature_cache=self.dataset.feature_cache, 
                cuda=self.cuda, 
            )
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            weights = train_dataset.get_weights()

            losses = []
            learning_rates = [self.max_lr]  # To track the learning rate changes
            start_time = time.time()

            model = AudioCNN(self.num_of_classes, cnn1_channels, cnn2_channels, fc_neurons).to(self.device)
            optimiser = torch.optim.SGD(model.parameters(), lr=self.max_lr)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimiser,
                max_lr=self.max_lr,
                epochs=self.num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=1e4
            )
            # Optionally, clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            for epoch in range(self.num_epochs):
                model.train()
                running_loss = 0.0

                for _, logmel, labels, _ in tqdm(
                    train_loader,
                    desc=f"Training {model_name} [Epoch {epoch + 1}/{self.num_epochs}] [LR: {optimiser.param_groups[0]['lr']}]",
                    unit="batch"
                ):
                    logmel, labels = logmel.to(self.device), labels.to(self.device)
                    outputs = model(logmel)
                    loss = loss_fn(weights, outputs, labels)

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()
                    scheduler.step()

                    running_loss += loss.item()
                    learning_rates.append(optimiser.param_groups[0]['lr'])

                epoch_loss = running_loss / len(train_loader)
                logger.info(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")
                losses.append(epoch_loss)

            end_time = time.time()
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
            extra_meta = {
                "Initial learning rate": self.max_lr,
                "Learning rates": learning_rates,
                "Final loss": epoch_loss,
                "Weights": weights,
                "Model parameters": str(model)
            }
            self.save_metadata(model_dir, losses, start_time, end_time, extra_meta)
            self.save_speech_metadata(model_dir, train_dataset)

            del train_loader
            if hasattr(train_dataset, 'feature_cache'):
                train_dataset.feature_cache = None
            del train_dataset
            if self.cuda:
                torch.cuda.empty_cache()
            gc.collect()

            print(f"Model {model_name} saved with OneCycleLR training.")


class FixedLRTrainer(BaseTrainer):
    def __init__(self, root_dir, dataset: HEARDS, base_dir, num_epochs, batch_size,
                 learning_rates, cuda=False, split=True):
        super().__init__(root_dir, dataset, base_dir, num_epochs, batch_size, cuda, split)
        self.learning_rates = learning_rates

    def train(self):
        for model_name, (cnn1_channels, cnn2_channels, fc_neurons) in self.models_to_train.items():
            model_dir = self.prepare_directory(model_name)

            train_dataset = HEARDS(
                self.root_dir, 
                self.train_data, 
                feature_cache=self.dataset.feature_cache, 
                cuda=self.cuda, 
            )
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            weights = train_dataset.get_weights()

            losses = []
            start_time = time.time()
            model = AudioCNN(self.num_of_classes, cnn1_channels, cnn2_channels, fc_neurons).to(self.device)
            optimiser = torch.optim.SGD(model.parameters(), lr=self.learning_rates[0])

            # Determine when to change the learning rate, evenly splitting the epochs.
            lr_change_epoch = math.ceil(self.num_epochs / len(self.learning_rates))

            for epoch in range(self.num_epochs):
                # Update the learning rate at fixed epochs
                if ((epoch + 1) % lr_change_epoch == 0) and (epoch + 1 != self.num_epochs):
                    index = (epoch + 1) // lr_change_epoch
                    optimiser.param_groups[0]['lr'] = self.learning_rates[index]
                    logger.info(f"Learning rate changed to {optimiser.param_groups[0]['lr']} at epoch {epoch + 1}")

                model.train()
                running_loss = 0.0

                for _, logmel, labels in tqdm(
                    train_loader,
                    desc=f"Training {model_name} [Epoch {epoch + 1}/{self.num_epochs}] [LR: {optimiser.param_groups[0]['lr']}]",
                    unit="batch"
                ):
                    logmel, labels = logmel.to(self.device), labels.to(self.device)
                    outputs = model(logmel)
                    loss = loss_fn(weights, outputs, labels)

                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                    running_loss += loss.item()

                epoch_loss = running_loss / len(train_loader)
                logger.info(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")
                losses.append(epoch_loss)

            end_time = time.time()
            torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
            extra_meta = {
                "Initial learning rate": self.learning_rates[0],
                "Learning rates": self.learning_rates,
                "Final loss": epoch_loss,
                "Weights": weights,
                "Model parameters": str(model)
            }
            self.save_metadata(model_dir, losses, start_time, end_time, extra_meta)
            self.save_speech_metadata(model_dir, train_dataset)

            del train_loader
            if hasattr(train_dataset, 'feature_cache'):
                train_dataset.feature_cache = None
            del train_dataset
            if self.cuda:
                torch.cuda.empty_cache()
            gc.collect()

            print(f"Model {model_name} saved with fixed learning rate scheduling.")