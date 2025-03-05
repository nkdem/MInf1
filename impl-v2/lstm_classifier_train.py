import argparse
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import soundfile as sf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from models import LSTMClassifier
from helpers import get_truly_random_seed_through_os, seed_everything, compute_average_logmel
from heards_dataset import MixedAudioDataset, BackgroundDataset, SpeechDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom collate function to handle different data formats
def collate_fn(batch):
    audios, clean, environments, recsits, cut_ids, extra, snrs = zip(*batch)
    processed_audios = []
    for i in range(len(batch)):
        if snrs[i] is None:
            waveform_l, sr = sf.read(batch[i][0][0])
            waveform_r, sr = sf.read(batch[i][0][1]) 
            processed_audios.append([waveform_l, waveform_r])
        else:
            processed_audios.append([batch[i][0][0], batch[i][0][1]])
    return processed_audios, clean, environments, recsits, cut_ids, extra, snrs

class LSTMTrainer:
    def __init__(self, 
                 base_dir: str, 
                 num_epochs: int, 
                 batch_size: int, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader = None,
                 input_size: int = 40,  # Default for mel spectrogram features
                 hidden_size: int = 128, 
                 num_layers: int = 2, 
                 num_classes: int = None,  # Will be determined from data if not provided
                 learning_rate: float = 0.001, 
                 dropout: float = 0.3,
                 weight_decay: float = 1e-5,
                 patience: int = 5,
                 cuda: bool = False):
        """
        Trainer for LSTM classifier model.
        
        Args:
            base_dir: Directory to save model and logs
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data (optional)
            input_size: Feature dimension of input data
            hidden_size: Hidden size of LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of classes to classify
            learning_rate: Learning rate for optimizer
            dropout: Dropout probability
            weight_decay: Weight decay for optimizer
            patience: Patience for early stopping
            cuda: Whether to use CUDA for training
        """
        self.base_dir = base_dir
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.patience = patience
        self.feature_cache = {}
        
        # Determine device
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # Set random seed
        self.seed_val = get_truly_random_seed_through_os()
        seed_everything(self.seed_val)
        logger.info(f"Random seed: {self.seed_val}")
        
        # Determine number of classes if not provided
        if num_classes is None:
            self.envs = {}
            logger.info("Counting number of classes...")
            for batch in tqdm(self.train_loader, desc="Counting classes", unit="batch"):
                if len(batch) == 7:
                    # HEAR-DS format
                    _, _, envs, _, _, _, _ = batch
                    for env in envs:
                        if env not in self.envs:
                            self.envs[env] = len(self.envs)
                else:
                    # TUT format
                    _, labels = batch
                    for label in labels:
                        if label not in self.envs:
                            self.envs[label] = len(self.envs)
            
            self.num_classes = len(self.envs)
            logger.info(f"Detected {self.num_classes} classes: {self.envs}")
        else:
            self.num_classes = num_classes
            self.envs = {f"class_{i}": i for i in range(num_classes)}
        
        # Initialize model
        self.model = LSTMClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout=self.dropout,
            bidirectional=False  # Unidirectional LSTM
        )
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def prepare_directory(self, model_name):
        """Create directory for saving model and logs."""
        model_dir = os.path.join(self.base_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir
    
    def compute_features(self, waveforms, envs=None, recsits=None, cuts=None, snrs=None):
        """
        Compute mel spectrogram features from waveforms.
        
        For LSTM, we need to reshape the features to (batch_size, sequence_length, input_size)
        """
        # Compute log-mel spectrograms
        if isinstance(waveforms, list) and len(waveforms) == 2:
            # Stereo audio (left and right channels)
            left_feats = compute_average_logmel([waveforms[0]], self.device)
            right_feats = compute_average_logmel([waveforms[1]], self.device)
            # Average the features from both channels
            features = (left_feats + right_feats) / 2
        else:
            # Mono audio
            features = compute_average_logmel(waveforms, self.device)
        
        # The features from compute_average_logmel have shape (batch_size, 1, n_mels, n_frames)
        # For LSTM, we need (batch_size, sequence_length, input_size)
        # Where sequence_length = n_frames and input_size = n_mels
        
        # Remove the channel dimension (dim=1)
        features = features.squeeze(1)
        
        # Transpose from (batch_size, n_mels, n_frames) to (batch_size, n_frames, n_mels)
        features = features.transpose(1, 2)
        
        return features
    
    def save_metadata(self, model, losses, start_time, end_time, learning_rates, extra_meta=None):
        """Save model metadata."""
        metadata = {
            "model_type": "LSTMClassifier",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "bidirectional": False,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "seed": self.seed_val,
            "losses": losses,
            "learning_rates": learning_rates,
            "training_time": end_time - start_time,
            "classes": self.envs
        }
        
        if extra_meta:
            metadata.update(extra_meta)
        
        return metadata
    
    def train(self):
        """Train the LSTM model."""
        model_name = f"lstm-{self.hidden_size}-{self.num_layers}-{int(time.time())}"
        model_dir = self.prepare_directory(model_name)
        logger.info(f"Training model: {model_name}")
        
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        train_losses = []
        val_losses = []
        learning_rates = []
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
            for batch in progress_bar:
                if len(batch) == 7:
                    # HEAR-DS format
                    mixed_waveforms, _, envs, _, _, _, _ = batch
                    # Convert environment names to class indices
                    labels = torch.tensor([self.envs[env] for env in envs], dtype=torch.long).to(self.device)
                    # Compute features
                    features = self.compute_features(mixed_waveforms, envs)
                else:
                    # TUT format
                    waveforms, labels_str = batch
                    # Convert labels to indices
                    labels = torch.tensor([self.envs[label] for label in labels_str], dtype=torch.long).to(self.device)
                    # Compute features
                    features = self.compute_features(waveforms)
                
                # Move features to device
                features = features.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                batch_count += 1
                progress_bar.set_postfix({"loss": loss.item()})
            
            # Calculate average training loss for the epoch
            avg_train_loss = epoch_loss / batch_count
            train_losses.append(avg_train_loss)
            learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Train Loss: {avg_train_loss:.4f}")
            
            # Validation phase (if validation loader is provided)
            if self.val_loader:
                self.model.eval()
                val_loss = 0.0
                val_batch_count = 0
                all_preds = []
                all_labels = []
                
                with torch.no_grad():
                    progress_bar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
                    for batch in progress_bar:
                        if len(batch) == 7:
                            # HEAR-DS format
                            mixed_waveforms, _, envs, _, _, _, _ = batch
                            # Convert environment names to class indices
                            labels = torch.tensor([self.envs[env] for env in envs], dtype=torch.long).to(self.device)
                            # Compute features
                            features = self.compute_features(mixed_waveforms, envs)
                        else:
                            # TUT format
                            waveforms, labels_str = batch
                            # Convert labels to indices
                            labels = torch.tensor([self.envs[label] for label in labels_str], dtype=torch.long).to(self.device)
                            # Compute features
                            features = self.compute_features(waveforms)
                        
                        # Move features to device
                        features = features.to(self.device)
                        
                        # Forward pass
                        outputs = self.model(features)
                        loss = self.criterion(outputs, labels)
                        
                        # Get predictions
                        _, preds = torch.max(outputs, 1)
                        
                        # Update statistics
                        val_loss += loss.item()
                        val_batch_count += 1
                        
                        # Store predictions and labels for metrics calculation
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        
                        progress_bar.set_postfix({"val_loss": loss.item()})
                
                # Calculate average validation loss
                avg_val_loss = val_loss / val_batch_count
                val_losses.append(avg_val_loss)
                
                # Calculate metrics
                accuracy = accuracy_score(all_labels, all_preds)
                precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
                recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
                f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
                
                logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Val Loss: {avg_val_loss:.4f}, "
                           f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                           f"Recall: {recall:.4f}, F1: {f1:.4f}")
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model = self.model.state_dict()
                    patience_counter = 0
                    
                    # Save the best model
                    torch.save(self.model.state_dict(), os.path.join(model_dir, "best_model.pth"))
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
            else:
                # If no validation set, save the model after each epoch
                torch.save(self.model.state_dict(), os.path.join(model_dir, f"model_epoch_{epoch+1}.pth"))
        
        end_time = time.time()
        
        # Save the final model if no validation set or if early stopping didn't trigger
        if not self.val_loader or patience_counter < self.patience:
            torch.save(self.model.state_dict(), os.path.join(model_dir, "final_model.pth"))
        
        # If we have a best model from validation, load it
        if best_model is not None:
            self.model.load_state_dict(best_model)
        
        # Save metadata
        metadata = self.save_metadata(
            model=self.model,
            losses={"train": train_losses, "val": val_losses},
            start_time=start_time,
            end_time=end_time,
            learning_rates=learning_rates
        )
        
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            import json
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Training completed. Model saved to {model_dir}")
        return self.model, metadata

def main():
    parser = argparse.ArgumentParser(description="Train LSTM classifier for audio classification")
    parser.add_argument("--split_index", type=int, default=0, help="Index of the split to use (0-4)")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for training")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the split data
    split_file = f'splits/split_{args.split_index}.pkl'
    logger.info(f"Loading split data from {split_file}")
    
    try:
        with open(split_file, 'rb') as f:
            split_data = pickle.load(f)
            
            train_dataset = split_data['train']
            test_dataset = split_data['test']
            
            logger.info(f"Loaded train dataset with {len(train_dataset)} samples")
            logger.info(f"Loaded test dataset with {len(test_dataset)} samples")
            
            # Create data loaders with custom collate function
            train_loader = DataLoader(
                train_dataset, 
                batch_size=args.batch_size, 
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True
            )
            
            test_loader = DataLoader(
                test_dataset, 
                batch_size=args.batch_size, 
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True
            )
            
            # Initialize trainer
            trainer = LSTMTrainer(
                base_dir=args.output_dir,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                train_loader=train_loader,
                val_loader=test_loader,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                learning_rate=args.learning_rate,
                dropout=args.dropout,
                patience=args.patience,
                cuda=args.cuda
            )
            
            # Train model
            model, metadata = trainer.train()
            
            logger.info("Training completed successfully!")
            
    except FileNotFoundError:
        logger.error(f"Split file {split_file} not found. Make sure the splits are generated.")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading split data: {e}")
        exit(1)

if __name__ == "__main__":
    main() 