import os
import pickle
import sys
import torch
import logging
import numpy as np
import time
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import random

from base_experiment import BaseExperiment
sys.path.append(os.path.abspath(os.path.join('.')))
from models import LSTMClassifier
from classification.train import BaseTrainer
from constants import MODELS
from helpers import compute_average_logmel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LSTMTrainer(BaseTrainer):
    def __init__(self, cuda=False, base_dir=None, train_loader=None, num_epochs=50,
                 input_size=40, hidden_size=128, num_layers=2, learning_rate=0.001,
                 dropout=0.3, weight_decay=1e-5, patience=5, batch_size=32, 
                 use_validation=False, validation_split=0.2):
        # Check if CUDA is actually available when requested
        if cuda and not torch.cuda.is_available():
            logger.warning("CUDA was requested but is not available. Falling back to CPU.")
            cuda = False
        
        super().__init__(base_dir=base_dir, num_epochs=num_epochs, 
                         batch_size=batch_size, train_loader=train_loader, cuda=cuda)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.patience = patience
        self.use_validation = use_validation
        self.validation_split = validation_split
        
        # Create validation loader if validation is enabled
        if self.use_validation and self.train_loader is not None:
            self.create_validation_loader()
        else:
            self.val_loader = None
        
        # Initialize model
        self.model = LSTMClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_of_classes,
            dropout=self.dropout,
            bidirectional=False  # Unidirectional LSTM
        )
        self.model.to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weights)
        
        # For early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # For tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
    
    def create_validation_loader(self):
        """Create a validation loader by splitting the training data."""
        if not hasattr(self.train_loader.dataset, '__len__'):
            logger.warning("Cannot create validation set: dataset does not support length operation")
            self.val_loader = None
            return
            
        # Get the dataset from the train loader
        dataset = self.train_loader.dataset
        
        # Calculate split sizes
        dataset_size = len(dataset)
        val_size = int(self.validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        # Split the dataset
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create new train loader with the reduced dataset
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_loader.collate_fn
        )
        
        # Create validation loader
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.train_loader.collate_fn
        )
        
        logger.info(f"Created validation set with {val_size} samples ({self.validation_split*100:.1f}% of training data)")
        logger.info(f"Remaining training set has {train_size} samples")
    
    def train(self):
        """Train the LSTM model."""
        logger.info(f"Starting LSTM training with {self.num_epochs} epochs")
        start_time = time.time()
        
        # Save class mapping
        model_dir = self.prepare_directory("lstm")
        with open(os.path.join(model_dir, "int_to_label.txt"), "w") as f:
            for env, idx in self.env_to_int.items():
                f.write(f"{env} {idx}\n")
        logger.info(f"Saved class mapping to {os.path.join(model_dir, 'int_to_label.txt')}")
        logger.info(f"Class mapping: {self.env_to_int}")
        
        # Log model architecture
        logger.info(f"LSTM model architecture:")
        logger.info(f"  Input size: {self.input_size}")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Number of layers: {self.num_layers}")
        logger.info(f"  Number of classes: {self.num_of_classes}")
        logger.info(f"  Dropout: {self.dropout}")
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # Training phase
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", unit="batch")
            for batch_idx, batch in enumerate(progress_bar):
                noisy, clean, environments, recsits, cut_id, extra, snr = batch
                
                # Compute features
                features = self.compute_logmels(noisy)
                
                # Forward pass
                self.optimizer.zero_grad()
                labels = torch.tensor([self.env_to_int[env] for env in environments], dtype=torch.long).to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                current_loss = running_loss / (batch_idx + 1)
                current_accuracy = 100 * correct / total if total > 0 else 0
                progress_bar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'acc': f'{current_accuracy:.2f}%'
                })
            
            # Calculate average training loss and accuracy
            avg_train_loss = running_loss / len(self.train_loader)
            train_accuracy = correct / total
            self.train_losses.append(avg_train_loss)
            
            # Log epoch statistics
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            
            # Validation phase
            if self.use_validation and self.val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                # Validation loop
                val_progress = tqdm(self.val_loader, desc="Validation", unit="batch", leave=False)
                with torch.no_grad():
                    for batch in val_progress:
                        noisy, clean, environments, recsits, cut_id, extra, snr = batch
                        
                        # Compute features
                        features = self.compute_logmels(noisy)
                        
                        # Forward pass
                        labels = torch.tensor([self.env_to_int[env] for env in environments], dtype=torch.long).to(self.device)
                        outputs = self.model(features)
                        loss = self.criterion(outputs, labels)
                        
                        # Update statistics
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                        # Update validation progress bar
                        current_val_loss = val_loss / (val_progress.n + 1)
                        current_val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
                        val_progress.set_postfix({
                            'val_loss': f'{current_val_loss:.4f}',
                            'val_acc': f'{current_val_accuracy:.2f}%'
                        })
                
                # Calculate average validation loss and accuracy
                avg_val_loss = val_loss / len(self.val_loader)
                val_accuracy = val_correct / val_total
                self.val_losses.append(avg_val_loss)
                
                # Log validation statistics
                logger.info(f"Epoch {epoch+1}/{self.num_epochs} - "
                            f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
                
                # Early stopping check
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), os.path.join(model_dir, "best_model.pth"))
                    logger.info(f"Epoch {epoch+1}/{self.num_epochs} - New best model saved (val_loss: {avg_val_loss:.4f})")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
            else:
                # If no validation, save model after each epoch
                torch.save(self.model.state_dict(), os.path.join(model_dir, "best_model.pth"))
                logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Model saved")
            
            # Save learning rate
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(model_dir, "final_model.pth"))
        
        # Save metadata
        metadata = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "Number of classes": self.num_of_classes,
            "Class mapping": self.env_to_int,
            "Training duration": time.time() - start_time,
            "Number of epochs": self.num_epochs,
            "Early stopping patience": self.patience,
            "Learning rate": self.learning_rate,
            "Weight decay": self.weight_decay,
            "Use validation": self.use_validation,
            "Validation split": self.validation_split if self.use_validation else None
        }
        
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Model and metadata saved to {model_dir}")
        
        return self.model

    def compute_logmels(self, waveforms, envs=None, recsits=None, cuts=None, snrs=None):
        """
        Compute log-mel spectrograms for a batch of waveforms.
        
        Args:
            waveforms: Batch of audio waveforms (batch_size, channels, samples)
            envs: Optional list of environments
            recsits: Optional list of recording situations
            cuts: Optional list of cut IDs
            snrs: Optional list of SNRs
            
        Returns:
            features: Batch of log-mel spectrograms (batch_size, n_frames, n_mels)
        """
        try:
            # Compute log-mel spectrograms
            features = compute_average_logmel(waveforms, self.device)
            
            # Check for NaN or Inf values
            if torch.isnan(features).any() or torch.isinf(features).any():
                logger.warning(f"NaN or Inf values detected in features! Replacing with zeros.")
                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Remove the channel dimension (dim=1)
            features = features.squeeze(1)
            
            # Transpose from (batch_size, n_mels, n_frames) to (batch_size, n_frames, n_mels)
            features = features.transpose(1, 2)
            
            return features
        except Exception as e:
            logger.error(f"Error computing log-mel spectrograms: {e}")
            # Return a dummy tensor to avoid crashing
            batch_size = waveforms.shape[0]
            return torch.zeros((batch_size, 100, self.input_size), device=self.device)

class LSTMExperiment(BaseExperiment):
    def __init__(self, train_combined, test_combined, num_epochs=50, batch_size=32,
                 input_size=40, hidden_size=128, num_layers=2, learning_rate=0.001,
                 dropout=0.3, weight_decay=1e-5, patience=5, experiment_no=1, cuda=False,
                 use_validation=False, validation_split=0.2):
        # Initialize the base experiment
        super().__init__(batch_size=batch_size, cuda=cuda, 
                         train_combined=train_combined, test_combined=test_combined)
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.patience = patience
        self.exp_no = experiment_no
        self.use_validation = use_validation
        self.validation_split = validation_split
        
        # Check if CUDA is actually available when requested
        if cuda and not torch.cuda.is_available():
            logger.warning("CUDA was requested but is not available. Falling back to CPU.")
            self.cuda = False
        else:
            self.cuda = cuda
            
        self.device = torch.device("cuda" if self.cuda else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Set random seed for reproducibility
        self.seed = random.randint(0, 100000000)
        logger.info(f"Random seed: {self.seed}")
        torch.manual_seed(self.seed)
        if self.cuda:
            torch.cuda.manual_seed(self.seed)
        
        self.experiment_name = f"lstm-classifier-{experiment_no}"

    def run(self):
        """Run the LSTM experiment."""
        print(f"Starting experiment: {self.experiment_name}")
        print(f"Parameters: epochs={self.num_epochs}, batch_size={self.batch_size}")
        print(f"LSTM parameters: hidden_size={self.hidden_size}, num_layers={self.num_layers}")
        print(f"Validation: {'enabled' if self.use_validation else 'disabled'}")

        print(f"\nStarting experiment run {self.exp_no}...")
        base_dir = self.create_experiment_dir(self.experiment_name, self.exp_no)
        
        # Create trainer
        trainer = LSTMTrainer(
            cuda=self.cuda,
            base_dir=base_dir,
            train_loader=self.train_loader,
            num_epochs=self.num_epochs,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            learning_rate=self.learning_rate,
            dropout=self.dropout,
            weight_decay=self.weight_decay,
            patience=self.patience,
            batch_size=self.batch_size,
            use_validation=self.use_validation,
            validation_split=self.validation_split
        )
        
        # Train model
        lstm_model = trainer.train()
        
        print("\nTraining phase completed. Starting results collection and analysis...")
        
        # Override the collect_model_results method
        self.collect_model_results = self.collect_lstm_results.__get__(self, self.__class__)

        # Load the trained LSTM model
        lstm_model = LSTMClassifier(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=trainer.num_of_classes,
            dropout=self.dropout,
            bidirectional=False
        ).to(self.device)
        
        model_path = os.path.join(base_dir, "lstm", "best_model.pth")
        lstm_model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Evaluate the model
        classwise_accuracy, total_accuracy, confusion_matrix = self.collect_model_results(
            test_loader=self.test_loader,
            model=lstm_model,
            no_classes=trainer.num_of_classes,
            env_to_int=trainer.env_to_int
        )
        
        # Create results dictionary
        results = {
            'class_accuracies': {'lstm': [classwise_accuracy]},
            'total_accuracies': {'lstm': [total_accuracy]},
            'confusion_matrix_raw': {'lstm': [confusion_matrix]},
            'trained_models': {'lstm': [lstm_model]},
            'losses': {'lstm': [trainer.train_losses]},
            'duration': {'lstm': [trainer.durations.get('lstm', 0)]},
            'learning_rates': {'lstm': [trainer.learning_rates]}
        }

        # Save results
        with open(os.path.join(base_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        print(f"Results saved in {base_dir}")

        # Print accuracy
        print(f"\nModel: LSTM")
        print(f"Average total accuracy: {total_accuracy:.4f}")
        print(f"Class-wise accuracies: {classwise_accuracy}")
        
        return results

    def __str__(self):
        """String representation of the experiment configuration"""
        return (f"LSTMExperiment("
                f"num_epochs={self.num_epochs}, "
                f"batch_size={self.batch_size}, "
                f"hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers}, "
                f"learning_rate={self.learning_rate}, "
                f"cuda={self.cuda})")

    def get_experiment_config(self):
        return {
            "experiment_type": "LSTM Classifier",
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "learning_rate": self.learning_rate,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "patience": self.patience,
            "cuda": self.cuda,
            "experiment_name": self.experiment_name
        }

    def compute_logmels(self, waveforms, envs=None, recsits=None, cuts=None, snrs=None):
        """
        Compute log-mel spectrograms for a batch of waveforms.
        
        Args:
            waveforms: Batch of audio waveforms (batch_size, channels, samples)
            envs: Optional list of environments
            recsits: Optional list of recording situations
            cuts: Optional list of cut IDs
            snrs: Optional list of SNRs
            
        Returns:
            features: Batch of log-mel spectrograms (batch_size, n_frames, n_mels)
        """
        try:
            # Compute log-mel spectrograms
            features = compute_average_logmel(waveforms, self.device)
            
            # Check for NaN or Inf values
            if torch.isnan(features).any() or torch.isinf(features).any():
                logger.warning(f"NaN or Inf values detected in features! Replacing with zeros.")
                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Remove the channel dimension (dim=1)
            features = features.squeeze(1)
            
            # Transpose from (batch_size, n_mels, n_frames) to (batch_size, n_frames, n_mels)
            features = features.transpose(1, 2)
            
            return features
        except Exception as e:
            logger.error(f"Error computing log-mel spectrograms: {e}")
            # Return a dummy tensor to avoid crashing
            batch_size = waveforms.shape[0]
            return torch.zeros((batch_size, 100, self.input_size), device=self.device)

    def collect_lstm_results(self, test_loader, model, no_classes, env_to_int):
        """
        Evaluate the LSTM model on the test set.
        
        Args:
            test_loader: DataLoader for the test set
            model: Trained LSTM model
            no_classes: Number of classes
            env_to_int: Mapping from environment names to class indices
            
        Returns:
            classwise_accuracy: Accuracy for each class
            total_accuracy: Overall accuracy
            confusion_matrix: Confusion matrix
        """
        model.eval()
        total = 0
        correct = 0
        confusion_matrix = np.zeros((no_classes, no_classes))
        all_predictions = []
        all_labels = []
        
        logger.info(f"Evaluating LSTM model on {len(test_loader)} batches")
        logger.info(f"Number of classes: {no_classes}")
        logger.info(f"Environment to int mapping: {env_to_int}")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing", unit="batch")):
                noisy, clean, environments, recsits, cut_id, extra, snr = batch
                
                # Compute features for LSTM
                try:
                    features = self.compute_logmels(noisy)
                    
                    # Prepare labels
                    try:
                        labels = torch.tensor([env_to_int[env] for env in environments], dtype=torch.long).to(self.device)
                        
                        # Log labels for first batch
                        if batch_idx == 0:
                            logger.info(f"Feature shape: {features.shape}")
                            logger.info(f"Labels shape: {labels.shape}")
                            logger.info(f"Labels: {labels}")
                            logger.info(f"Environments: {environments}")
                        
                        # Forward pass
                        outputs = model(features)
                        _, predicted = torch.max(outputs.data, 1)
                        
                        # Log predictions for first batch
                        if batch_idx == 0:
                            logger.info(f"Predictions: {predicted}")
                            logger.info(f"Correct predictions: {(predicted == labels).sum().item()}/{len(labels)}")
                        
                        # Update statistics
                        total += len(predicted)
                        correct += (predicted == labels).sum().item()
                        all_predictions.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        
                        # Update confusion matrix
                        for i in range(len(predicted)):
                            confusion_matrix[labels[i], predicted[i]] += 1
                    except KeyError as e:
                        logger.error(f"KeyError with environment: {e}")
                        logger.error(f"Available environments: {list(env_to_int.keys())}")
                        logger.error(f"Batch environments: {environments}")
                        continue
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        # Calculate metrics
        if total > 0:
            classwise_accuracy = confusion_matrix.diagonal() / (confusion_matrix.sum(axis=1) + 1e-10)  # Add small epsilon to avoid division by zero
            total_accuracy = correct / total
            
            # Log overall results
            logger.info(f"Total accuracy: {total_accuracy:.4f} ({correct}/{total})")
            logger.info(f"Class-wise accuracies:")
            for i in range(no_classes):
                class_total = confusion_matrix.sum(axis=1)[i]
                if class_total > 0:
                    logger.info(f"  Class {i}: {classwise_accuracy[i]:.4f} ({int(confusion_matrix[i, i])}/{int(class_total)})")
            
            # Log confusion matrix
            logger.info(f"Confusion matrix:")
            for i in range(no_classes):
                logger.info(f"  {confusion_matrix[i]}")
            
            # Calculate distribution of predictions
            pred_dist = np.bincount(all_predictions, minlength=no_classes)
            logger.info(f"Prediction distribution:")
            for i in range(no_classes):
                logger.info(f"  Class {i}: {pred_dist[i]} ({pred_dist[i]/len(all_predictions)*100:.2f}%)")
        else:
            logger.warning("No samples were processed successfully!")
            classwise_accuracy = np.zeros(no_classes)
            total_accuracy = 0.0
        
        return classwise_accuracy, total_accuracy, confusion_matrix

def collate_fn(batch):
    """
    Custom collate function for the DataLoader.
    Handles variable length audio samples by padding.
    """
    # Filter out None values (if any)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # Unpack the batch
    noisy, clean, environments, recsits, cut_id, extra, snr = zip(*batch)
    
    # Convert to tensors
    noisy = torch.stack(noisy)
    clean = torch.stack(clean)
    
    # Keep the rest as lists
    return noisy, clean, environments, recsits, cut_id, extra, snr

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LSTM Classifier Experiment")
    parser.add_argument("--experiment_no", type=int, default=1, help="Experiment number")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size for LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--use_validation", action="store_true", help="Use validation set")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--split_index", type=int, default=0, help="Index of the split to use (0-4)")
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.cuda and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available. Falling back to CPU.")
        args.cuda = False
    
    print(f"Starting experiment: lstm-classifier-{args.experiment_no}")
    print(f"Parameters: epochs={args.num_epochs}, batch_size={args.batch_size}")
    print(f"LSTM parameters: hidden_size={args.hidden_size}, num_layers={args.num_layers}")
    print(f"Validation: {'enabled' if args.use_validation else 'disabled'}")
    
    # Load data from splits
    split_file = os.path.join("splits", f"split_{args.split_index}.pkl")
    if not os.path.exists(split_file):
        logger.error(f"Split file not found: {split_file}")
        sys.exit(1)
    
    logger.info(f"Loading data from split file: {split_file}")
    try:
        with open(split_file, 'rb') as f:
            split_data = pickle.load(f)
            train_combined = split_data['train']
            test_combined = split_data['test']
    except Exception as e:
        logger.error(f"Error loading split file: {e}")
        sys.exit(1)
    
    # Create and run experiment
    experiment = LSTMExperiment(
        train_combined=train_combined,
        test_combined=test_combined,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        patience=args.patience,
        experiment_no=args.experiment_no,
        cuda=args.cuda,
        use_validation=args.use_validation,
        validation_split=args.validation_split
    )
    
    try:
        print(f"Starting experiment run {args.experiment_no}...")
        experiment.run()
        print(f"Experiment completed successfully!")
    except Exception as e:
        logger.error(f"Error during experiment: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1) 