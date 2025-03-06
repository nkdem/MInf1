import argparse
import os
import logging
import json
import torch
import numpy as np
import pickle
import soundfile as sf
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import re

sys.path.append(os.path.abspath(os.path.join('.')))
from models import LSTMClassifier
from helpers import compute_average_logmel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collate_fn(batch):
    """Custom collate function to handle different data formats."""
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

def load_model(model_dir):
    """
    Load the trained LSTM model and its metadata.
    
    Args:
        model_dir: Directory containing the trained model
        
    Returns:
        model: Loaded LSTM model
        metadata: Model metadata
    """
    # Load metadata
    metadata_path = os.path.join(model_dir, "metadata.json")
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Metadata file not found at {metadata_path}. Using default values.")
        metadata = {
            "input_size": 40,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "num_classes": 10
        }
    
    # Create model
    input_size = metadata.get("input_size", 40)
    hidden_size = metadata.get("hidden_size", 128)
    num_layers = metadata.get("num_layers", 2)
    dropout = metadata.get("dropout", 0.3)
    num_classes = metadata.get("Number of classes", 10)
    
    logger.info(f"Creating LSTM model with parameters:")
    logger.info(f"  Input size: {input_size}")
    logger.info(f"  Hidden size: {hidden_size}")
    logger.info(f"  Number of layers: {num_layers}")
    logger.info(f"  Dropout: {dropout}")
    logger.info(f"  Number of classes: {num_classes}")
    
    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=False
    )
    
    # Load model weights
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "final_model.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "model.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No model file found in {model_dir}")
    
    # Load model weights with appropriate device mapping
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Store paths in metadata
    metadata["model_path"] = model_path
    metadata["metadata_path"] = metadata_path
    
    return model, metadata

# Note: The preprocess_audio function has been replaced with direct feature computation
# in the predict and evaluate_model functions

def predict(model, waveform, device=None):
    """
    Make a prediction for a single audio waveform.
    
    Args:
        model: Trained LSTM model
        waveform: Audio waveform tensor of shape (channels, samples)
        device: Device to run the model on
        
    Returns:
        predicted_class: Predicted class index
        confidence: Confidence score (probability)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    # Ensure waveform is a tensor and has batch dimension
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)
    
    if waveform.dim() == 2:  # (channels, samples)
        waveform = waveform.unsqueeze(0)  # Add batch dimension
    
    # Move to device
    waveform = waveform.to(device)
    
    with torch.no_grad():
        try:
            # Compute features
            features = compute_average_logmel(waveform, device)
            
            # Check for NaN or Inf values
            if torch.isnan(features).any() or torch.isinf(features).any():
                logger.warning(f"NaN or Inf values detected in features! Replacing with zeros.")
                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Remove the channel dimension (dim=1)
            features = features.squeeze(1)
            
            # Transpose from (batch_size, n_mels, n_frames) to (batch_size, n_frames, n_mels)
            features = features.transpose(1, 2)
            
            # Forward pass
            outputs = model(features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get prediction and confidence
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), confidence.item()
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return -1, 0.0

def compute_logmels(waveforms, device, input_size=40):
    """Compute log-mel spectrograms for a batch of waveforms or file paths."""
    try:
        # If waveforms is a tuple of file paths or arrays, load/process them
        if isinstance(waveforms[0], (tuple, list)):
            batch_waveforms = []
            for stereo_data in waveforms:
                try:
                    if isinstance(stereo_data[0], str):
                        # Case 1: File paths
                        left_path, right_path = stereo_data
                        left_waveform, sr = sf.read(left_path)
                        right_waveform, sr = sf.read(right_path)
                    elif isinstance(stereo_data[0], np.ndarray):
                        # Case 2: Pre-loaded audio data as numpy arrays
                        left_waveform, right_waveform = stereo_data
                    else:
                        # Try to convert to numpy array (handles array-like objects)
                        try:
                            left_waveform = np.array(stereo_data[0], dtype=np.float32)
                            right_waveform = np.array(stereo_data[1], dtype=np.float32)
                        except Exception as e:
                            logger.warning(f"Could not convert data to numpy array: {e}")
                            continue
                    
                    # Stack channels
                    stereo_waveform = np.stack([left_waveform, right_waveform])
                    batch_waveforms.append(stereo_waveform)
                except Exception as e:
                    logger.warning(f"Error processing stereo data: {e}")
                    continue
            
            if not batch_waveforms:
                raise ValueError("No valid audio data in batch")
            
            # Convert to tensor
            waveforms = torch.tensor(np.stack(batch_waveforms), dtype=torch.float32).to(device)
        
        # Case 3: Already a tensor
        elif not isinstance(waveforms, torch.Tensor):
            raise ValueError(f"Unexpected waveforms type: {type(waveforms)}")
        
        # Compute log-mel spectrograms
        features = compute_average_logmel(waveforms, device)
        if torch.isnan(features).any() or torch.isinf(features).any():
            logger.warning("NaN or Inf values detected in features! Replacing with zeros.")
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = features.squeeze(1)
        features = features.transpose(1, 2)
        return features
    except Exception as e:
        logger.error(f"Error computing log-mel spectrograms: {e}")
        if isinstance(waveforms, torch.Tensor):
            batch_size = waveforms.shape[0]
        else:
            batch_size = len(waveforms)
        return torch.zeros((batch_size, 100, input_size), device=device)

def evaluate_model(model, test_loader, env_to_int, device):
    """Evaluate the LSTM model on the test set."""
    model.eval()
    total = 0
    correct = 0
    num_classes = len(env_to_int)
    confusion_matrix = np.zeros((num_classes, num_classes))
    all_predictions = []
    all_labels = []
    
    logger.info(f"Evaluating LSTM model on {len(test_loader)} batches")
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Environment to int mapping: {env_to_int}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing", unit="batch")):
            noisy, clean, environments, recsits, cut_id, extra, snr = batch
            
            try:
                # Log data types for debugging
                if batch_idx % 20 == 0:  # Log every 20th batch
                    logger.info(f"\nBatch {batch_idx} data types:")
                    logger.info(f"Noisy type: {type(noisy)}")
                    if isinstance(noisy, (tuple, list)) and len(noisy) > 0:
                        logger.info(f"First noisy item type: {type(noisy[0])}")
                        if isinstance(noisy[0], (tuple, list)) and len(noisy[0]) > 0:
                            logger.info(f"First noisy stereo item type: {type(noisy[0][0])}")
                    logger.info(f"Environment types: {[type(env) for env in environments[:3]]}...")
                
                # Compute features
                features = compute_logmels(noisy, device)
                
                # Prepare labels
                labels = torch.tensor([env_to_int[env] for env in environments], dtype=torch.long).to(device)
                
                # Log first batch details
                if batch_idx == 0:
                    logger.info(f"Feature shape: {features.shape}")
                    logger.info(f"Labels shape: {labels.shape}")
                    logger.info(f"Labels: {labels}")
                    logger.info(f"Environments: {environments}")
                
                # Forward pass
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                
                # Log first batch predictions
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
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                logger.error(f"Batch data shapes/types:")
                logger.error(f"Noisy: {type(noisy)}, len={len(noisy) if isinstance(noisy, (list, tuple)) else 'N/A'}")
                logger.error(f"Environments: {type(environments)}, len={len(environments)}")
                continue
    
    # Calculate metrics
    metrics = {}
    if total > 0:
        classwise_accuracy = confusion_matrix.diagonal() / (confusion_matrix.sum(axis=1) + 1e-10)
        total_accuracy = correct / total
        
        # Log results
        logger.info(f"Total accuracy: {total_accuracy:.4f} ({correct}/{total})")
        logger.info("Class-wise accuracies:")
        for i in range(num_classes):
            class_total = confusion_matrix.sum(axis=1)[i]
            if class_total > 0:
                logger.info(f"  Class {i}: {classwise_accuracy[i]:.4f} ({int(confusion_matrix[i, i])}/{int(class_total)})")
        
        # Calculate prediction distribution
        pred_dist = np.bincount(all_predictions, minlength=num_classes)
        logger.info("Prediction distribution:")
        for i in range(num_classes):
            logger.info(f"  Class {i}: {pred_dist[i]} ({pred_dist[i]/len(all_predictions)*100:.2f}%)")
        
        # Store metrics
        metrics = {
            'total_accuracy': float(total_accuracy),
            'classwise_accuracy': classwise_accuracy.tolist(),
            'confusion_matrix': confusion_matrix.tolist(),
            'prediction_distribution': pred_dist.tolist(),
            'num_samples': total
        }
    else:
        logger.warning("No samples were processed successfully!")
        metrics = {
            'error': 'No samples processed successfully',
            'total_accuracy': 0.0,
            'classwise_accuracy': [0.0] * num_classes
        }
    
    return metrics

def main():
    """Main function for LSTM inference."""
    parser = argparse.ArgumentParser(description="Evaluate trained LSTM model")
    parser.add_argument("--split_index", type=int, default=0, help="Index of the split to use (0-4)")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the trained model")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load test data
    split_file = os.path.join("splits", f"split_{args.split_index}.pkl")
    if not os.path.exists(split_file):
        logger.error(f"Split file not found: {split_file}")
        sys.exit(1)
    
    logger.info(f"Loading test data from split file: {split_file}")
    try:
        with open(split_file, 'rb') as f:
            split_data = pickle.load(f)
            test_combined = split_data['test']
    except Exception as e:
        logger.error(f"Error loading split file: {e}")
        sys.exit(1)
    
    # Load model metadata
    metadata_path = os.path.join(args.model_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        logger.error(f"Model metadata not found: {metadata_path}")
        sys.exit(1)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create model
    model = LSTMClassifier(
        input_size=metadata['input_size'],
        hidden_size=metadata['hidden_size'],
        num_layers=metadata['num_layers'],
        num_classes=metadata['Number of classes'],
        dropout=metadata['dropout'],
        bidirectional=False
    ).to(device)
    
    # Load model weights
    model_path = os.path.join(args.model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        logger.error(f"Model weights not found: {model_path}")
        sys.exit(1)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info("Model loaded successfully")
    
    # Create test dataloader
    test_loader = DataLoader(
        test_combined,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*[i for i in x if i is not None]))
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        env_to_int=metadata['Class mapping'],
        device=device
    )
    
    # Save results
    results_dir = os.path.join(os.path.dirname(args.model_dir), "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"metrics_split_{args.split_index}.json")
    
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Results saved to {results_path}")
    print(f"\nTotal Accuracy: {metrics['total_accuracy']:.4f}")
    print("Class-wise Accuracies:")
    for i, acc in enumerate(metrics['classwise_accuracy']):
        print(f"  Class {i}: {acc:.4f}")

if __name__ == "__main__":
    main() 