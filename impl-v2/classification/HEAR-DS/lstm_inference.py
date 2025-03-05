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

def evaluate_model(model, test_loader, metadata, device=None):
    """
    Evaluate the model on the test set.
    
    Args:
        model: Trained LSTM model
        test_loader: DataLoader for the test set
        metadata: Model metadata
        device: Device to run the model on
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    # Get model directory from metadata
    model_dir = metadata.get("model_dir", "")
    if not model_dir and "model_path" in metadata:
        model_dir = os.path.dirname(metadata["model_path"])
    
    # Load class mapping
    int_to_label_path = os.path.join(model_dir, "int_to_label.txt")
    logger.info(f"Looking for class mapping at {int_to_label_path}")
    
    env_to_int = {}
    if os.path.exists(int_to_label_path):
        try:
            with open(int_to_label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        env = parts[0]
                        idx = int(parts[1])
                        env_to_int[env] = idx
                    elif len(parts) == 1:
                        # Try to parse as "env idx" format
                        match = re.match(r"(\w+)\s+(\d+)", line.strip())
                        if match:
                            env, idx = match.groups()
                            env_to_int[env] = int(idx)
            logger.info(f"Loaded class mapping with {len(env_to_int)} classes")
        except Exception as e:
            logger.error(f"Error loading class mapping: {e}")
            env_to_int = {}
    
    # If no mapping file, create a default mapping
    if not env_to_int:
        logger.warning(f"No class mapping found at {int_to_label_path}. Using default mapping.")
        num_classes = metadata.get("Number of classes", 0)
        if num_classes == 0:
            # Try to infer from model's output layer
            if hasattr(model, 'fc') and hasattr(model.fc[-1], 'out_features'):
                num_classes = model.fc[-1].out_features
                logger.info(f"Inferred {num_classes} classes from model architecture")
            else:
                num_classes = 10  # Default fallback
                logger.warning(f"Could not determine number of classes. Using default: {num_classes}")
        
        env_to_int = {f"class_{i}": i for i in range(num_classes)}
        logger.info(f"Created default mapping with {num_classes} classes")
    
    int_to_env = {v: k for k, v in env_to_int.items()}
    
    # Initialize metrics
    all_predictions = []
    all_labels = []
    
    # Evaluate on test set
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating", unit="batch")):
            noisy, clean, environments, recsits, cut_id, extra, snr = batch
            
            try:
                # Compute features for LSTM
                features = compute_average_logmel(noisy, device)
                
                # Check for NaN or Inf values
                if torch.isnan(features).any() or torch.isinf(features).any():
                    logger.warning(f"NaN or Inf values detected in features! Replacing with zeros.")
                    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Remove the channel dimension (dim=1)
                features = features.squeeze(1)
                
                # Transpose from (batch_size, n_mels, n_frames) to (batch_size, n_frames, n_mels)
                features = features.transpose(1, 2)
                
                # Log shapes for first batch
                if batch_idx == 0:
                    logger.info(f"Feature shape: {features.shape}")
                    logger.info(f"Environments: {environments}")
                
                # Get labels
                try:
                    labels = torch.tensor([env_to_int.get(env, 0) for env in environments], dtype=torch.long).to(device)
                    
                    # Log labels for first batch
                    if batch_idx == 0:
                        logger.info(f"Labels shape: {labels.shape}")
                        logger.info(f"Labels: {labels}")
                    
                    # Forward pass
                    outputs = model(features)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Log predictions for first batch
                    if batch_idx == 0:
                        logger.info(f"Predictions: {predicted}")
                        logger.info(f"Correct predictions: {(predicted == labels).sum().item()}/{len(labels)}")
                    
                    # Store predictions and labels
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                except KeyError as e:
                    logger.error(f"KeyError with environment: {e}")
                    logger.error(f"Available environments: {list(env_to_int.keys())}")
                    logger.error(f"Batch environments: {environments}")
                    continue
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Calculate metrics
    metrics = {}
    try:
        if len(all_predictions) > 0 and len(all_labels) > 0:
            # Convert to numpy arrays
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            
            # Calculate metrics
            metrics["accuracy"] = float(accuracy_score(all_labels, all_predictions))
            metrics["precision"] = float(precision_score(all_labels, all_predictions, average="weighted", zero_division=0))
            metrics["recall"] = float(recall_score(all_labels, all_predictions, average="weighted", zero_division=0))
            metrics["f1_score"] = float(f1_score(all_labels, all_predictions, average="weighted", zero_division=0))
            
            # Calculate confusion matrix
            cm = confusion_matrix(all_labels, all_predictions)
            metrics["confusion_matrix"] = cm.tolist()
            
            # Calculate per-class accuracy
            class_accuracy = {}
            for i in range(len(cm)):
                if i in int_to_env:
                    class_name = int_to_env[i]
                    if cm[i].sum() > 0:
                        class_accuracy[class_name] = float(cm[i, i] / cm[i].sum())
                    else:
                        class_accuracy[class_name] = 0.0
            metrics["class_accuracy"] = class_accuracy
            
            # Calculate prediction distribution
            pred_dist = np.bincount(all_predictions, minlength=len(env_to_int))
            metrics["prediction_distribution"] = {int_to_env.get(i, f"class_{i}"): float(pred_dist[i] / len(all_predictions)) for i in range(len(pred_dist))}
            
            logger.info(f"Evaluation metrics:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        else:
            logger.warning("No predictions or labels collected. Cannot calculate metrics.")
            metrics["error"] = "No predictions or labels collected"
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        metrics["error"] = str(e)
        metrics["predictions_shape"] = str(np.array(all_predictions).shape if all_predictions else "empty")
        metrics["labels_shape"] = str(np.array(all_labels).shape if all_labels else "empty")
        metrics["unique_predictions"] = str(np.unique(all_predictions).tolist() if all_predictions else "empty")
        metrics["unique_labels"] = str(np.unique(all_labels).tolist() if all_labels else "empty")
    
    return metrics

def main():
    """Main function for LSTM inference."""
    parser = argparse.ArgumentParser(description="LSTM Inference")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the trained model")
    parser.add_argument("--split_index", type=int, default=0, help="Index of the split to evaluate")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--split_file", type=str, help="Path to the split file (optional)")
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.cuda and not torch.cuda.is_available():
        logger.warning("CUDA was requested but is not available. Falling back to CPU.")
        args.cuda = False
    
    # Set device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model, metadata = load_model(args.model_dir)
    model = model.to(device)
    metadata["model_dir"] = args.model_dir
    
    # Load test data
    if args.split_file:
        split_file = args.split_file
    else:
        # Try different possible locations
        possible_locations = [
            os.path.join("data", f"test_split_{args.split_index}.pkl"),
            os.path.join("data", f"split_{args.split_index}.pkl"),
            os.path.join("splits", f"split_{args.split_index}.pkl"),
            os.path.join("classification", "HEAR-DS", "data", f"test_split_{args.split_index}.pkl"),
            os.path.join("classification", "HEAR-DS", "splits", f"split_{args.split_index}.pkl")
        ]
        
        split_file = None
        for loc in possible_locations:
            if os.path.exists(loc):
                split_file = loc
                break
        
        if not split_file:
            logger.error(f"Split file not found. Tried the following locations:")
            for loc in possible_locations:
                logger.error(f"  - {loc}")
            sys.exit(1)
    
    logger.info(f"Loading test data from {split_file}")
    try:
        with open(split_file, "rb") as f:
            split_data = pickle.load(f)
            
        # Handle different split file formats
        if isinstance(split_data, dict) and "test" in split_data:
            test_data = split_data["test"]
        else:
            test_data = split_data
    except Exception as e:
        logger.error(f"Error loading split file: {e}")
        sys.exit(1)
    
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=32, shuffle=False, collate_fn=collate_fn
    )
    
    # Evaluate model
    logger.info(f"Evaluating model on {len(test_loader)} batches")
    metrics = evaluate_model(model, test_loader, metadata, device)
    
    # Create results directory
    results_dir = os.path.join(args.model_dir, "evaluation")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save metrics
    results_file = os.path.join(results_dir, f"metrics_split_{args.split_index}.json")
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"Accuracy: {metrics.get('accuracy', 'N/A')}")
    print(f"F1 Score: {metrics.get('f1_score', 'N/A')}")
    
    return metrics

if __name__ == "__main__":
    main() 