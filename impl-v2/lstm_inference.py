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

from models import LSTMClassifier
from helpers import compute_average_logmel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

def load_model(model_dir):
    """
    Load a trained LSTM model from the specified directory.
    
    Args:
        model_dir (str): Directory containing the model files
        
    Returns:
        model (LSTMClassifier): Loaded model
        metadata (dict): Model metadata
    """
    # Load metadata
    metadata_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Create model with the same parameters
    model = LSTMClassifier(
        input_size=metadata["input_size"],
        hidden_size=metadata["hidden_size"],
        num_layers=metadata["num_layers"],
        num_classes=metadata["num_classes"],
        dropout=metadata["dropout"],
        bidirectional=metadata.get("bidirectional", False)
    )
    
    # Load model weights
    model_path = os.path.join(model_dir, "best_model.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "final_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found in {model_dir}")
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    return model, metadata

def preprocess_audio(waveform, device):
    """
    Preprocess audio waveform for LSTM model.
    
    Args:
        waveform (np.ndarray or torch.Tensor): Audio waveform
        device (torch.device): Device to use for processing
        
    Returns:
        torch.Tensor: Preprocessed features ready for model input
    """
    # Compute log-mel spectrogram
    if isinstance(waveform, list) and len(waveform) == 2:
        # Stereo audio (left and right channels)
        left_feats = compute_average_logmel([waveform[0]], device)
        right_feats = compute_average_logmel([waveform[1]], device)
        # Average the features from both channels
        features = (left_feats + right_feats) / 2
    else:
        # Mono audio
        features = compute_average_logmel([waveform], device)
    
    # The features from compute_average_logmel have shape (batch_size, 1, n_mels, n_frames)
    # For LSTM, we need (batch_size, sequence_length, input_size)
    # Where sequence_length = n_frames and input_size = n_mels
    
    # Remove the channel dimension (dim=1)
    features = features.squeeze(1)
    
    # Transpose from (batch_size, n_mels, n_frames) to (batch_size, n_frames, n_mels)
    features = features.transpose(1, 2)
    
    return features

def predict(model, waveform, device=None):
    """
    Make a prediction using the LSTM model.
    
    Args:
        model (LSTMClassifier): Trained LSTM model
        waveform (np.ndarray or torch.Tensor): Audio waveform
        device (torch.device, optional): Device to run inference on
        
    Returns:
        int: Predicted class index
        torch.Tensor: Class probabilities
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Preprocess audio
    features = preprocess_audio(waveform, device)
    
    # Move to device
    features = features.to(device)
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item(), probabilities.squeeze().cpu()

def evaluate_model(model, test_loader, metadata, device=None):
    """
    Evaluate the model on a test dataset.
    
    Args:
        model (LSTMClassifier): Trained LSTM model
        test_loader (DataLoader): DataLoader for test data
        metadata (dict): Model metadata containing class mapping
        device (torch.device, optional): Device to run evaluation on
        
    Returns:
        dict: Evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else "cpu")
    
    model = model.to(device)
    model.eval()
    
    # Get class mapping
    class_mapping = metadata.get("classes", {})
    # Invert the mapping to get index -> class name
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", unit="batch"):
            if len(batch) == 7:
                # HEAR-DS format
                mixed_waveforms, _, envs, _, _, _, _ = batch
                # Convert environment names to class indices
                labels = [class_mapping.get(env, 0) for env in envs]
                
                # Process each sample in the batch
                for i, waveform in enumerate(mixed_waveforms):
                    # Preprocess audio
                    features = preprocess_audio(waveform, device)
                    features = features.to(device)
                    
                    # Make prediction
                    outputs = model(features)
                    _, preds = torch.max(outputs, 1)
                    
                    # Store predictions and labels
                    all_preds.append(preds.item())
                    all_labels.append(labels[i])
            else:
                # TUT format
                waveforms, labels_str = batch
                # Convert labels to indices
                labels = [class_mapping.get(label, 0) for label in labels_str]
                
                # Process each sample in the batch
                for i, waveform in enumerate(waveforms):
                    # Preprocess audio
                    features = preprocess_audio(waveform, device)
                    features = features.to(device)
                    
                    # Make prediction
                    outputs = model(features)
                    _, preds = torch.max(outputs, 1)
                    
                    # Store predictions and labels
                    all_preds.append(preds.item())
                    all_labels.append(labels[i])
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Convert confusion matrix to class names
    cm_with_names = {
        "matrix": cm.tolist(),
        "classes": [idx_to_class.get(i, f"Unknown-{i}") for i in range(len(cm))]
    }
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm_with_names
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Inference with LSTM classifier for audio classification")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing the trained model")
    parser.add_argument("--audio_file", type=str, help="Path to audio file for single prediction")
    parser.add_argument("--split_index", type=int, default=0, help="Index of the split to use for evaluation (0-4)")
    parser.add_argument("--output_file", type=str, help="Path to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for inference")
    
    args = parser.parse_args()
    
    # Determine device
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    
    # Load model
    model, metadata = load_model(args.model_dir)
    
    # Get class mapping
    class_mapping = metadata.get("classes", {})
    # Invert the mapping to get index -> class name
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    if args.audio_file:
        # Single file prediction
        import soundfile as sf
        
        logger.info(f"Loading audio file: {args.audio_file}")
        waveform, sample_rate = sf.read(args.audio_file)
        
        # Make prediction
        pred_idx, probabilities = predict(model, waveform, device)
        
        # Get class name
        pred_class = idx_to_class.get(pred_idx, f"Unknown-{pred_idx}")
        
        logger.info(f"Prediction: {pred_class} (index: {pred_idx})")
        logger.info("Class probabilities:")
        for i, prob in enumerate(probabilities):
            class_name = idx_to_class.get(i, f"Unknown-{i}")
            logger.info(f"  {class_name}: {prob.item():.4f}")
    
    # Evaluation using split data
    split_file = f'splits/split_{args.split_index}.pkl'
    if os.path.exists(split_file):
        logger.info(f"Evaluating model using split {args.split_index}")
        
        # Load the split data
        with open(split_file, 'rb') as f:
            split_data = pickle.load(f)
            test_dataset = split_data['test']
            
            logger.info(f"Loaded test dataset with {len(test_dataset)} samples")
            
            # Create data loader with custom collate function
            test_loader = DataLoader(
                test_dataset, 
                batch_size=args.batch_size, 
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True
            )
            
            # Evaluate model
            metrics = evaluate_model(model, test_loader, metadata, device)
            
            logger.info(f"Evaluation metrics:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1']:.4f}")
            
            if args.output_file:
                with open(args.output_file, "w") as f:
                    json.dump(metrics, f, indent=4)
                logger.info(f"Evaluation results saved to {args.output_file}")
    else:
        logger.warning(f"Split file {split_file} not found. Skipping evaluation.")

if __name__ == "__main__":
    main() 