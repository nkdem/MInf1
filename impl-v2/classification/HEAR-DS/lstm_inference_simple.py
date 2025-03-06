import os
import sys
import torch
import logging
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
import soundfile as sf

sys.path.append(os.path.abspath(os.path.join('.')))
from models import LSTMClassifier
from helpers import compute_average_logmel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def collect_model_results(test_loader, model, no_classes, env_to_int, device):
    """
    Simple evaluation following the base experiment approach.
    """
    model.eval()
    total = 0
    correct = 0
    confusion_matrix = np.zeros((no_classes, no_classes))
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            noisy, clean, environments, recsits, cut_id, extra, snr = batch
            # Compute features
            logmels = compute_average_logmel(noisy, device)
            
            # Log shapes for debugging
            if total == 0:
                logger.info(f"Initial logmels shape: {logmels.shape}")
            
            # Remove the channel dimension and transpose to (batch, time, features)
            logmels = logmels.squeeze(1)  # Remove channel dim
            logmels = logmels.transpose(1, 2)  # Swap time and features dims
            
            if total == 0:
                logger.info(f"Processed logmels shape: {logmels.shape}")
            
            # Get labels
            labels = torch.tensor([env_to_int[env] for env in environments], dtype=torch.long).to(device)
            # Forward pass
            outputs = model(logmels)
            _, predicted = torch.max(outputs.data, 1)
            # Update statistics
            total += len(predicted)
            correct += (predicted == labels).sum().item()
            # Update confusion matrix
            for i in range(len(predicted)):
                confusion_matrix[labels[i], predicted[i]] += 1
    
    classwise_accuracy = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
    total_accuracy = correct / total
    return classwise_accuracy, total_accuracy, confusion_matrix

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Simple LSTM Inference")
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
    
    with open(split_file, 'rb') as f:
        split_data = pickle.load(f)
        test_combined = split_data['test']
    
    # Load model metadata
    metadata_path = os.path.join(args.model_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        logger.error(f"Model metadata not found: {metadata_path}")
        sys.exit(1)
    
    import json
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
    
    # Create test dataloader with proper collate function
    def collate_fn(batch):
        audios, clean, environments, recsits, cut_ids, extra, snrs = zip(*batch)
        processed_audios = []
        for i in range(len(batch)):
            if snrs[i] is None:
                # Load audio files
                waveform_l, sr = sf.read(batch[i][0][0])
                waveform_r, sr = sf.read(batch[i][0][1])
                processed_audios.append([waveform_l, waveform_r])
            else:
                # Already loaded audio data
                processed_audios.append([batch[i][0][0], batch[i][0][1]])
        return processed_audios, clean, environments, recsits, cut_ids, extra, snrs
    
    test_loader = DataLoader(
        test_combined,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Evaluate model
    classwise_accuracy, total_accuracy, confusion_matrix = collect_model_results(
        test_loader=test_loader,
        model=model,
        no_classes=metadata['Number of classes'],
        env_to_int=metadata['Class mapping'],
        device=device
    )
    
    # Print results
    print(f"\nTotal Accuracy: {total_accuracy:.4f}")
    print("Class-wise Accuracies:")
    for i, acc in enumerate(classwise_accuracy):
        print(f"  Class {i}: {acc:.4f}")
    
    # Save results
    results = {
        'classwise_accuracy': classwise_accuracy.tolist(),
        'total_accuracy': float(total_accuracy),
        'confusion_matrix': confusion_matrix.tolist()
    }
    
    results_dir = os.path.join(os.path.dirname(args.model_dir), "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"metrics_split_{args.split_index}.json")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    main() 