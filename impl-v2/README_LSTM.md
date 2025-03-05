# Unidirectional LSTM Classifier for Audio Classification

This repository contains a unidirectional LSTM (Long Short-Term Memory) model for audio classification tasks. The model is designed to classify audio samples into different environmental categories based on their acoustic features.

## Model Architecture

The LSTM classifier consists of the following components:

1. **LSTM Layers**: One or more unidirectional LSTM layers that process the input sequence.
2. **Fully Connected Layers**: A series of fully connected layers with ReLU activation and dropout for classification.

The model takes mel-spectrogram features as input and outputs class probabilities.

## Files

- `models.py`: Contains the `LSTMClassifier` class definition.
- `lstm_classifier_train.py`: Script for training the LSTM classifier.
- `lstm_inference.py`: Script for making predictions with a trained LSTM model.

## Usage

### Training

To train the LSTM classifier, use the `lstm_classifier_train.py` script:

```bash
python lstm_classifier_train.py --split_index 0 --output_dir models --batch_size 32 --num_epochs 50 --learning_rate 0.001 --hidden_size 128 --num_layers 2 --dropout 0.3 --patience 5 --cuda
```

Arguments:
- `--split_index`: Index of the split to use (0-4) (default: 0)
- `--output_dir`: Directory to save models (default: "models")
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 0.001)
- `--hidden_size`: Hidden size of LSTM (default: 128)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--dropout`: Dropout probability (default: 0.3)
- `--patience`: Patience for early stopping (default: 5)
- `--cuda`: Use CUDA for training (if available)

### Inference

To make predictions with a trained LSTM model, use the `lstm_inference.py` script:

```bash
python lstm_inference.py --model_dir models/lstm-128-2-1234567890 --split_index 0 --output_file results.json --cuda
```

Arguments:
- `--model_dir`: Directory containing the trained model (required)
- `--audio_file`: Path to audio file for single prediction
- `--split_index`: Index of the split to use for evaluation (0-4) (default: 0)
- `--output_file`: Path to save evaluation results
- `--batch_size`: Batch size for evaluation (default: 32)
- `--cuda`: Use CUDA for inference (if available)

## Data Splits

The model uses pre-generated data splits stored in the `splits` directory. Each split contains:
- Training dataset
- Test dataset

The splits are stored as pickle files (`split_0.pkl`, `split_1.pkl`, etc.) and are loaded automatically by the training and inference scripts.

## Model Parameters

The LSTM classifier has the following parameters:

- `input_size`: The number of expected features in the input (feature dimension)
- `hidden_size`: The number of features in the hidden state
- `num_layers`: Number of recurrent layers
- `num_classes`: Number of classes to classify
- `dropout`: Dropout probability (default: 0.3)
- `bidirectional`: If True, becomes a bidirectional LSTM (default: False for unidirectional)

## Data Format

The model expects audio data to be preprocessed into mel-spectrogram features. The input shape for the LSTM should be:
- `(batch_size, sequence_length, input_size)`

Where:
- `batch_size` is the number of samples in a batch
- `sequence_length` is the number of time steps
- `input_size` is the number of features (e.g., mel bands)

## Example

```python
import torch
from models import LSTMClassifier

# Initialize model
model = LSTMClassifier(
    input_size=40,  # Number of mel bands
    hidden_size=128,
    num_layers=2,
    num_classes=10,
    dropout=0.3,
    bidirectional=False  # Unidirectional LSTM
)

# Forward pass
batch_size = 16
sequence_length = 100  # Time steps
input_size = 40  # Feature dimension
x = torch.randn(batch_size, sequence_length, input_size)
output = model(x)  # Shape: (batch_size, num_classes)
```

## Performance Considerations

- Increasing `hidden_size` and `num_layers` can improve model capacity but requires more computational resources.
- Using dropout helps prevent overfitting, especially with larger models.
- For longer sequences, consider using a smaller batch size to avoid memory issues.
- The unidirectional LSTM processes the sequence in one direction, which can be more efficient than bidirectional LSTMs but may capture less contextual information.

## Requirements

- PyTorch
- NumPy
- scikit-learn
- tqdm
- soundfile (for audio loading)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 