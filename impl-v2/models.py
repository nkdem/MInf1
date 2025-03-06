import torch.nn as nn
import torch

class AudioCNN(nn.Module):
    def __init__(self, num_classes, cnn1_channels, cnn2_channels, fc_neurons):
        super(AudioCNN, self).__init__()

        # First CNN module
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=cnn1_channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.BatchNorm2d(cnn1_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5)),
            nn.Dropout(0.3)
        )

        # Second CNN module
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=cnn1_channels, out_channels=cnn2_channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.BatchNorm2d(cnn2_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 100), stride=(4, 100)),
            nn.Dropout(0.3)
        )

        # Fully connected module
        self.fc = nn.Sequential(
            nn.Linear(cnn2_channels * 2 * 1, fc_neurons),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fc_neurons, num_classes),
        )

    def forward(self, x):
        x = x.to(next(self.parameters()).device)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

class CNNSpeechEnhancer(nn.Module):
    def __init__(self):
        super(CNNSpeechEnhancer, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv_final = nn.Conv2d(in_channels=64, out_channels=40, kernel_size=3, padding="same")

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        feats = self.conv_block(x)
        print(f"After conv_block: {feats.shape}")
        out = self.conv_final(feats)
        print(f"Output shape: {out.shape}")
        return out

class CNNSpeechEnhancer(nn.Module):
    def __init__(self):
        super(CNNSpeechEnhancer, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv_final = nn.Conv2d(in_channels=64, out_channels=40, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x):
        feats = self.conv_block(x)
        out = self.conv_final(feats)
        return out.squeeze(2)  # Remove the height dimension

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=3, num_classes=10, dropout=0.4, bidirectional=True):
        """
        Enhanced LSTM classifier for audio classification tasks.
        
        Args:
            input_size (int): The number of expected features in the input x (feature dimension)
            hidden_size (int): The number of features in the hidden state h (default: 256)
            num_layers (int): Number of recurrent layers (default: 3)
            num_classes (int): Number of classes to classify
            dropout (float): Dropout probability (default: 0.4)
            bidirectional (bool): If True, becomes a bidirectional LSTM (default: True)
        """
        super(LSTMClassifier, self).__init__()
        
        # Convolutional front-end for feature extraction
        self.conv_frontend = nn.Sequential(
            nn.Conv1d(input_size, hidden_size // 2, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size // 2, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layers with batch normalization
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Batch normalization for LSTM outputs
        self.lstm_bn = nn.BatchNorm1d(hidden_size * 2 if bidirectional else hidden_size)
        
        # Self-attention mechanism
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_dim // 2, 1)
        )
        
        # Fully connected layers with batch normalization
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.BatchNorm1d(lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4),
            nn.BatchNorm1d(lstm_output_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 4, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass of the enhanced LSTM classifier.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Move input to the same device as model parameters
        x = x.to(next(self.parameters()).device)
        
        # Apply convolutional front-end
        # Transpose for conv1d: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.conv_frontend(x)
        # Transpose back: (batch, features, seq_len) -> (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply batch normalization to LSTM outputs
        # Reshape for batch norm: (batch, seq_len, features) -> (batch, features, seq_len)
        lstm_out_bn = self.lstm_bn(lstm_out.transpose(1, 2)).transpose(1, 2)
        
        # Self-attention mechanism
        attention_weights = self.attention(lstm_out_bn)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)  # (batch, seq_len, 1)
        
        # Apply attention weights
        attended_out = torch.bmm(lstm_out_bn.transpose(1, 2), attention_weights)  # (batch, features, 1)
        attended_out = attended_out.squeeze(-1)  # (batch, features)
        
        # Pass through fully connected layers
        logits = self.fc(attended_out)
        
        return logits