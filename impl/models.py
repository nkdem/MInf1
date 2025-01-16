import torch.nn as nn
class AudioCNN(nn.Module):
    def __init__(self, num_classes, cnn1_channels, cnn2_channels, fc_neurons):
        super(AudioCNN, self).__init__()

        # First CNN module
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=cnn1_channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.BatchNorm2d(cnn1_channels),  # Added normalization layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5)),
            nn.Dropout(0.3)
        )

        # Second CNN module
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=cnn1_channels, out_channels=cnn2_channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
            nn.BatchNorm2d(cnn2_channels),  # Added normalization layer
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