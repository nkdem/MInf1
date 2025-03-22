import torch.nn as nn
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
        feats = self.conv_block(x)
        out = self.conv_final(feats)
        return out.squeeze(2)