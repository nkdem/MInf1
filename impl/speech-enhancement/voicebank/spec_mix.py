import torch.nn as nn
class UNET(nn.Module):
    # 8 encoder blocks, 1 mid block, 8 decoder blocks and 1 output block

    def __init__(self):
        super(UNET, self).__init__()

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.mid_block = nn.ModuleList()
        self.output_block = nn.ModuleList()

        # Encoder blocks
        # The shape of the processed input is [B, F, T, C] = [B, 256, T, 2], where B denotes the batch size, F denotes frequency bins, T denotes time bins and C denotes channels. T depends on the audio length. The first channel is the real part of the spectrogram and the second channel is the imaginary part of the spectrogram.

        encoder_strides = [(1,1), (1,1), (2,1), (2,1), (2,1), (2,1), (1,1), (1,1)]
        encoder_filters = [(7,1), (1,7), (7,5), (7,5), (5,3), (5,3), (5,3), (5,3)]
        encoder_channel_outputs = [45, 45, 90, 90, 90, 90, 90, 64]

        for i in range(8):
            # in_channels is i-1th block's out_channels
            in_channels = encoder_channel_outputs[i-1] if i > 0 else 2 # first block has 2 channels (real and imaginary)
            out_channels = encoder_channel_outputs[i]
            self.encoder_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, encoder_filters[i], stride=encoder_strides[i]),
                # nn.BatchNorm2d(encoder_channel_outputs[i+1]),
                # nn.ReLU(inplace=True),
            ))

        # Mid block
        self.mid_block = nn.Sequential(
            nn.Conv2d(in_channels=encoder_channel_outputs[7], out_channels=encoder_channel_outputs[7], kernel_size=(3,3), stride=(1,1)),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),
        )

        # Decoder blocks
        for i in range(7, -1, -1):
            in_channels = encoder_channel_outputs[i]
            out_channels = encoder_channel_outputs[i-1] if i != 0 else 45
            self.decoder_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, encoder_filters[i], stride=encoder_strides[i]),
                # nn.BatchNorm2d(encoder_channel_outputs[i-1]),
                # nn.ReLU(inplace=True),
            ))
        
        # Output block
        self.output_block = nn.Sequential(
            nn.Conv2d(in_channels=encoder_channel_outputs[0], out_channels=1, kernel_size=(1,1), stride=(1,1)),
            # nn.BatchNorm2d(2),
            # nn.ReLU(inplace=True),
        )

        # phase sensitive mask

if __name__ == "__main__":
    unet = UNET()
    print(unet)


