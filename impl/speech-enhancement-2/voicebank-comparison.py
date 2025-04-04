import collections
import json
import logging
import os
import pickle
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchaudio
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
from pesq import pesq
from pystoi import stoi

import torchaudio.transforms as T


sys.path.append(os.path.abspath(os.path.join('.')))
from voicebank_dataset import get_loaders

def process_waveform(x, device):
    """
    Efficiently convert a waveform to a torch.Tensor on the target device.
    If x is a numpy array, torch.as_tensor is used to avoid copying.
    If it's already a tensor, it's moved to the target device.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device) if x.device != device else x
    elif isinstance(x, np.ndarray):
        return torch.as_tensor(x, dtype=torch.float32, device=device)
    else:
        return torch.tensor(x, dtype=torch.float32, device=device)
def compute_spectrogram(audio_batch, device, normalize=True):
    """
    Compute the spectrogram for a batch of waveforms.
    """
    # Convert waveforms to tensors
    audio_batch = process_waveform(audio_batch, device)
    # compute spectrogram
    # power = None to get complex spectrogram
    spectrogram = T.Spectrogram(n_fft=512, hop_length=256, power=None, normalized=True).to(device)(audio_batch)

    return spectrogram



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CNNSpeechEnhancer2(nn.Module):
    def __init__(self):
        super(CNNSpeechEnhancer2, self).__init__()

        self.conv_block = nn.Sequential(
            # magnitude and phase
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv_final = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding="same")

    def forward(self, x):
        # x is of shape (batch_size, 1, 40, n_frames)
        # train on magnitude only
        feats = self.conv_block(x)
        out = self.conv_final(feats)
        return out.squeeze(2)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        encoder_channels = [(1,45), (45,45), (45,90), (90,90), (90,90), (90,90), (90,90), (90,64)]
        encoder_strides = [(1,1), (1,1), (2,1), (2,1), (2,1), (2,1), (2,1), (1,1)]
        encoder_kernels = [(7,1), (1,7), (7,5), (7,5), (5,3), (5,3), (5,3), (5,3)]

        decoder_channels = [(64,90), (90,90), (90,90), (90,90), (90, 45), (45,45), (45,45), (45,1)]
        decoder_strides = [(1,1), (2,1), (2,1), (2,1), (2,1), (2,1), (1,1), (1,1)]
        decoder_kernels = [(5,3), (5,3), (5,3), (5,3), (7,5), (7,5), (1,7), (7,1)]

        self.encoder = nn.ModuleList()
        for i in range(len(encoder_channels)):
            # Calculate padding to maintain spatial dimensions
            padding_h = (encoder_kernels[i][0] - 1) // 2
            padding_w = (encoder_kernels[i][1] - 1) // 2
            
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        encoder_channels[i][0], 
                        encoder_channels[i][1], 
                        encoder_kernels[i], 
                        stride=encoder_strides[i],
                        padding=(padding_h, padding_w)
                    ),
                    nn.ReLU()
                )
            )

        # Mid-level layer
        self.mid_level = nn.Sequential(
            nn.Conv2d(64, 64, (3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU()
        )

        self.decoder = nn.ModuleList()
        for i in range(len(decoder_channels)):
            # Calculate padding for transposed convolutions
            padding_h = (decoder_kernels[i][0] - 1) // 2
            padding_w = (decoder_kernels[i][1] - 1) // 2
            
            # For stride > 1, we need output_padding to ensure correct shape
            output_padding = (
                decoder_strides[i][0] - 1 if decoder_strides[i][0] > 1 else 0,
                decoder_strides[i][1] - 1 if decoder_strides[i][1] > 1 else 0
            )
            
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        decoder_channels[i][0], 
                        decoder_channels[i][1], 
                        decoder_kernels[i], 
                        stride=decoder_strides[i],
                        padding=(padding_h, padding_w),
                        output_padding=output_padding
                    ),
                    nn.ReLU() if i < len(decoder_channels) - 1 else nn.Identity()  # No ReLU on final layer
                )
            )
        
    def forward(self, x):
        # Store encoder outputs for skip connections (if you want to add them)
        encoder_outputs = []
        
        # Encoder path
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            encoder_outputs.append(x)

        # Mid-level
        x = self.mid_level(x)
        
        # Decoder path
        for i in range(len(self.decoder)):
            x = self.decoder[i](x)
            
            # Add skip connections if you want (commented out for now)
            # if i < len(self.decoder) - 1:
            #     skip_idx = len(encoder_outputs) - 2 - i
            #     if skip_idx >= 0:
            #         # You might need to adjust dimensions for proper concatenation
            #         x = torch.cat([x, encoder_outputs[skip_idx]], dim=1)
            
        return x


        

class SpeechEnhancementExperiment():
    def __init__(self, batch_size=1, cuda=False, experiment_no=1, use_splits=True, augment=True, num_epochs=3, patience=5):
        self.batch_size = batch_size
        self.cuda = cuda
        self.device = torch.device("mps" if not self.cuda else "cuda")
        self.experiment_no = experiment_no
        self.use_splits = use_splits
        self.augment = augment
        self.feature_cache = {}
        self.num_epochs = num_epochs
        self.patience = patience
        self.train_loader, self.test_loader = get_loaders(batch_size=batch_size, cuda=cuda)
        self.model = None
        # self.setup_data_from_splits()
        # self.precompute_spectrograms()

    def set_snr(self, loader, snr):
        dataset = loader.dataset
        if hasattr(dataset, 'snr'):
            dataset.snr = snr
        else:
            method = getattr(dataset, 'set_snr', None)
            if method is not None and callable(method):
                method(snr)

    def set_load_waveforms(self, loader, load_waveforms):
        dataset = loader.dataset
        if hasattr(dataset, 'load_waveforms'):
            dataset.load_waveforms = load_waveforms
        else:
            method = getattr(dataset, 'set_load_waveforms', None)
            if method is not None and callable(method):
                method(load_waveforms)

    def create_experiment_dir(self, experiment_name, i):
        base_dir = f'models/{experiment_name}/{i}'
        os.makedirs(base_dir, exist_ok=True)
        return base_dir
    def initialize_result_containers(self):
        return collections.OrderedDict(
            {
            'losses': [],
            'duration': 0,
            'learning_rates': [],
            }
        )
    def test(self):
        # read model
        model = UNet().to(self.device)
        if self.model is not None:
            model.load_state_dict(self.model.state_dict())
        else:
            model.load_state_dict(torch.load(os.path.join('models/speech_enhancement_2', 'model.pth')))
        model.eval()
        istft = torchaudio.transforms.InverseSpectrogram(n_fft=512, hop_length=256, normalized=True).to(self.device)

        self.set_snr(self.test_loader, 0)

        # keep track of basenames before and after pesq and stoi
        before_pesq  = {}
        after_pesq = {}
        before_stoi = {}
        after_stoi = {}

        before_pesq_avg = 0
        after_pesq_avg = 0
        before_stoi_avg = 0
        after_stoi_avg = 0
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing", unit="batch")
            for batch in pbar:
                noisy_list, clean_list, base_name = batch
                noisy_spec = compute_spectrogram(noisy_list, device=self.device, normalize=True)
                clean_spec = compute_spectrogram(clean_list, device=self.device, normalize=True)

                enhanced = model(noisy_spec.abs())
                enhanced = enhanced.to(torch.complex64)

                # lets take the imaginary part of the noisy spectrogram and add it to the enhanced spectrogram
                noisy_spec = noisy_spec.to(torch.complex64)
                noisy_spec = torch.imag(noisy_spec)

                # lets verify there's at least one non-zero value in the noisy_spec
                if torch.sum(noisy_spec) == 0:
                    continue

                enhanced = enhanced + noisy_spec

                enhanced = enhanced.to(torch.complex64)
                enhanced = istft(enhanced)

                # pesq and stoi are not multi batch so we need to loop through the batch
                for i, (noisy, clean, base_name) in enumerate(zip(noisy_list, clean_list, base_name)):
                    if base_name not in before_pesq:
                        before_pesq[base_name] = 0
                        after_pesq[base_name] = 0
                        before_stoi[base_name] = 0
                        after_stoi[base_name] = 0
                    before_pesq[base_name] += pesq(16000, clean.squeeze(0).to('cpu').numpy(), noisy.squeeze(0).to('cpu').numpy(), 'wb')
                    after_pesq[base_name] += pesq(16000, clean.squeeze(0).to('cpu').numpy(), enhanced[i].squeeze(0).to('cpu').numpy(), 'wb')
                    before_stoi[base_name] += stoi(clean.squeeze(0).to('cpu').numpy(), noisy.squeeze(0).to('cpu').numpy(), 16000, extended=False)
                    after_stoi[base_name] += stoi(clean.squeeze(0).to('cpu').numpy(), enhanced[i].squeeze(0).to('cpu').numpy(), 16000, extended=False)

                # let's get overall average pesq and stoi

                pbar.set_postfix({
                    'pesq': f'{np.mean(list(before_pesq.values()))} -> {np.mean(list(after_pesq.values()))}',
                    'stoi': f'{np.mean(list(before_stoi.values()))} -> {np.mean(list(after_stoi.values() ))}'
                    })
                

                # lets get the top 5 basenames that did the best after enhancement 
                # done by subtracting before from after
            after_pesq_values = list(after_pesq.values())
            before_pesq_values = list(before_pesq.values())
            after_stoi_values = list(after_stoi.values())
            before_stoi_values = list(before_stoi.values())
            pesq_diff = [after_pesq_values[i] - before_pesq_values[i] for i in range(len(after_pesq_values))]
            stoi_diff = [after_stoi_values[i] - before_stoi_values[i] for i in range(len(after_stoi_values))]

            pesq_stoi_diff = [pesq_diff[i] + stoi_diff[i] for i in range(len(pesq_diff))]
            # get the top 5 indices
            top_5_pesq_indices = np.argsort(pesq_diff)[-5:]
            top_5_stoi_indices = np.argsort(stoi_diff)[-5:]
            top_5_pesq_stoi_indices = np.argsort(pesq_stoi_diff)[-5:]
            # get the basenames
            top_5_pesq_basenames = [list(after_pesq.keys())[i] for i in top_5_pesq_indices]
            top_5_stoi_basenames = [list(after_pesq.keys())[i] for i in top_5_stoi_indices]
            top_5_pesq_stoi_basenames = [list(after_pesq.keys())[i] for i in top_5_pesq_stoi_indices]
            # print(top_5_pesq_basenames)
            # print(top_5_stoi_basenames)
            # print(top_5_pesq_stoi_basenames)

            basenames = set(top_5_pesq_basenames + top_5_stoi_basenames + top_5_pesq_stoi_basenames)

            os.makedirs(os.path.join('models/speech_enhancement_2', 'test_wavs'), exist_ok=True)
            for base_name in basenames:
                # find the base_name in the loader
                for noisy, clean, base_names in self.test_loader:
                    if base_name in base_names:
                        # get the index of the base_name
                        index = base_names.index(base_name)
                        # save the wavs
                        torchaudio.save(f"models/speech_enhancement_2/test_wavs/{base_name}_noisy.wav", noisy[index], 16000)
                        torchaudio.save(f"models/speech_enhancement_2/test_wavs/{base_name}_clean.wav", clean[index], 16000)

                        # enhanced
                        noisy_spec = compute_spectrogram(noisy[index], device=self.device, normalize=True)
                        clean_spec = compute_spectrogram(clean[index], device=self.device, normalize=True)
                        # add batch dimension
                        noisy_spec = noisy_spec.unsqueeze(0)
                        clean_spec = clean_spec.unsqueeze(0)
                        enhanced = model(noisy_spec.abs())
                        enhanced = enhanced.to(torch.complex64)

                        # lets take the imaginary part of the noisy spectrogram and add it to the enhanced spectrogram
                        noisy_spec = noisy_spec.to(torch.complex64)
                        noisy_spec = torch.imag(noisy_spec)

                        enhanced = enhanced + noisy_spec

                        enhanced = enhanced.to(torch.complex64)
                        enhanced = istft(enhanced)

                        # save the wavs
                        torchaudio.save(f"models/speech_enhancement_2/test_wavs/{base_name}_enhanced.wav", enhanced.squeeze(0).to('cpu'), 16000) 
            # lets save the metadata with the basenames 
            with open(os.path.join('models/speech_enhancement_2', 'test_wavs', 'metadata.json'), 'w') as f:
                json.dump({
                    'before_pesq': before_pesq,
                    'after_pesq': after_pesq,
                    'before_stoi': before_stoi,
                    'after_stoi': after_stoi
                }, f)
    def run(self):
        criterion = nn.MSELoss()

        model = UNet().to(self.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_loss = float('inf')
        epochs_no_improve = 0
        losses = []


        for epoch in range(self.num_epochs):
            running_loss = 0.0

            counter = 0
            pbar = tqdm(
                self.train_loader,
                desc=f"[Epoch {epoch + 1}/{self.num_epochs}] [LR: {optimiser.param_groups[0]['lr']}]",
                unit="batch"
            )
            count = 0
            for batch in pbar: 
                count += 1
                if not self.cuda and count > 40:
                    # in non cuda mode, we are testing so we can break after first batch
                    break
                noisy_list, clean_list, base_name = batch
                # compute spectrograms
                noisy_spec = compute_spectrogram(noisy_list, device=self.device, normalize=True)
                clean_spec = compute_spectrogram(clean_list, device=self.device, normalize=True)

                noisy_spec = noisy_spec.abs()
                clean_spec = clean_spec.abs()

                noisy_spec = noisy_spec.to(self.device)
                clean_spec = clean_spec.to(self.device)

                optimiser.zero_grad()
                output = model(noisy_spec)
                loss = criterion(output, clean_spec)
                loss.backward()
                optimiser.step()

                running_loss += loss.item()
                
                avg_loss = running_loss / len(self.train_loader)
                pbar.set_postfix({
                    'loss': f'{running_loss:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                })
            losses.append(avg_loss)


            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        logger.info("Training phase completed. Starting results collection and analysis...")


        os.makedirs(os.path.join('models/speech_enhancement_2'), exist_ok=True)

        # save losses
        with open(os.path.join('models/speech_enhancement_2', 'losses.csv'), 'w') as f:
            for loss in losses:
                f.write(f"{loss}\n")

        # save model
        torch.save(model.state_dict(), os.path.join('models/speech_enhancement_2', 'model.pth'))
        self.model = model
                

if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_no", type=int)
    parser.add_argument("--cuda", action='store_true', default=False)
    parser.add_argument("--no_splits", action='store_true', default=False, 
                        help="If set, don't use pre-generated splits")
    parser.add_argument("--no_augment", action='store_true', default=False,
                        help="If set, don't use data augmentation and use cached features")
    args = parser.parse_args()

    # if arg is not provided, default to 1
    # but warn 
    if args.experiment_no is None:
        print("No experiment number provided. Defaulting to 1.")
        experiment_no = 1
    else:
        experiment_no = args.experiment_no
    cuda = args.cuda
    use_splits = not args.no_splits
    augment = not args.no_augment
    exp = SpeechEnhancementExperiment(batch_size=4 if not cuda else 16, cuda=cuda, experiment_no=experiment_no, use_splits=use_splits, augment=augment, num_epochs=20 if not cuda else 60, patience=5)
    exp.run()
    exp.test()
    logger.info("Done.")