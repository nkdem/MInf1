import os
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
import numpy as np
import librosa
from pesq import pesq
from pystoi import stoi
import soundfile as sf
from models import CRNN
from voicebank_dataset import get_loaders

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoicebankEnhancementExperiment:
    def __init__(self, cuda, experiment_no):
        self.experiment_no = experiment_no
        self.base_dir = f'experiments/voicebank-enh-exp-{experiment_no}'
        self.cuda = cuda
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.batch_size = 16 if cuda else 4
        self.lr = 0.0006
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epochs = 120
        self.patience = 5
        self.patience_counter = 0
        self.hop_length = 160
        self.win_length = 320
        self._setup_data()

    def _setup_data(self):
        """Set up data loaders"""
        self.train_loader, self.test_loader = get_loaders(cuda=self.cuda, batch_size=self.batch_size)
        self.test_loader.dataset.train = False

    def train(self):
        model = CRNN().to(self.device)
        losses = []
        best_loss = float('inf')

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", total=len(self.train_loader))
            avg_loss = 0
            for i, batch in enumerate(pbar):
                noisy, clean, basename = batch
                noisy_mag, clean_mag = noisy[0], clean[0]
                noisy_mag = noisy_mag.to(self.device)  # [T, F]
                clean_mag = clean_mag.to(self.device)  # [T, F]

                optimizer.zero_grad()
                output = model(noisy_mag)  # [T, F]
                loss = criterion(output, clean_mag)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                pbar.set_postfix({
                    'avg_loss': avg_loss / (i + 1)
                })
            avg_loss /= len(self.train_loader)
            losses.append(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save final model
        return model, losses

    def test(self, model):
        model.eval()
        before_pesq = {}
        after_pesq = {}
        before_stoi = {}
        after_stoi = {}

        pbar = tqdm(self.test_loader, desc="Testing", unit="batch")
        for batch in pbar:
            noisy, clean, basename, (noisy_wav, clean_wav) = batch
            noisy_mag = torch.tensor(noisy[0], device=self.device)  # [B, T, F]
            clean_mag = torch.tensor(clean[0], device=self.device)  # [B, T, F]
            noisy_phase = noisy[1].detach().cpu().numpy()  # [B, T, F]

            with torch.no_grad():
                enhanced_mag = model(noisy_mag)  # [B, T, F]
                enhanced_mag = enhanced_mag.permute(0, 2, 1).detach().cpu().numpy()  # [B, T, F] => [B, F, T]

            # librosa operates in [F, T]
            noisy_phase = np.permute_dims(noisy_phase, (0, 2, 1))  # [B, T, F] => [B, F, T]

            enhanced = librosa.istft(enhanced_mag * noisy_phase, 
                                   hop_length=self.hop_length, 
                                   win_length=self.win_length, 
                                   length=32768)

            # iterate through each sample
            for i in range(len(basename)):
                if basename[i] not in before_pesq:
                    before_pesq[basename[i]] = []
                    before_stoi[basename[i]] = []
                    after_pesq[basename[i]] = []
                    after_stoi[basename[i]] = []

                before_pesq[basename[i]].append(pesq(16000, clean_wav[i].numpy(), noisy_wav[i].numpy()))
                before_stoi[basename[i]].append(stoi(clean_wav[i].numpy(), noisy_wav[i].numpy(), 16000, extended=True))
                after_pesq[basename[i]].append(pesq(16000, clean_wav[i].numpy(), enhanced[i]))
                after_stoi[basename[i]].append(stoi(clean_wav[i].numpy(), enhanced[i], 16000, extended=True))

            pbar.set_postfix(
                {
                    'pesq': f'{np.mean(list(before_pesq.values())):.2f} -> {np.mean(list(after_pesq.values())):.2f}',
                    'stoi': f'{np.mean(list(before_stoi.values())):.2f} -> {np.mean(list(after_stoi.values())):.2f}'
                }
            )

        return before_pesq, after_pesq, before_stoi, after_stoi

    def run(self):
        os.makedirs(self.base_dir, exist_ok=True)
        model, losses = self.train()
        before_pesq, after_pesq, before_stoi, after_stoi = self.test(model)
        torch.save(model.state_dict(), f'{self.base_dir}/model.pth')
        
        # Print final results
        print("\nFinal Results:")
        print(f"PESQ: {np.mean(list(before_pesq.values())):.2f} -> {np.mean(list(after_pesq.values())):.2f}")
        print(f"STOI: {np.mean(list(before_stoi.values())):.2f} -> {np.mean(list(after_stoi.values())):.2f}")

        # save the results
        with open(f'{self.base_dir}/results.pkl', 'wb') as f:
            pickle.dump({
                'before_pesq': before_pesq,
                'after_pesq': after_pesq,
                'before_stoi': before_stoi,
                'after_stoi': after_stoi,
                'losses': losses
            }, f)

        # save the model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action='store_true', default=True)
    parser.add_argument("--experiment_no", type=int, default=1)
    args = parser.parse_args()

    experiment = VoicebankEnhancementExperiment(args.cuda, args.experiment_no)
    experiment.run() 