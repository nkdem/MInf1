import librosa
import numpy as np
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from models import CRNN
import torch
import soundfile as sf

from voicebank_dataset import get_loaders
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
batch_size = 32 if cuda else 4
train_loader, test_loader = get_loaders(cuda=cuda, batch_size=batch_size)
test_loader.dataset.train = False

model = CRNN()
model.load_state_dict(torch.load("model_69.pth"))
model = model.to(device)

hop_length = 160
win_length = 320


before_pesq = {}
after_pesq = {}
before_stoi = {}
after_stoi = {}

pbar = tqdm(test_loader, desc="Testing", unit="batch")
for batch in pbar:
    noisy, clean, basename, (noisy_wav, clean_wav) = batch
    noisy_mag = torch.tensor(noisy[0], device=device) # [B, T, F]
    clean_mag = torch.tensor(clean[0], device=device) # [B, T, F]
    noisy_phase = noisy[1].detach().cpu().numpy() # [B, T, F]

    enhanced_mag = model(noisy_mag) # [B, T, F]
    enhanced_mag = enhanced_mag.permute(0, 2, 1).detach().cpu().numpy() # [B, T, F] => [B, F, T]


    # librosa operates in [F, T]
    noisy_phase = np.permute_dims(noisy_phase, (0, 2, 1)) # [B, T, F] => [B, F, T]

    enhanced = librosa.istft(enhanced_mag * noisy_phase, hop_length=hop_length, win_length=win_length, length=32768)

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