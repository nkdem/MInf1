import os
    
import random

import numpy as np
import torch
import torchaudio.transforms as T


def seed_everything(seed=42):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def get_truly_random_seed_through_os():
    """
    Usually the best random sample you could get in any programming language is generated through the operating system. 
    In Python, you can use the os module.

    source: https://stackoverflow.com/questions/57416925/best-practices-for-generating-a-random-seeds-to-seed-pytorch/57416967#57416967
    """
    RAND_SIZE = 4
    random_data = os.urandom(
        RAND_SIZE
    )  # Return a string of size random bytes suitable for cryptographic use.
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed

def calculate_rms(signal):
    """Calculate RMS of a signal"""
    return np.sqrt(np.mean(np.square(signal)))

def adjust_snr(speech, noise, target_snr):
    """Adjust noise level to match target SNR with speech"""
    speech_rms = calculate_rms(speech)
    noise_rms = calculate_rms(noise)
    
    adjustment = speech_rms / (10 ** (target_snr / 20)) / noise_rms
    adjusted_noise = noise * adjustment
    
    return adjusted_noise

def mix_signals(speech, background, target_snr):
    """Mix speech and background at target SNR"""
    adjusted_background = adjust_snr(speech, background, target_snr)
    mixed = speech + adjusted_background
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1:
        mixed = mixed / max_val
    
    return mixed


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

def compute_average_logmel(
    audio_batch,
    device,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    n_mels: int = 40,
) -> torch.Tensor:
    """
    Compute the average log-mel spectrogram for a batch of stereo waveforms.
    This function is optimised to emulate your original processing speed.

    Parameters:
        audio_batch (list): A list where each element is a list/tuple containing two waveforms 
                            (left and right channels). Waveforms can be numpy arrays or torch.Tensors.
        device (torch.device): The device to use.
        sample_rate (int): Sample rate of the audio signals.
        n_fft (int): FFT size.
        n_mels (int): Number of mel bands.

    Returns:
        torch.Tensor: Averaged log-mel spectrogram with shape (batch_size, 1, n_mels, n_frames)
    """
    # Convert and stack the left and right channels similar to your original approach.
    batch_waveforms_l = torch.stack([
        process_waveform(a[0], device)
        for a in audio_batch
    ])
    batch_waveforms_r = torch.stack([
        process_waveform(a[1], device)
        for a in audio_batch
    ])

    # Define transformation parameters.
    hop_length = int(0.02 * sample_rate)  # 20 ms
    win_length = int(0.04 * sample_rate)    # 40 ms

    # Cache the MelSpectrogram transform to avoid re-instantiation.
    cache_key = (device, sample_rate, n_fft, win_length, hop_length, n_mels)
    if not hasattr(compute_average_logmel, "_mel_cache"):
        compute_average_logmel._mel_cache = {}
    if cache_key in compute_average_logmel._mel_cache:
        mel_transform = compute_average_logmel._mel_cache[cache_key]
    else:
        mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
        ).to(device)
        compute_average_logmel._mel_cache[cache_key] = mel_transform

    # Compute mel spectrograms for left and right channels.
    mel_l = mel_transform(batch_waveforms_l)
    mel_r = mel_transform(batch_waveforms_r)

    # Compute log-mel spectrograms.
    logmel_l = 20 * torch.log10(mel_l + 1e-10)
    logmel_r = 20 * torch.log10(mel_r + 1e-10)

    # Average the two channels.
    avg_logmel = (logmel_l + logmel_r) / 2

    # Add a channel dimension to match expected shape: (batch_size, 1, n_mels, n_frames)
    return avg_logmel.unsqueeze(1)

