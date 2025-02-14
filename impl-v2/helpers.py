import os
import random

import numpy as np
import torch


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