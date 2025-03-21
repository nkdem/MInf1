import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft

# Load the WAV file
file_path = '/Users/nkdem/Downloads/HEAR-DS/WindTurbulence/All/06_203_39_012_ITC_L_16kHz.wav'  # Replace with your file path
sample_rate, sound_wave = wavfile.read(file_path)

# Ensure sample rate is 16kHz
target_sample_rate = 16000
if sample_rate != target_sample_rate:
    num_samples = len(sound_wave)
    new_num_samples = int(num_samples * target_sample_rate / sample_rate)
    sound_wave = np.interp(
        np.linspace(0, 1, new_num_samples),
        np.linspace(0, 1, num_samples),
        sound_wave
    )
    sample_rate = target_sample_rate

# Normalize the wave if it's not already in the range [-1, 1]
if sound_wave.dtype == 'int16':
    sound_wave = sound_wave / (2**15)

# Ensure it's mono if it's stereo
if len(sound_wave.shape) > 1:
    sound_wave = sound_wave[:, 0]

# Time array
duration = len(sound_wave) / sample_rate
t = np.linspace(0, duration, len(sound_wave), False)

# Plot and save the sound wave
plt.figure(figsize=(8, 8))
plt.plot(t, sound_wave, color='blue')
plt.axis('off')
plt.box(on=True)
# Add border
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')
plt.savefig('waveform.png', bbox_inches='tight', dpi=300, edgecolor='black')
plt.close()

# Compute the STFT with specific parameters for 500 time bins
n_fft = 1024  # Changed from 2048
desired_time_bins = 500

# Calculate hop_length to achieve desired time bins
signal_length = len(sound_wave)
hop_length = max(1, int(np.ceil(signal_length / desired_time_bins)))

frequencies, times, Zxx = stft(sound_wave, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length)

# Convert to Mel scale
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

# Define Mel filter bank
n_mels = 40  # Already at 40
mel_frequencies = np.linspace(hz_to_mel(0), hz_to_mel(sample_rate / 2), n_mels + 2)
mel_frequencies_hz = mel_to_hz(mel_frequencies)
mel_filters = np.zeros((n_mels, int(1 + n_fft // 2)))

for i in range(n_mels):
    m1 = mel_frequencies_hz[i]
    m2 = mel_frequencies_hz[i + 2]
    mel_filters[i, :] = np.maximum(0, np.minimum((frequencies - m1) / (m2 - m1), (m2 - frequencies) / (m2 - m1)))

# Apply Mel filters to the STFT magnitude
S = np.abs(Zxx)
mel_spec = np.dot(mel_filters, S)

# Convert to dB scale
mel_spec_db = 20 * np.log10(np.maximum(mel_spec, 1e-5))

# Plot and save the Mel spectrogram
plt.figure(figsize=(8, 8))
plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='viridis', extent=[times.min(), times.max(), 0, n_mels])
plt.axis('off')
plt.box(on=True)
# Add border
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
for spine in plt.gca().spines.values():
    spine.set_linewidth(2)
    spine.set_color('black')
plt.savefig('mel_spectrogram.png', bbox_inches='tight', dpi=300, edgecolor='black')
plt.close()

# Print the actual dimensions of the spectrogram
print(f"Mel-spectrogram shape: {mel_spec_db.shape}")  # Should be close to (40, 500)
