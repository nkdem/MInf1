import json
import os
import shutil
import numpy as np
import soundfile as sf
from pathlib import Path
import random
from tqdm.auto import tqdm

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

def ensure_valid_lengths(speech, background, target_length=160000):
    """Ensure both signals are the same length"""
    # If speech is longer than target_length, trim it
    if len(speech) > target_length:
        start = random.randint(0, len(speech) - target_length)
        speech = speech[start:start + target_length]
    
    # If speech is shorter than target_length, pad it
    elif len(speech) < target_length:
        padding = target_length - len(speech)
        speech = np.pad(speech, (0, padding), 'constant')
    
    # Do the same for background
    if len(background) > target_length:
        start = random.randint(0, len(background) - target_length)
        background = background[start:start + target_length]
    elif len(background) < target_length:
        padding = target_length - len(background)
        background = np.pad(background, (0, padding), 'wrap')
    
    return speech, background

def mix_signals(speech, background, target_snr):
    """Mix speech and background at target SNR"""
    adjusted_background = adjust_snr(speech, background, target_snr)
    mixed = speech + adjusted_background
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1:
        mixed = mixed / max_val
    
    return mixed

def create_directory_structure(base_dir):
    """Create directory structure for each environment and SNR level"""
    environments = ['InTraffic', 'InVehicle',
                   'Music', 'QuietIndoors', 'ReverberantEnvironment', 'WindTurbulence']
    
    snr_levels = [-21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21]
    
    print("Creating directory structure...")
    for env in tqdm(environments, desc="Creating directories"):
        # Create Speech directories
        speech_dir = os.path.join(base_dir, env, 'Speech')
        for snr in snr_levels:
            os.makedirs(os.path.join(speech_dir, f'{snr}'), exist_ok=True)
        
        # Create background-use directory
        bg_use_dir = os.path.join(base_dir, env, 'Background-use')
        os.makedirs(bg_use_dir, exist_ok=True)
    
    return environments, snr_levels

def safe_save_audio(file_path, audio_data, sample_rate):
    """Safely save audio file with error handling"""
    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        sf.write(file_path, audio_data, sample_rate)
        return True
    except Exception as e:
        return False

def get_split_counts():
    """Get the target counts for background and speech in background"""
    splits = {
        'InTraffic': {'background': 530, 'speech': 470},
        'InVehicle': {'background': 584, 'speech': 511},
        'Music': {'background': 1496, 'speech': 1495},
        'QuietIndoors': {'background': 525, 'speech': 426},
        'ReverberantEnvironment': {'background': 315, 'speech': 692},
        'WindTurbulence': {'background': 595, 'speech': 439}
    }
    return splits

def get_base_filename(filename):
    """Get base filename without L/R indicator and extension"""
    # Example: 01_552_16_004_ITC_L_16kHz.wav -> 01_552_16_004_ITC
    # First remove the extension
    filename = os.path.splitext(filename)[0]  # Remove .wav
    parts = filename.split('_')
    # Find ITC index
    itc_index = parts.index('ITC')
    # Take everything up to and including ITC
    base = '_'.join(parts[:itc_index + 1])
    return base

def split_background_files(background_files, bg_target_count, speech_target_count):
    """Split background files according to target counts, keeping L/R pairs together"""
    # Verify target counts are even (since we need pairs)
    if bg_target_count % 2 != 0 or speech_target_count % 2 != 0:
        print(f"Warning: Target counts ({bg_target_count}, {speech_target_count}) must be even numbers for L/R pairs")
        # Round up to nearest even number
        bg_target_count = (bg_target_count + 1) // 2 * 2
        speech_target_count = (speech_target_count + 1) // 2 * 2
        print(f"Adjusted to: {bg_target_count}, {speech_target_count}")
    
    # Group files by their base name
    file_pairs = {}
    for file in background_files:
        base = get_base_filename(file)
        if base not in file_pairs:
            file_pairs[base] = set()
        file_pairs[base].add(file)
    
    # Assert that each base has exactly 2 files (L and R)
    for base, files in file_pairs.items():
        assert len(files) == 2, f"Base {base} does not have exactly 2 files: {files}"
        # Assert one has _L_ and one has _R_
        l_count = sum(1 for f in files if '_L_' in f)
        r_count = sum(1 for f in files if '_R_' in f)
        assert l_count == 1 and r_count == 1, f"Base {base} does not have one L and one R file: {files}"
    
    # Calculate number of pairs needed
    bg_pair_count = bg_target_count // 2
    speech_pair_count = speech_target_count // 2
    
    # Get list of base names and shuffle
    base_names = list(file_pairs.keys())
    random.shuffle(base_names)
    
    # Split into background and speech sets
    bg_bases = base_names[:bg_pair_count]
    speech_bases = base_names[bg_pair_count:bg_pair_count + speech_pair_count]
    
    # Get complete file lists
    background_set = []
    speech_set = []
    
    for base in bg_bases:
        background_set.extend(file_pairs[base])
    
    for base in speech_bases:
        speech_set.extend(file_pairs[base])
    
    # Final assertions
    assert len(background_set) == bg_target_count, f"Background set size {len(background_set)} != target {bg_target_count}"
    assert len(speech_set) == speech_target_count, f"Speech set size {len(speech_set)} != target {speech_target_count}"
    
    # Assert no overlap between sets
    assert len(set(background_set) & set(speech_set)) == 0, "Overlap found between background and speech sets"
    
    # Assert L/R pairs are kept together
    for file_set in [background_set, speech_set]:
        bases_in_set = {get_base_filename(f) for f in file_set}
        for base in bases_in_set:
            l_files = [f for f in file_set if f.startswith(base) and '_L_' in f]
            r_files = [f for f in file_set if f.startswith(base) and '_R_' in f]
            assert len(l_files) == 1 and len(r_files) == 1, f"L/R pair split found for base {base}"
    
    print(f"Background pairs: {len(bg_bases)}, total files: {len(background_set)}")
    print(f"Speech pairs: {len(speech_bases)}, total files: {len(speech_set)}")
    
    return background_set, speech_set


def concatenate_speech_samples(speech_files, chime_dir, target_length):
    """Concatenate speech samples until reaching 75% coverage of target length"""
    required_length = int(0.75 * target_length)
    concatenated_speech = np.array([])
    used_files = []
    
    while len(concatenated_speech) < required_length:
        # Select random speech file
        speech_file = random.choice(speech_files)
        speech, sr = sf.read(os.path.join(chime_dir, speech_file))
        
        concatenated_speech = np.concatenate([concatenated_speech, speech])
        used_files.append(speech_file)
    
    # Trim if longer than target_length
    if len(concatenated_speech) > target_length:
        concatenated_speech = concatenated_speech[:target_length]
    
    return concatenated_speech, used_files, sr

def ensure_valid_lengths_with_speech(speech, background, target_length=160000):
    """Modified version to handle concatenated speech"""
    # If background is longer than target_length, trim it
    if len(background) > target_length:
        start = random.randint(0, len(background) - target_length)
        background = background[start:start + target_length]
    
    # If background is shorter than target_length, pad it
    elif len(background) < target_length:
        padding = target_length - len(background)
        background = np.pad(background, (0, padding), 'wrap')
    
    # Pad speech if shorter than target_length
    if len(speech) < target_length:
        padding = target_length - len(speech)
        speech = np.pad(speech, (0, padding), 'constant')
    
    return speech, background
def main():
    # Paths
    base_dir = '/Users/nkdem/Downloads/HEAR-DS/Down-Sampled'
    chime_dir = '/Volumes/SSD/Datasets/CHiME3/CHiME3-Isolated-DEV/dt05_bth'
    target_length = 160000
    
    # Get split counts
    splits = get_split_counts()
    
    # Create directory structure
    environments, snr_levels = create_directory_structure(base_dir)

    concat_speech_dir = os.path.join(base_dir, 'concatenated_speech')
    speech_mapping = {}

    os.makedirs(concat_speech_dir, exist_ok=True)
    
    # Get speech files (CH0 only)
    print("Loading speech files...")
    speech_files = [f for f in os.listdir(chime_dir) if f.endswith('CH0.wav')]
    male_speech = [f for f in speech_files if f.startswith('M')]
    female_speech = [f for f in speech_files if f.startswith('F')]
    
    print(f"Found {len(male_speech)} male and {len(female_speech)} female speech files")
    
    # Process each environment
    for env in tqdm(environments, desc="Processing environments", position=0):
        background_dir = os.path.join(base_dir, env, 'Background')
        bg_use_dir = os.path.join(base_dir, env, 'Background-use')
        background_files = [f for f in os.listdir(background_dir) if f.endswith('.wav')]
        
        # Split background files according to target counts
        target_bg_count = splits[env]['background']
        target_speech_count = splits[env]['speech']
        
        print(f"\nEnvironment: {env}")
        print(f"Total files: {len(background_files)}")
        print(f"Target background: {target_bg_count}")
        print(f"Target speech: {target_speech_count}")
        
        bg_files_for_bg, bg_files_for_mixing = split_background_files(
            background_files, 
            target_bg_count,
            target_speech_count
        )
        
        # Copy pure background files to Background-use
        for bg_file in bg_files_for_bg:
            src = os.path.join(background_dir, bg_file)
            dst = os.path.join(bg_use_dir, bg_file)
            shutil.copy2(src, dst)
        
        # Create progress bars for background files and SNR levels
        with tqdm(total=len(bg_files_for_mixing) * len(snr_levels), desc=f"{env} progress", position=1, leave=True) as pbar:
            bg_file_pairs = {}
            for bg_file in bg_files_for_mixing:
                base = get_base_filename(bg_file)
                if base not in bg_file_pairs:
                    bg_file_pairs[base] = []
                bg_file_pairs[base].append(bg_file)
            
            # Now process each pair
            for i, (base, pair) in enumerate(bg_file_pairs.items()):
                l_file = next(f for f in pair if '_L_' in f)
                r_file = next(f for f in pair if '_R_' in f)
                
                try:
                    # Load background audio for both channels
                    background_l, bg_sr_l = sf.read(os.path.join(background_dir, l_file))
                    background_r, bg_sr_r = sf.read(os.path.join(background_dir, r_file))
                    
                    # Mix at different SNR levels
                    for snr in snr_levels:
                        # Create concatenated speech for this mix
                        speech_files = male_speech if i % 2 == 0 else female_speech
                        concatenated_speech, used_files, sp_sr = concatenate_speech_samples(
                            speech_files, chime_dir, target_length)
                        
                        # Save concatenated speech
                        concat_filename = f"{env}_{base}_{snr}_concat.wav"
                        concat_path = os.path.join(concat_speech_dir, concat_filename)
                        sf.write(concat_path, concatenated_speech, sp_sr)
                        
                        # Update speech mapping
                        speech_mapping[f"{l_file}_{snr}"] = {
                            'concatenated_speech': concat_filename,
                            'used_files': used_files
                        }
                        speech_mapping[f"{r_file}_{snr}"] = {
                            'concatenated_speech': concat_filename,
                            'used_files': used_files
                        }
                        
                        # Ensure sampling rates match
                        if bg_sr_l != sp_sr or bg_sr_r != sp_sr:
                            continue
                        
                        # Ensure valid lengths for both channels
                        speech_l, background_l_chunk = ensure_valid_lengths_with_speech(
                            concatenated_speech, background_l, target_length)
                        speech_r, background_r_chunk = ensure_valid_lengths_with_speech(
                            concatenated_speech, background_r, target_length)
                        
                        # Mix signals for both channels
                        mixed_l = mix_signals(speech_l, background_l_chunk, snr)
                        mixed_r = mix_signals(speech_r, background_r_chunk, snr)
                        
                        # Save mixed audio for both channels
                        output_dir = os.path.join(base_dir, env, 'Speech', f'{snr}')
                        if (safe_save_audio(os.path.join(output_dir, l_file), mixed_l, sp_sr) and
                            safe_save_audio(os.path.join(output_dir, r_file), mixed_r, sp_sr)):
                            pbar.update(2)
                
                except Exception as e:
                    print(f"Error processing pair {l_file}, {r_file}: {str(e)}")
                    continue
        
        # Save speech mapping to JSON
        mapping_file = os.path.join(base_dir, 'speech_mapping.json')
        with open(mapping_file, 'w') as f:
            json.dump(speech_mapping, f, indent=4)
if __name__ == "__main__":
    main()
