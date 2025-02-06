import numpy as np
import soundfile as sf
from pathlib import Path
import random
import json
from scipy import signal
from tqdm import tqdm

def calculate_rms(audio):
    return np.sqrt(np.mean(np.square(audio)))

def adjust_snr(speech, background, target_snr):
    speech_rms = calculate_rms(speech)
    background_rms = calculate_rms(background)
    gain = background_rms / (speech_rms * (10 ** (target_snr/20)))
    return speech * gain

def calculate_durations(timit_root_dir, timit_data):
    """Calculate and add durations to the TIMIT selection data"""
    print("Calculating utterance durations...")
    for gender in timit_data:
        for speaker_id, speaker_info in tqdm(timit_data[gender].items()):
            utterance_info = []
            for utt in speaker_info['utterances']:
                speech_path = Path(timit_root_dir) / utt
                audio, sr = sf.read(speech_path)
                duration = len(audio) / sr
                utterance_info.append({
                    'path': utt,
                    'duration': duration
                })
            speaker_info['utterances'] = utterance_info
    return timit_data

def get_utterances_for_duration(utterance_pool, target_duration):
    selected = []
    current_duration = 0
    available_duration = target_duration * 0.75  # 75% coverage
    
    available_utterances = utterance_pool.copy()
    random.shuffle(available_utterances)
    
    for utt in available_utterances:
        if current_duration + utt['duration'] <= available_duration:
            max_start = target_duration - utt['duration']
            start_time = random.uniform(0, max_start)
            
            selected.append((utt['path'], start_time, utt['duration']))
            current_duration += utt['duration']
    
    return selected

def mix_background_with_speech(background_file, utterances, timit_root_dir, target_snr, output_file):
    """Mix multiple speech utterances into one background file"""
    # Load background
    background, sr_bg = sf.read(background_file)
    if len(background.shape) > 1:
        background = np.mean(background, axis=1)
    
    # Create output buffer
    output = np.copy(background)
    mixed_regions = np.zeros_like(background, dtype=bool)
    
    # Process each utterance
    for utt_path, start_time, _ in utterances:
        # Load speech
        speech_path = Path(timit_root_dir) / utt_path
        speech, sr_speech = sf.read(speech_path)
        
        # Resample speech if needed
        if sr_speech != sr_bg:
            num_samples = int(len(speech) * sr_bg / sr_speech)
            speech = signal.resample(speech, num_samples)
        
        # Calculate start and end samples
        start_sample = int(start_time * sr_bg)
        end_sample = min(start_sample + len(speech), len(background))
        speech = speech[:end_sample - start_sample]  # Trim speech if needed
        
        # Get corresponding background segment
        bg_segment = background[start_sample:end_sample]
        
        # Adjust SNR
        speech_scaled = adjust_snr(speech, bg_segment, target_snr)
        
        # Add to output
        output[start_sample:end_sample] = bg_segment + speech_scaled
        mixed_regions[start_sample:end_sample] = True
    
    # Normalize to prevent clipping
    output = output / np.max(np.abs(output))
    
    # Save
    sf.write(output_file, output, sr_bg)
    
    coverage = np.sum(mixed_regions) / len(background) * 100
    
    return {
        'background_file': str(background_file),
        'utterances': [(str(utt), start) for utt, start, _ in utterances],
        'target_snr': target_snr,
        'coverage_percentage': coverage,
        'output_file': str(output_file)
    }

def process_dataset(timit_json, timit_root_dir, background_dir, output_dir):
    # Load or create TIMIT selection with durations
    timit_json_path = Path(timit_json)
    duration_json_path = timit_json_path.with_name(timit_json_path.stem + '_with_durations.json')
    
    if duration_json_path.exists():
        print("Loading pre-calculated durations...")
        with open(duration_json_path, 'r') as f:
            timit_data = json.load(f)
    else:
        print("First run - calculating durations...")
        with open(timit_json, 'r') as f:
            timit_data = json.load(f)
        timit_data = calculate_durations(timit_root_dir, timit_data)
        with open(duration_json_path, 'w') as f:
            json.dump(timit_data, f, indent=2)
    
    # Create flat list of all utterances with durations
    all_utterances = []
    for gender in ['male', 'female']:
        for speaker_info in timit_data[gender].values():
            all_utterances.extend(speaker_info['utterances'])
    
    # Setup paths
    timit_root_dir = Path(timit_root_dir)
    background_dir = Path(background_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # SNR levels
    snr_levels = [-10, -5, 0, 5, 10]
    
    # Store mixing metadata
    metadata = {'mixtures': []}
    
    # Process each background file
    print("\nProcessing background files...")
    background_files = list(background_dir.glob('*.wav'))
    
    # Pre-load background durations
    bg_durations = {}
    print("Loading background file durations...")
    for bg_file in tqdm(background_files):
        audio, sr = sf.read(bg_file)
        bg_durations[bg_file] = len(audio) / sr
    
    # Main processing loop
    for bg_file in tqdm(background_files):
        bg_duration = bg_durations[bg_file]
        
        for snr in snr_levels:
            # Select utterances to fill the duration
            selected_utterances = get_utterances_for_duration(all_utterances, bg_duration)
            
            output_file = output_dir / f"{bg_file.stem}_snr{snr}.wav"
            
            try:
                result = mix_background_with_speech(
                    bg_file, selected_utterances, timit_root_dir, snr, output_file)
                metadata['mixtures'].append(result)
            except Exception as e:
                print(f"Error processing {bg_file}: {e}")
    
    # Save metadata
    with open(output_dir / 'mixing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\nProcessing complete!")
    print(f"Total mixtures created: {len(metadata['mixtures'])}")

if __name__ == "__main__":
    process_dataset(
        timit_json='timit_balanced_selection.json',
        timit_root_dir='/Volumes/SSD/Datasets/TIMIT/original',
        background_dir='/Users/nkdem/Downloads/HEAR-DS/Down-Sampled/InVehicle/Background',
        output_dir='output_mixed_dataset'
    )
