import os
from pathlib import Path
import random
import json

def get_speaker_utterances(timit_base_path):
    male_speakers = {}
    female_speakers = {}
    
    # Use the 'original' directory
    timit_path = Path(timit_base_path) / 'original'
    
    # Look in both train and test directories
    for split in ['train', 'test']:
        split_path = timit_path / split
        
        # Go through each dialect region (dr1-dr8)
        for dr in split_path.glob('dr*'):
            # List all speakers in this dialect region
            for speaker_dir in dr.glob('*'):
                if speaker_dir.is_dir():
                    speaker_id = speaker_dir.name.lower()  # Keep lowercase for consistency
                    # Get all wav files for this speaker
                    wav_files = list(speaker_dir.glob('*.wav'))
                    speaker_info = {
                        'split': split,
                        'dialect_region': dr.name,
                        'utterances': [str(wav.relative_to(timit_path)) for wav in wav_files]
                    }
                    
                    if speaker_id.startswith('f'):
                        female_speakers[speaker_id] = speaker_info
                    elif speaker_id.startswith('m'):
                        male_speakers[speaker_id] = speaker_info

    return male_speakers, female_speakers

def create_balanced_selection(male_speakers, female_speakers):
    # Get number of female speakers
    n_females = len(female_speakers)
    
    # Randomly select the same number of male speakers
    selected_male_ids = random.sample(list(male_speakers.keys()), n_females)
    
    # Create final selection
    selected_speakers = {
        'female': female_speakers,
        'male': {speaker_id: male_speakers[speaker_id] for speaker_id in selected_male_ids}
    }
    
    return selected_speakers

def print_statistics(selected_speakers):
    n_female = len(selected_speakers['female'])
    n_male = len(selected_speakers['male'])
    
    print(f"\nBalanced Selection Statistics:")
    print(f"Female speakers: {n_female}")
    print(f"Male speakers: {n_male}")
    
    # Count utterances
    female_utts = sum(len(spk['utterances']) for spk in selected_speakers['female'].values())
    male_utts = sum(len(spk['utterances']) for spk in selected_speakers['male'].values())
    
    print(f"\nTotal utterances:")
    print(f"Female: {female_utts}")
    print(f"Male: {male_utts}")
    print(f"Total: {female_utts + male_utts}")
    
    # Add detailed statistics per split
    train_female = sum(1 for spk in selected_speakers['female'].values() if spk['split'] == 'train')
    train_male = sum(1 for spk in selected_speakers['male'].values() if spk['split'] == 'train')
    test_female = sum(1 for spk in selected_speakers['female'].values() if spk['split'] == 'test')
    test_male = sum(1 for spk in selected_speakers['male'].values() if spk['split'] == 'test')
    
    print(f"\nSplit Statistics:")
    print(f"Train set - Female: {train_female}, Male: {train_male}")
    print(f"Test set  - Female: {test_female}, Male: {test_male}")

# Path to your TIMIT directory (parent directory containing 'original', 'ntimit', etc.)
timit_base_path = "/Volumes/SSD/Datasets/TIMIT"

# Get speaker lists
male_speakers, female_speakers = get_speaker_utterances(timit_base_path)
selected_speakers = create_balanced_selection(male_speakers, female_speakers)

# Print statistics
print_statistics(selected_speakers)

# Save the selection to a JSON file for later use
with open('timit_balanced_selection.json', 'w') as f:
    json.dump(selected_speakers, f, indent=2)

print("\nSelection saved to 'timit_balanced_selection.json'")
