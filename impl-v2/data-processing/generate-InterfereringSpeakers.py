import json
import os
import numpy as np
import soundfile as sf
from pathlib import Path
import librosa
from datetime import datetime

class ChimeProcessor:
    def __init__(self, chime_root, output_dir):
        self.chime_root = Path(chime_root)  # This should point to the CHiME6 directory
        self.output_dir = Path(output_dir)
        self.sample_rate = 16000  # HEAR-DS uses 16kHz
        self.snippet_duration = 10  # seconds
        
        # Create output directories for dev, eval and train
        self.dev_dir = self.output_dir / 'dev'
        self.eval_dir = self.output_dir / 'eval'
        self.train_dir = self.output_dir / 'train'  # Added train directory

        self.dev_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir.mkdir(parents=True, exist_ok=True)
        
        # Global counter to be used for assigning env_id alternation
        self.global_segment_count = 0

    def load_transcription(self, session, split):
        """Load JSON transcription file for a session."""
        trans_path = self.chime_root / 'transcriptions' / split / f'{session}.json'
        if not trans_path.exists():
            raise FileNotFoundError(f"Transcription file not found: {trans_path}")
        with open(trans_path, 'r') as f:
            return json.load(f)

    def get_session_audio_files(self, session, split):
        """Get participant worn microphone files for a session."""
        audio_dir = self.chime_root / 'audio' / split
        if not audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        return sorted(list(audio_dir.glob(f'{session}_P*.wav')))

    def time_str_to_seconds(self, time_str):
        """Convert time string 'H:MM:SS.xx' to seconds."""
        time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
        return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond/1000000

    def find_overlapping_speech(self, utterances, duration=10.0):
        """Find segments with overlapping speech."""
        segments = []
        
        # Convert utterances to time segments with speaker info
        time_segments = []
        for utt in utterances:
            start = self.time_str_to_seconds(utt['start_time'])
            end = self.time_str_to_seconds(utt['end_time'])
            speaker = utt['speaker']
            time_segments.append((start, end, speaker))
        
        # Sort by start time
        time_segments.sort(key=lambda x: x[0])
        
        # Find overlapping segments
        for i in range(len(time_segments)):
            current_start, current_end, current_speaker = time_segments[i]
            
            # Look for overlaps within duration window
            window_end = current_start + duration
            if window_end > current_end:  # Skip if segment is too short
                continue
                
            overlaps = []
            
            for j in range(len(time_segments)):
                if i != j:
                    other_start, other_end, other_speaker = time_segments[j]
                    if (other_start < window_end and 
                        other_end > current_start and 
                        other_start >= current_start):
                        overlaps.append((other_speaker, other_start, other_end))
            
            # If we found overlapping speech
            if overlaps:
                segments.append({
                    'start': current_start,
                    'duration': duration,
                    'speakers': [current_speaker] + [o[0] for o in overlaps],
                    'reference_speaker': current_speaker
                })
                
        return segments

    def process_audio_segment(self, audio_file, start_time, duration):
        """Extract and process audio segment."""
        try:
            audio, sr = librosa.load(audio_file, 
                                   sr=self.sample_rate,
                                   offset=start_time,
                                   duration=duration,
                                   mono=False)  # Keep stereo channels
            return audio
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            return None

    def process_session(self, session, split):
        """Process one session."""
        print(f"Processing {split} session {session}...")
        
        try:
            # Load transcription
            utterances = self.load_transcription(session, split)
            
            # Find segments with overlapping speech
            segments = self.find_overlapping_speech(utterances)
            print(f"Found {len(segments)} segments with overlapping speech")
            
            # Get participant audio files (P*)
            p_files = self.get_session_audio_files(session, split)
            print(f"Found audio files: {[p.name for p in p_files]}")
            
            # Select output directory based on split
            if split == 'dev':
                output_dir = self.dev_dir
            elif split == 'eval':
                output_dir = self.eval_dir
            else:  # train
                output_dir = self.train_dir
                
            # Make sure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each segment
            for i, segment in enumerate(segments):
                try:
                    # Determine env_id based on the global counter (half samples 100, half 101)
                    env_id = "100" if self.global_segment_count % 2 == 0 else "101"
                    
                    # REC_ID: derived from session number by stripping the 'S' and adding 200
                    rec_num = int(session[1:])  # e.g., "S02" -> 2
                    rec_id = f"{rec_num + 200:03d}"  # S02 -> 202
                    
                    cut_id = "00"  # fixed as "00"
                    snip_id = f"{i:03d}"  # segment index as SNIP_ID
                    
                    # Find the P* file corresponding to the reference speaker
                    ref_speaker = segment['reference_speaker']
                    speaker_num = int(ref_speaker[1:])  # Extract number from P1, P2, etc.
                    matching_files = [f for f in p_files if f'{speaker_num:02d}' in f.name]
                    
                    if not matching_files:
                        print(f"No matching audio file found for speaker {ref_speaker}")
                        continue
                        
                    p_file = matching_files[0]
                    
                    # Extract audio segment
                    audio = self.process_audio_segment(
                        p_file, 
                        segment['start'], 
                        self.snippet_duration
                    )
                    
                    if audio is None:
                        print(f"Failed to process audio segment from {p_file}")
                        continue
                    
                    # Save left and right channels
                    for ch_idx, side in enumerate(['L', 'R']):
                        track_name = f"ITC_{side}"
                        filename = f"{env_id}_{rec_id}_{cut_id}_{snip_id}_{track_name}_16kHz.wav"
                        output_path = output_dir / filename
                        
                        try:
                            # Save individual channel; ensure audio has at least the expected number of channels
                            sf.write(str(output_path), audio[ch_idx], self.sample_rate)
                        except Exception as e:
                            print(f"Error saving file {output_path}: {e}")
                            continue
                    
                    self.global_segment_count += 1
                    if i % 10 == 0:
                        print(f"Processed {i+1}/{len(segments)} segments")
                        
                except Exception as e:
                    print(f"Error processing segment {i} in session {session}: {e}")
                    continue
                        
            print(f"Successfully processed {self.global_segment_count} segments in session {session}")
                
        except Exception as e:
            print(f"Error processing session {session}: {e}")

    def process_all(self):
        """Process all sessions in both dev and eval sets."""
        # Process dev set (S02 and S09)
        print("Processing dev set...")
        for session in ['S02', 'S09']:
            self.process_session(session, 'dev')
            
        # Process eval set (S01 and S21)
        print("\nProcessing eval set...")
        for session in ['S01', 'S21']:
            self.process_session(session, 'eval')
            
        # Process train set (S03-S08, S12-S13, S16-S20, S22-S24)
        print("\nProcessing train set...")
        train_sessions = [
            'S03', 'S04', 'S05', 'S06', 'S07', 'S08',
            'S12', 'S13',
            'S16', 'S17', 'S18', 'S19', 'S20',
            'S22', 'S23', 'S24'
        ]
        for session in train_sessions:
            self.process_session(session, 'train')
            
        # Print summary
        dev_files = list(self.dev_dir.glob('*.wav'))
        eval_files = list(self.eval_dir.glob('*.wav'))
        train_files = list(self.train_dir.glob('*.wav'))
        print(f"\nSummary:")
        print(f"Dev set: {len(dev_files)//2} segments")
        print(f"Eval set: {len(eval_files)//2} segments")
        print(f"Train set: {len(train_files)//2} segments")
        
        # Print detailed summary
        print("\nDetailed Summary:")
        for split in ['dev', 'eval', 'train']:
            dir_path = getattr(self, f'{split}_dir')
            files = list(dir_path.glob('*.wav'))
            print(f"{split.capitalize()} set:")
            print(f"  - Segments processed: {len(files)//2}")
        
        # Merge the dev, eval and train sets into a single folder
        print("\nMerging train, dev and eval sets into a single folder...")
        merged_dir = self.output_dir / 'merged'
        merged_dir.mkdir(parents=True, exist_ok=True)
        
        # Move files from dev, eval and train to merged
        for split in ['dev', 'eval', 'train']:
            dir_path = getattr(self, f'{split}_dir')
            files = list(dir_path.glob('*.wav'))
            for f in files:
                f.rename(merged_dir / f.name)
        
        # Rename merged folder to "InterferingSpeakers"
        merged_dir.rename(self.output_dir / 'InterferingSpeakers')

        self.dev_dir.rmdir()
        self.eval_dir.rmdir()
        self.train_dir.rmdir()


def main():
    # Setup paths - adjust these paths according to your system
    chime_root = Path('/Volumes/SSD/Datasets/CHiME6')  # Point to the CHiME6 directory
    output_dir = Path('/Volumes/SSD/Datasets/HEAR-DS/Down-Sampled/InterfereringSpeakers')
    
    # Create processor and run
    processor = ChimeProcessor(chime_root, output_dir)
    processor.process_all()

if __name__ == "__main__":
    main()
