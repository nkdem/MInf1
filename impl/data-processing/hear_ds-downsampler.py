import os
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np

class HEARDownsampler:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_duration = 10  # seconds
        
        # Define environment IDs (starting from 0)
        self.env_ids = {
            'CocktailParty': ('00', 'CocktailParty'),
            # 'InterfereringSpeakers': ('01', 'InterfereringSpeakers'),
            'InTraffic': ('01', 'InTraffic'),
            'InVehicle': ('02', 'InVehicle'),
            'Music': ('03', 'Music'),
            'QuietIndoors': ('04', 'QuietIndoors'),
            'ReverberantEnvironment': ('05', 'ReverberantEnvironment'),
            'WindTurbulence': ('06', 'WindTurbulence')
        }

    def extract_components(self, filepath):
        """Extract components from original filename."""
        filename = filepath.name
        parts = filename.split('_')
        
        # Handle different filename patterns
        if 'rec' in parts[0]:  # Original HEAR-DS format
            # For format like: rec_id_001_cut_00_description_Mic_ITC_L_raw_48kHz32bit.wav
            try:
                rec_id = [p for p in parts if p.isdigit()][0]  # Get first number sequence
                cut_parts = [i for i, p in enumerate(parts) if 'cut' in p]
                cut_id = parts[cut_parts[0] + 1] if cut_parts else '00'
                
                # Find ITC and side (L/R)
                itc_parts = [i for i, p in enumerate(parts) if 'ITC' in p]
                if itc_parts:
                    itc_idx = itc_parts[0]
                    if itc_idx + 1 < len(parts):
                        side = parts[itc_idx + 1]  # Should be L or R
                        track_name = f"ITC_{side}"
                    else:
                        track_name = "ITC_L"  # Default if no side specified
                else:
                    return None, None, None
                
            except (IndexError, ValueError):
                return None, None, None
                
        else:  # InterfereringSpeakers format
            # For format like: S02_000_ITC_L_16kHz.wav
            rec_id = parts[0].replace('S', '')  # Remove 'S' from S02
            cut_id = parts[1]
            track_name = f"ITC_{parts[3]}"  # ITC_L or ITC_R
            
        return rec_id, cut_id, track_name


    def process_file(self, input_file, env_id, output_dir):
        """Process a single audio file and split into 10-second chunks."""
        try:
            # Extract components
            rec_id, cut_id, track_name = self.extract_components(input_file)
            
            # Skip if not ITC or components couldn't be extracted
            if not rec_id or not track_name or 'ITC' not in track_name:
                return 0
            
            # Load audio
            audio, sr = librosa.load(input_file, sr=16000, mono=True)
            
            # Calculate number of complete 10-second chunks
            chunk_samples = self.chunk_duration * 16000
            num_chunks = int(len(audio) / chunk_samples)
            
            chunks_processed = 0
            for i in range(num_chunks):
                # Extract chunk
                start_idx = i * chunk_samples
                end_idx = start_idx + chunk_samples
                chunk = audio[start_idx:end_idx]
                
                # Create filename for chunk
                new_filename = f"{env_id}_{rec_id}_{cut_id}_{i:03d}_{track_name}_16kHz.wav"
                output_path = output_dir / "Background" / new_filename
                
                # Save chunk
                sf.write(output_path, chunk, 16000)
                chunks_processed += 1
            
            return chunks_processed
            
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            return 0

    def process_environment(self, env_name):
        """Process all files in an environment directory."""
        env_path = self.input_dir / env_name
        if not env_path.exists():
            print(f"Environment directory not found: {env_path}")
            return
        
        env_id, folder_name = self.env_ids[env_name]
        
        # Create environment-specific output directory
        env_output_dir = self.output_dir / folder_name
        env_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Background subdirectory
        background_dir = env_output_dir / "Background"
        background_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all wav files recursively
        wav_files = list(env_path.rglob("*ITC*.wav"))  # Only get ITC files
        print(f"Processing {len(wav_files)} ITC files in {env_name}")
        
        total_chunks = 0
        for wav_file in tqdm(wav_files, desc=f"Processing {env_name}"):
            chunks = self.process_file(wav_file, env_id, env_output_dir)
            total_chunks += chunks
        
        print(f"Successfully processed {len(wav_files)} files into {total_chunks} 10-second chunks in {env_name}")
        return total_chunks

    def process_all(self):
        """Process all environments."""
        total_chunks = 0
        for env_name in self.env_ids.keys():
            print(f"\nProcessing environment: {env_name}")
            chunks = self.process_environment(env_name)
            total_chunks += chunks
            
        print(f"\nTotal 10-second chunks created: {total_chunks}")
        
        # Print summary of files in each environment
        print("\nChunks per environment:")
        for env_name, (_, folder_name) in self.env_ids.items():
            env_dir = self.output_dir / folder_name / "Background"
            if env_dir.exists():
                file_count = len(list(env_dir.glob('*.wav')))
                print(f"{env_name}: {file_count} chunks")
            return

def main():
    input_dir = Path("/Volumes/SSD/Datasets/HEAR-DS")
    # output_dir = Path("/Volumes/SSD/Datasets/HEAR-DS/Down-Sampled")
    output_dir = Path("/Users/nkdem/Downloads/HEAR-DS/Down-Sampled")
    
    processor = HEARDownsampler(input_dir, output_dir)
    processor.process_all()

if __name__ == "__main__":
    main()
