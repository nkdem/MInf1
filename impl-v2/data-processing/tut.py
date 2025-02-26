import os
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm
import numpy as np

class AudioProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_sample_rate = 16000  # Target sample rate for downsampling

    def process_file(self, input_file):
        """Process a single audio file and downsample it to 16kHz."""
        try:
            # Load audio
            audio, sr = librosa.load(input_file, sr=None, mono=True)
            
            # Downsample if necessary
            if sr != self.target_sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sample_rate)
            

            # Create output filename the same as the input filename
            basename = os.path.basename(input_file)
            output_path = self.output_dir / basename
            
            # Save downsampled audio
            sf.write(output_path, audio, self.target_sample_rate)
            
            return 1  # Return 1 to indicate successful processing
            
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            return 0

    def process_directory(self):
        """Process all audio files in the input directory and its subdirectories."""
        # Get all audio files recursively
        audio_files = list(self.input_dir.rglob("*audio/*.wav"))  # Only get audio files in 'audio' folders
        
        print(f"Processing {len(audio_files)} audio files")
        
        total_files_processed = 0
        for audio_file in tqdm(audio_files, desc="Processing audio files"):
            files_processed = self.process_file(audio_file)
            total_files_processed += files_processed
        
        print(f"Successfully processed {total_files_processed} files")

def main():
    input_dir = Path("/Users/nkdem/Downloads/TUT-acoustic-scenes-2017-development")
    output_dir = Path("/Users/nkdem/Downloads/TUT-acoustic-scenes-2017-development-16k")
    # input_dir = Path("/home/s2203859/TUT-acoustic-scenes-2017-development")
    # output_dir = Path("/home/s2203859/TUT-acoustic-scenes-2017-development-16k")
    
    processor = AudioProcessor(input_dir, output_dir)
    processor.process_directory()

if __name__ == "__main__":
    main()
