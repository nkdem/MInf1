import os
import wave
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_empty_wav(filepath):
    """Create an empty WAV file with minimal header information."""
    try:
        # Open original file to get parameters
        with wave.open(filepath, 'rb') as orig_wav:
            channels = orig_wav.getnchannels()
            sampwidth = orig_wav.getsampwidth()
            framerate = orig_wav.getframerate()
        
        # Create new empty WAV file with same parameters
        with wave.open(filepath, 'wb') as new_wav:
            new_wav.setnchannels(channels)
            new_wav.setsampwidth(sampwidth)
            new_wav.setframerate(framerate)
            # Write minimal data (1 frame of silence)
            new_wav.writeframes(b'\x00' * channels * sampwidth)
            
        logger.info(f"Successfully created empty WAV: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {filepath}: {str(e)}")
        return False

def compress_with_zstd(filepath):
    """Compress a file using zstd."""
    try:
        # Compress with maximum compression level (-19) and remove original file
        subprocess.run(['zstd', '-19', '--rm', filepath], check=True)
        logger.info(f"Successfully compressed: {filepath}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error compressing {filepath}: {str(e)}")
        return False
    except FileNotFoundError:
        logger.error("zstd command not found. Please install zstd first.")
        return False

def process_directory(root_dir):
    """Process all WAV files in the directory structure."""
    count = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.wav'):
                filepath = os.path.join(root, file)
                original_size = os.path.getsize(filepath)
                
                # Create empty WAV
                if create_empty_wav(filepath):
                    intermediate_size = os.path.getsize(filepath)
                    
                    # Compress with zstd
                    if compress_with_zstd(filepath):
                        final_size = os.path.getsize(filepath + '.zst')
                        count += 1
                        logger.info(f"File {filepath}: "
                                  f"Original: {original_size/1024:.2f}KB → "
                                  f"Empty WAV: {intermediate_size/1024:.2f}KB → "
                                  f"Compressed: {final_size/1024:.2f}KB")
    
    logger.info(f"Processed {count} WAV files")

def check_zstd_installed():
    """Check if zstd is installed."""
    try:
        subprocess.run(['zstd', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

if __name__ == "__main__":
    dataset_path = "abc"  # Update this path to your copied dataset location
    
    # Check if zstd is installed
    if not check_zstd_installed():
        logger.error("zstd is not installed. Please install it first:")
        logger.error("  macOS: brew install zstd")
        logger.error("  Ubuntu/Debian: sudo apt install zstd")
        exit(1)
    
    # Confirm with user before proceeding
    response = input(f"This will modify all WAV files in {dataset_path} "
                    f"and compress them with zstd. Are you sure you want to proceed? (y/n): ")
    
    if response.lower() == 'y':
        logger.info("Starting processing...")
        process_directory(dataset_path)
        logger.info("Processing complete!")
    else:
        logger.info("Operation cancelled")
