import os
import time
import torch
import numpy as np
import pickle
from tqdm import tqdm
import soundfile as sf
import pandas as pd
from pesq import pesq
from pystoi import stoi
import sys
from itertools import product
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join('.')))
from heards_dataset import BackgroundDataset, MixedAudioDataset, SpeechDataset, split_background_dataset, split_speech_dataset
from models import CNNSpeechEnhancer
from helpers import compute_average_logmel, linear_to_waveform, logmel_to_linear
from train_enhance import SpeechEnhanceAdamEarlyStopTrainer

class GriffinLimExperiment:
    def __init__(self, model_path, experiment_no, batch_size=32, cuda=False):
        self.model_path = model_path
        self.experiment_no = experiment_no
        self.batch_size = batch_size
        self.cuda = cuda
        self.device = torch.device("mps" if not self.cuda else "cuda")
        
        # Load the split data
        split_file = f'splits/split_{self.experiment_no - 1}.pkl'
        with open(split_file, 'rb') as f:
            split = pickle.load(f)
            self.test_mixed_ds = split['subsets']['test']['mixed']
            
        # Create data loader
        self.test_loader = torch.utils.data.DataLoader(
            self.test_mixed_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: self.speech_enh_collate_fn(x, ignore=False)
        )
        
        # Load the model
        self.model = CNNSpeechEnhancer().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True, map_location=self.device))
        self.model.eval()
        
        # Initialize trainer for normalization
        self.trainer = SpeechEnhanceAdamEarlyStopTrainer(
            base_dir=os.path.dirname(self.model_path),
            num_epochs=1,  # We don't need training
            train_loader=self.test_loader,
            batch_size=self.batch_size,
            cuda=self.cuda,
            initial_lr=1e-3,
            early_stop_threshold=1e-4,
            patience=5,
        )
        
        # Define parameter combinations to test
        self.param_combinations = list(product(
            # [128, 64, 32],  # n_iter
            # [0.99, 0.1],  # momentum
            # [2,1]  # power
            [32],
            [0.99],
            [1]
        ))
        
    def speech_enh_collate_fn(self, batch, ignore=True):
        noisy_list = []
        clean_list = []
        envs = []
        recs = []
        cut_ids = []
        extras = []
        snrs = []

        for (noisy, clean, env, recsit, cut_id, extra, snr) in batch:
            noisy_list.append(noisy)
            clean_list.append(clean)
            envs.append(env)
            recs.append(recsit)
            cut_ids.append(cut_id)
            extras.append(extra)
            snrs.append(snr)

        return noisy_list, clean_list, envs, recs, cut_ids, extras, snrs
    
    def test_parameters(self):
        results = []
        env_results = defaultdict(list)
        
        for n_iter, momentum, power in tqdm(self.param_combinations, desc="Testing parameter combinations"):
            start_time = time.time()
            param_results = {
                'n_iter': n_iter,
                'momentum': momentum,
                'power': power,
                'before_pesq': [],
                'before_stoi': [],
                'after_pesq': [],
                'after_stoi': [],
                'time_taken': None
            }
            
            # Initialize environment-specific results for this parameter combination
            env_param_results = defaultdict(lambda: {
                'before_pesq': [],
                'before_stoi': [],
                'after_pesq': [],
                'after_stoi': []
            })
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(self.test_loader, desc=f"Testing batch for n_iter={n_iter}, momentum={momentum}, power={power}", leave=False)):
                    noisy, clean, envs, recs, cut_ids, extras, snrs = batch
                    if batch_idx >= 1:
                        break
                    if len(noisy) == 0:
                        continue
                        
                    noisy_computed_logmel = compute_average_logmel(noisy, self.device)
                    normalised_noisy_logmels = self.trainer.normalize_logmels(noisy_computed_logmel)
                    
                    # Reshape the input to match the expected 4D format
                    batch_size, channels, time_frames = normalised_noisy_logmels.shape
                    reshaped_input = normalised_noisy_logmels.view(batch_size, channels, 1, time_frames)
                    
                    enhanced = self.model(reshaped_input)
                    denormalised = self.trainer.denormalize_logmels(enhanced.squeeze(2), is_clean=True)
                    
                    # Convert to linear spectrogram
                    sample_rate = 16000
                    inverted = logmel_to_linear(
                        logmel_spectrogram=denormalised,
                        sample_rate=sample_rate,
                        n_fft=1024,
                        n_mels=40,
                        device=self.device
                    )
                    
                    # Convert to waveform with current parameters
                    hop_length = int(0.02 * sample_rate)
                    win_length = int(0.04 * sample_rate)
                    waveform = linear_to_waveform(
                        linear_spectrogram_batch=inverted,
                        sample_rate=sample_rate,
                        n_fft=1024,
                        hop_length=hop_length,
                        win_length=win_length,
                        device=self.device,
                        num_iters=n_iter,
                        momentum=momentum,
                        power=power
                    )
                    
                    # Calculate metrics
                    for i, (clean_audio, snr, env) in enumerate(zip(clean, snrs, envs)):
                        if clean_audio is None:
                            continue
                            
                        noisy_audio = np.array(noisy[i][0])
                        clean_audio = np.array(clean_audio[0])
                        enhanced_audio = waveform[i].cpu().numpy()
                        
                        before_pesq = pesq(sample_rate, clean_audio, noisy_audio, 'wb')
                        after_pesq = pesq(sample_rate, clean_audio, enhanced_audio, 'wb')
                        before_stoi = stoi(clean_audio, noisy_audio, sample_rate, extended=False)
                        after_stoi = stoi(clean_audio, enhanced_audio, sample_rate, extended=False)
                        
                        # Store overall results
                        param_results['before_pesq'].append(before_pesq)
                        param_results['before_stoi'].append(before_stoi)
                        param_results['after_pesq'].append(after_pesq)
                        param_results['after_stoi'].append(after_stoi)
                        
                        # Store environment-specific results
                        env_param_results[env]['before_pesq'].append(before_pesq)
                        env_param_results[env]['before_stoi'].append(before_stoi)
                        env_param_results[env]['after_pesq'].append(after_pesq)
                        env_param_results[env]['after_stoi'].append(after_stoi)
            
            # Calculate averages for this parameter combination
            param_results['time_taken'] = time.time() - start_time
            results.append({
                'n_iter': n_iter,
                'momentum': momentum,
                'power': power,
                'before_pesq': np.mean(param_results['before_pesq']),
                'before_stoi': np.mean(param_results['before_stoi']),
                'after_pesq': np.mean(param_results['after_pesq']),
                'after_stoi': np.mean(param_results['after_stoi']),
                'time_taken': param_results['time_taken']
            })
            
            # Store environment-specific results
            for env, env_data in env_param_results.items():
                env_results[env].append({
                    'n_iter': n_iter,
                    'momentum': momentum,
                    'power': power,
                    'before_pesq': np.mean(env_data['before_pesq']),
                    'before_stoi': np.mean(env_data['before_stoi']),
                    'after_pesq': np.mean(env_data['after_pesq']),
                    'after_stoi': np.mean(env_data['after_stoi'])
                })
            
            # Save intermediate results
            self.save_results(results, env_results)
            
        return results, env_results
    
    def save_results(self, results, env_results):
        # Create results directory
        results_dir = os.path.join(os.path.dirname(self.model_path), 'griffin_lim_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save overall results
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(results_dir, 'results.csv'), index=False)
        df.to_json(os.path.join(results_dir, 'results.json'), orient='records', indent=4)
        
        # Save environment-specific results
        for env, env_data in env_results.items():
            env_df = pd.DataFrame(env_data)
            env_df.to_csv(os.path.join(results_dir, f'{env}_results.csv'), index=False)
            env_df.to_json(os.path.join(results_dir, f'{env}_results.json'), orient='records', indent=4)
        
        # Print summary
        print("\nOverall Results Summary:")
        print("Parameter Combination\tBefore Enhancement\t\tAfter Enhancement")
        print("\t\tPESQ\tSTOI\t\tPESQ\tSTOI")
        for result in results:
            print(f"n_iter={result['n_iter']}, momentum={result['momentum']}, power={result['power']} time taken: {result['time_taken']}")
            print(f"\t\t{result['before_pesq']:.2f}\t{result['before_stoi']:.2f}\t\t{result['after_pesq']:.2f}\t{result['after_stoi']:.2f}")
        
        print("\nEnvironment-Specific Results:")
        for env, env_data in env_results.items():
            print(f"\n{env}:")
            print("Parameter Combination\tBefore Enhancement\t\tAfter Enhancement")
            print("\t\tPESQ\tSTOI\t\tPESQ\tSTOI")
            for result in env_data:
                print(f"n_iter={result['n_iter']}, momentum={result['momentum']}, power={result['power']}")
                print(f"\t\t{result['before_pesq']:.2f}\t{result['before_stoi']:.2f}\t\t{result['after_pesq']:.2f}\t{result['after_stoi']:.2f}")

def main():
    # Example usage
    model_path = "models/speech_enhance/1/model.pth"
    experiment = GriffinLimExperiment(
        model_path=model_path,
        experiment_no=1,  # Update this number
        batch_size=32,
        cuda=False
    )
    
    results, env_results = experiment.test_parameters()
    print("\nExperiment completed!")

if __name__ == "__main__":
    main() 