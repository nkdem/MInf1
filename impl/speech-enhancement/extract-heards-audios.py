# for each model in hear-ds-speech-enh-exp-N
# for each environment
# read the corresponding model.pth file 
# load the splits for the N-th 
# load the waveforms in a cache
# test it again beofre STOI, before PESQ, after PESQ
# find top 5 improvements (in terms of STOI, PESQ, audio quality)
# save the waveforms in a cache
# repeat for all environments
# repeat for the nth model

import os
import pickle
import soundfile as sf
import librosa
import numpy as np
import torch
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from models import CRNN
def set_snr(loader, snr):
    dataset = loader.dataset
    if hasattr(dataset, 'snr'):
        dataset.snr = snr
    else:
        method = getattr(dataset, 'set_snr', None)
        if method is not None and callable(method):
            method(snr)
        else:
            dataset = loader.dataset.dataset
            if hasattr(dataset, 'snr'):
                dataset.snr = snr
            else:
                method = getattr(dataset, 'set_snr', None)
                if method is not None and callable(method):
                    method(snr)
                else:
                    raise ValueError("Dataset does not have a set_snr method")


class AudioExtractor:
    def __init__(self, experiment_no, cuda=True):
        self.experiment_no = experiment_no
        self.base_dir = f'experiments/hear-ds-speech-enh-exp-{experiment_no}'
        self.cuda = cuda
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.snr_levels = [-10, -5, 0, 5, 10]
        self.environments = [
            'InTraffic',
            'InVehicle',
            'QuietIndoors',
            'ReverberantEnvironment',
            'WindTurbulence',
            'Music',
        ]
        self._setup_data_from_splits()

    def _setup_data_from_splits(self):
        """Set up data loaders using pre-generated splits"""
        split_file = f'splits/split_{self.experiment_no - 1}.pkl'  # 0-indexed
        with open(split_file, 'rb') as f:
            split = pickle.load(f)
            self.test_mixed_ds = split['random_snr']['subsets']['test']['mixed']

    def _get_environment_specific_loader(self, test_mixed_ds, environment):
        """Create environment-specific data loader."""
        test_indices = []
        for i in range(len(test_mixed_ds)):
            _, _, env, _, _, _, _, _ = test_mixed_ds[i]
            if env == f'SpeechIn_{environment}':
                test_indices.append(i)

        test_env_ds = torch.utils.data.Subset(test_mixed_ds, test_indices)
        
        def speech_enh_collate_fn(batch):
            noisy_list = []
            clean_list = []
            envs = []
            recs = []
            cut_ids = []
            snip_ids = []
            extras = []
            snrs = []

            for (noisy, clean, env, recsit, cut_id, snip_id, extra, snr) in batch:
                noisy_list.append(noisy[0])  # Using left channel
                clean_list.append(clean[0])
                envs.append(env)
                recs.append(recsit)
                cut_ids.append(cut_id)
                snip_ids.append(snip_id)
                extras.append(extra)
                snrs.append(snr)

            return noisy_list, clean_list, envs, recs, cut_ids, snip_ids, extras, snrs

        test_loader = torch.utils.data.DataLoader(
            test_env_ds,
            batch_size=4,  # Process one at a time for better control
            shuffle=False,
            collate_fn=speech_enh_collate_fn
        )
        return test_loader

    def process_environment(self, environment):
        """Process a single environment and find top improvements"""
        print(f"\nProcessing environment: {environment}")
        
        # Load model
        model_path = f"{self.base_dir}/model_{environment}.pth"
        if not os.path.exists(model_path):
            print(f"Model not found for {environment}")
            return
        
        model = CRNN().to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Get test loader for this environment
        test_loader = self._get_environment_specific_loader(self.test_mixed_ds, environment)
        
        # Initialize results storage
        results = {
            'stoi_improvements': [],
            'pesq_improvements': [],
            'samples': []
        }

        # Process each sample
        for snr in self.snr_levels:
            print(f"Processing SNR: {snr}")
            set_snr(test_loader, snr)
            for batch in tqdm(test_loader, desc="Processing samples"):
                noisy_batch, clean_batch, env, recsit, cut_id, snip_id, extra, snrs = batch
                
                # Convert to numpy arrays
                noisy = np.array(noisy_batch, dtype=np.float32)
                clean = np.array(clean_batch, dtype=np.float32)

                # Calculate metrics before enhancement

                # Enhance the audio
                noisy_mag, noisy_phase = librosa.magphase(librosa.stft(noisy, n_fft=320, win_length=320, hop_length=160))
                noisy_mag = torch.tensor(noisy_mag, device=self.device, dtype=torch.float32).permute(0, 2, 1)
                enhanced_mag = model(noisy_mag)
                enhanced_mag = enhanced_mag.permute(0, 2, 1).detach().cpu().numpy()
                enhanced = librosa.istft(enhanced_mag * noisy_phase, hop_length=160, win_length=320, length=160000)

                for i in range(len(noisy_batch)):       
                    before_pesq = pesq(16000, clean_batch[i], noisy_batch[i], 'wb')
                    before_stoi = stoi(clean_batch[i], noisy_batch[i], 16000, extended=True)
                    after_pesq = pesq(16000, clean_batch[i], enhanced[i], 'wb')
                    after_stoi = stoi(clean_batch[i], enhanced[i], 16000, extended=True)

                # Calculate improvements
                stoi_improvement = after_stoi - before_stoi
                pesq_improvement = after_pesq - before_pesq

                # Store results
                sample_info = [{
                    'env': env[i],
                    'rec': recsit[i],
                    'cut_id': cut_id[i],
                    'snip_id': snip_id[i],
                    'snr': snr,
                    'before_stoi': before_stoi,
                    'after_stoi': after_stoi,
                    'before_pesq': before_pesq,
                    'after_pesq': after_pesq,
                    'noisy': noisy[i],
                    'clean': clean[i],
                    'enhanced': enhanced[i]
                } for i in range(len(noisy_batch))]

                for sample in sample_info:
                    results['stoi_improvements'].append((stoi_improvement, sample))
                    results['pesq_improvements'].append((pesq_improvement, sample))
                    results['samples'].append(sample)

        # Sort and get top 5 improvements
        results['stoi_improvements'].sort(reverse=True, key=lambda x: x[0])
        results['pesq_improvements'].sort(reverse=True, key=lambda x: x[0])

        # Save top 5 samples for each metric
        self._save_top_samples(results, environment)

    def _save_top_samples(self, results, environment):
        """Save the top 5 samples for each metric"""
        output_dir = f"{self.base_dir}/top_samples_{environment}"
        os.makedirs(output_dir, exist_ok=True)

        # Save top 5 STOI improvements
        for i, (improvement, sample) in enumerate(results['stoi_improvements'][:5]):
            # difference
            difference = sample['after_stoi'] - sample['before_stoi']
            self._save_sample(sample, f"{output_dir}/stoi_top_{i+1}_[diff={difference:.3f}]", improvement)

        # Save top 5 PESQ improvements
        for i, (improvement, sample) in enumerate(results['pesq_improvements'][:5]):
            # difference
            difference = sample['after_pesq'] - sample['before_pesq']
            self._save_sample(sample, f"{output_dir}/pesq_top_{i+1}_[diff={difference:.3f}]", improvement)

    def _save_sample(self, sample, prefix, improvement):
        """Save a single sample with its metrics"""
        # Save audio files
        sf.write(f"{prefix}_noisy.wav", sample['noisy'], 16000)
        sf.write(f"{prefix}_clean.wav", sample['clean'], 16000)
        sf.write(f"{prefix}_enhanced.wav", sample['enhanced'], 16000)

        # Save metrics
        with open(f"{prefix}_metrics.txt", 'w') as f:
            f.write(f"Environment: {sample['env']}\n")
            f.write(f"Recording: {sample['rec']}\n")
            f.write(f"Cut ID: {sample['cut_id']}\n")
            f.write(f"Snippet ID: {sample['snip_id']}\n")
            f.write(f"SNR: {sample['snr']}\n")
            f.write(f"STOI: {sample['before_stoi']:.3f} -> {sample['after_stoi']:.3f} (improvement: {improvement:.3f})\n")
            f.write(f"PESQ: {sample['before_pesq']:.3f} -> {sample['after_pesq']:.3f}\n")

    def run(self):
        """Run the extraction process for all environments"""
        for environment in self.environments:
            self.process_environment(environment)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_no", type=int, required=True, default=1)
    parser.add_argument("--cuda", action='store_true', default=True)
    args = parser.parse_args()

    extractor = AudioExtractor(args.experiment_no, args.cuda)
    extractor.run()


