import collections
import logging
import os
import pickle
import sys
import torch
from torch.utils.data import DataLoader
import torchaudio
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join('.')))

from helpers import compute_spectrogram
from models import CNNSpeechEnhancer2
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CachedFeaturesDataset(Dataset):
    def __init__(self, feature_cache):
        self.noisy_features = []
        self.clean_features = []
        
        # Group features by environment to ensure we handle speech and non-speech correctly
        grouped_features = {}
        for key, (noisy, clean) in feature_cache.items():
            # key format is: "env_recsit_cut_snippet_snr"
            parts = key.split('_')
            if parts[0] == 'SpeechIn':  
                env = f"SpeechIn_{parts[1]}"
            else:
                env = parts[0]
            recsit = parts[2]
            cut = parts[3]
            snippet = parts[4]
            snr = parts[5]
            
            if env not in grouped_features:
                grouped_features[env] = []
            # For non-speech environments, SNR might be None
            snr_val = int(snr) if snr != 'None' else None
            grouped_features[env].append((snr_val, noisy, clean))
        
        # For each environment, add features appropriately
        for env, snr_features in grouped_features.items():
            # Sort by SNR to ensure consistent ordering
            # Filter out None SNRs just in case
            valid_snr_features = [(s, n, c) for s, n, c in snr_features if s is not None]
            valid_snr_features.sort(key=lambda x: x[0])
            for _, noisy, clean in valid_snr_features:
                self.noisy_features.append(noisy)
                self.clean_features.append(clean)
    def __len__(self):
        return len(self.noisy_features)
    def __getitem__(self, idx):
        return self.noisy_features[idx], self.clean_features[idx]

class SpeechEnhancementExperiment():
    def __init__(self, batch_size=1, cuda=False, experiment_no=1, use_splits=True, augment=True):
        self.batch_size = batch_size
        self.cuda = cuda
        self.device = torch.device("mps" if not self.cuda else "cuda")
        self.experiment_no = experiment_no
        self.use_splits = use_splits
        self.augment = augment
        self.feature_cache = {}
        self.num_epochs = 10
        self.patience = 5
        self.setup_data_from_splits()
        # self.precompute_spectrograms()

    def set_snr(self, loader, snr):
        dataset = loader.dataset
        if hasattr(dataset, 'snr'):
            dataset.snr = snr
        else:
            method = getattr(dataset, 'set_snr', None)
            if method is not None and callable(method):
                method(snr)

    def set_load_waveforms(self, loader, load_waveforms):
        dataset = loader.dataset
        if hasattr(dataset, 'load_waveforms'):
            dataset.load_waveforms = load_waveforms
        else:
            method = getattr(dataset, 'set_load_waveforms', None)
            if method is not None and callable(method):
                method(load_waveforms)

    def create_experiment_dir(self, experiment_name, i):
        base_dir = f'models/{experiment_name}/{i}'
        os.makedirs(base_dir, exist_ok=True)
        return base_dir
    def initialize_result_containers(self):
        return collections.OrderedDict(
            {
            'losses': [],
            'duration': 0,
            'learning_rates': [],
            }
        )
    def setup_data_from_splits(self):
        """Set up data loaders using pre-generated splits"""
        split_file = f'splits/split_{self.experiment_no - 1}.pkl'  # 0-indexed
        
        # open split file
        with open(split_file, 'rb') as f:
            split = pickle.load(f)
            
            # Extract the mixed datasets from the splits
            train_mixed_ds = split['random_snr']['subsets']['train']['mixed']
            test_mixed_ds = split['random_snr']['subsets']['test']['mixed']
            
            def speech_enh_collate_fn(batch, ignore = True):
                noisy_list = []
                clean_list = []
                envs = []
                recs = []
                cut_ids = []
                snip_ids = []
                extras = []
                snrs = []

                for (noisy, clean, env, recsit, cut_id, snip_id, extra, snr) in batch:
                    if ignore and env in ["CocktailParty", "InterfereringSpeakers"]:
                        # Skip these examples during training since we don't have clean audio for them
                        continue
                    noisy_list.append(noisy)
                    clean_list.append(clean)
                    envs.append(env)
                    recs.append(recsit)
                    cut_ids.append(cut_id)
                    snip_ids.append(snip_id)
                    extras.append(extra)
                    snrs.append(snr)

                return noisy_list, clean_list, envs, recs, cut_ids, snip_ids, extras, snrs
            
            self.train_loader = DataLoader(
                train_mixed_ds, batch_size=self.batch_size, shuffle=True, collate_fn=speech_enh_collate_fn
            )
            self.test_loader = DataLoader(
                test_mixed_ds, batch_size=self.batch_size, shuffle=False, collate_fn=lambda x: speech_enh_collate_fn(x, ignore=True)
            )
        
    def test(self):
        # read model
        model = CNNSpeechEnhancer2().to(self.device)
        model.load_state_dict(torch.load(os.path.join('models/speech_enhancement_2', 'model.pth')))
        model.eval()
        istft = torchaudio.transforms.InverseSpectrogram(n_fft=1024, hop_length=256, win_length=1024, normalized=True).to(self.device)

        self.set_snr(self.test_loader, 0)

        with torch.no_grad():
            for batch in self.test_loader:
                noisy_list, clean_list, envs, recs, cut_ids, snip_ids, extras, snrs = batch
                noisy_spec = compute_spectrogram(noisy_list, device=self.device, normalize=True)
                clean_spec = compute_spectrogram(clean_list, device=self.device, normalize=True)

                enhanced = model(noisy_spec.abs())
                enhanced = enhanced.to(torch.complex64)

                # lets take the imaginary part of the noisy spectrogram and add it to the enhanced spectrogram
                noisy_spec = noisy_spec.to(torch.complex64)
                noisy_spec = torch.imag(noisy_spec)

                # lets verify there's at least one non-zero value in the noisy_spec
                if torch.sum(noisy_spec) == 0:
                    continue

                enhanced = enhanced + noisy_spec

                enhanced = enhanced.to(torch.complex64)
                enhanced = istft(enhanced)

                # convert to complex

                # save to wav
                for i, (noisy, clean, env, rec, cut_id, snip_id, extra, snr) in enumerate(zip(noisy_list, clean_list, envs, recs, cut_ids, snip_ids, extras, snrs)):
                    os.makedirs(f"models/speech_enhancement_2/test_wavs/{env}", exist_ok=True)
                    wav_path = f"models/speech_enhancement_2/test_wavs/{env}_{rec}_{cut_id}_{snip_id}_{snr}_enhanced.wav"
                    torchaudio.save(wav_path, enhanced[i].to('cpu'), 16000)
                    # lets save the original noisy and clean wavs
                    noisy_wav_path = f"models/speech_enhancement_2/test_wavs/{env}_{rec}_{cut_id}_{snip_id}_{snr}_noisy.wav"
                    # the noisy and clean are arrays with length 2 and we need to convert them to tensors
                    noisy_tensor = torch.tensor(noisy, dtype=torch.float32)
                    clean_tensor = torch.tensor(clean, dtype=torch.float32)
                    # save as stereo
                    torchaudio.save(noisy_wav_path, noisy_tensor, 16000)
                    clean_wav_path = f"models/speech_enhancement_2/test_wavs/{env}_{rec}_{cut_id}_{snip_id}_{snr}_clean.wav"
                    torchaudio.save(clean_wav_path, clean_tensor, 16000)








    def run(self):
        criterion = nn.MSELoss()

        model = CNNSpeechEnhancer2().to(self.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_loss = float('inf')
        epochs_no_improve = 0
        losses = []

        self.set_snr(0)
        for epoch in range(self.num_epochs):
            running_loss = 0.0

            counter = 0
            pbar = tqdm(
                self.train_loader,
                desc=f"[Epoch {epoch + 1}/{self.num_epochs}] [LR: {optimiser.param_groups[0]['lr']}]",
                unit="batch"
            )
            count = 0
            for batch in pbar: 
                count += 1
                if count > 50:
                    break
                noisy_list, clean_list, envs, recs, cut_ids, snip_ids, extras, snrs = batch
                # compute spectrograms
                noisy_spec = compute_spectrogram(noisy_list, device=self.device, normalize=True)
                clean_spec = compute_spectrogram(clean_list, device=self.device, normalize=True)

                # convert to magnitude only
                noisy_spec = noisy_spec.abs()
                clean_spec = clean_spec.abs()

                noisy_spec = noisy_spec.to(self.device)
                clean_spec = clean_spec.to(self.device)

                optimiser.zero_grad()
                output = model(noisy_spec)
                loss = criterion(output, clean_spec)
                loss.backward()
                optimiser.step()

                running_loss += loss.item()
                
                avg_loss = running_loss / len(self.train_loader)
                losses.append(avg_loss)
                pbar.set_postfix({
                    'loss': f'{running_loss:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                })


            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        logger.info("Training phase completed. Starting results collection and analysis...")

        # save losses
        with open(os.path.join('models/speech_enhancement_2', 'losses.pkl'), 'wb') as f:
            pickle.dump(losses, f)
        # save model
        torch.save(model.state_dict(), os.path.join('models/speech_enhancement_2', 'model.pth'))

        pass



                
        
        

                

if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_no", type=int)
    parser.add_argument("--cuda", action='store_true', default=False)
    parser.add_argument("--no_splits", action='store_true', default=False, 
                        help="If set, don't use pre-generated splits")
    parser.add_argument("--no_augment", action='store_true', default=False,
                        help="If set, don't use data augmentation and use cached features")
    args = parser.parse_args()

    # if arg is not provided, default to 1
    # but warn 
    if args.experiment_no is None:
        print("No experiment number provided. Defaulting to 1.")
        experiment_no = 1
    else:
        experiment_no = args.experiment_no
    cuda = args.cuda
    use_splits = not args.no_splits
    augment = not args.no_augment
    exp = SpeechEnhancementExperiment(batch_size=4, cuda=cuda, experiment_no=experiment_no, use_splits=use_splits, augment=augment)
    # exp.run()
    exp.test()
    logger.info("Done.")