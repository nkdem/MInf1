import os
import pickle
import soundfile as sf
import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
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

def set_load_waveforms(loader, load_waveforms):
    dataset = loader.dataset
    if hasattr(dataset, 'load_waveforms'):
        dataset.load_waveforms = load_waveforms
    else:
        method = getattr(dataset, 'set_load_waveforms', None)
        if method is not None and callable(method):
            method(load_waveforms)
        else:
            dataset = loader.dataset.dataset
            if hasattr(dataset, 'load_waveforms'):
                dataset.load_waveforms = load_waveforms
            else:
                method = getattr(dataset, 'set_load_waveforms', None)
                if method is not None and callable(method):
                    method(load_waveforms)
                else:
                    raise ValueError("Dataset does not have a set_load_waveforms method")
            


class SpeechEnhancementExperiment:
    def __init__(self, experiment_no, cuda):
        self.experiment_no = experiment_no
        self.base_dir = f'experiments/hear-ds-speech-enh-exp-one-model-{experiment_no}'
        self.cuda = cuda
        self.device = torch.device('cuda' if cuda else 'cpu')
        self.batch_size = 16
        self.train_cached_grouped_by_env = None
        self.test_cached_grouped_by_env = None
        self.snr_levels = [-10,-5,0,5,10]
        # self.snr_levels = [0]
        self._setup_data_from_splits()
        self.patience = 5
        self.patience_counter = 0
        self.lr = 0.0006
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epochs = 120

    def _setup_data_from_splits(self):
        """Set up data loaders using pre-generated splits"""
        split_file = f'splits/split_{self.experiment_no - 1}.pkl'  # 0-indexed
        
        # open split file
        with open(split_file, 'rb') as f:
            split = pickle.load(f)
            
            # Extract the mixed datasets from the splits
            self.train_mixed_ds = split['random_snr']['subsets']['train']['mixed']
            self.test_mixed_ds = split['random_snr']['subsets']['test']['mixed']
            


    def get_loaders(self):
        def speech_enh_collate_fn(batch, ignore=True):
            noisy_list = []
            clean_list = []
            envs = []
            recs = []
            cut_ids = []
            snip_ids = []
            extras = []
            snrs = []

            for (noisy, clean, env, recsit, cut_id, snip_id, extra, snr) in batch:
                if env in ["CocktailParty", "InterfereringSpeakers"]:
                    # skip these environments for now
                    continue
                noisy_list.append(noisy)
                clean_list.append(clean)
                envs.append(env)
                recs.append(recsit)
                cut_ids.append(cut_id)
                snip_ids.append(snip_id)
                extras.append(extra)
                snrs.append(snr)
            
            # TODO: FOr now lets just use left channel so monoaural speech enhancemenmt
            noisy_list = [noisy[0] for noisy in noisy_list] if noisy_list[0] is not None else []
            clean_list = [clean[0] for clean in clean_list] if clean_list[0] is not None else []

            return noisy_list, clean_list, envs, recs, cut_ids, snip_ids, extras, snrs

        # Create data loaders for the environment-specific subsets
        train_loader = DataLoader(
            self.train_mixed_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=speech_enh_collate_fn
        )
        test_loader = DataLoader(
            self.test_mixed_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=lambda x: speech_enh_collate_fn(x, ignore=True)
        )

        return train_loader, test_loader

    
    def train(self, train_loader):
        model = CRNN().to(self.device)
        losses = []
        best_loss = float('inf')

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        criterion = nn.MSELoss()

        waveform_cache = {}

        set_snr(train_loader, 0) # load only snr 0
        set_load_waveforms(train_loader, True) # load waveforms

        # precompute waveforms
        for snr in self.snr_levels:
            set_snr(train_loader, snr)
            for batch in tqdm(train_loader, desc=f"Precomputing waveforms [snr={snr}]"):
                noisy_batch, clean_batch, env, recsit, cut_id, snip_id, extra, snr = batch
                for i in range(len(noisy_batch)):
                    key = (env[i], recsit[i], cut_id[i], snip_id[i], snr[i])
                    waveform_cache[key] = (noisy_batch[i], clean_batch[i])
        set_load_waveforms(train_loader, False)
        set_snr(train_loader, None) # random snr

        train_loader.dataset.snr_levels = self.snr_levels

        for epoch in range(self.epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", total=len(train_loader))
            avg_loss = 0
            for i, batch in enumerate(pbar):
                noisy_batch, clean_batch, env, recsit, cut_id, snip_id, extra, snr = batch
                noisy_list = []
                clean_list = []
                for i in range(len(env)):
                    key = (env[i], recsit[i], cut_id[i], snip_id[i], snr[i])
                    noisy, clean = waveform_cache[key]
                    noisy_list.append(noisy)
                    clean_list.append(clean)
                
                # convert to numpy
                noisy_list = np.array(noisy_list, dtype=np.float32)
                clean_list = np.array(clean_list, dtype=np.float32)

                noisy_mag = librosa.magphase(librosa.stft(noisy_list, n_fft=320, win_length=320, hop_length=160))[0]# [B,F,T]
                clean_mag = librosa.magphase(librosa.stft(clean_list, n_fft=320, win_length=320, hop_length=160))[0] # [B,F,T]
                # transpose to [B,T,F]
                noisy_mag = torch.tensor(noisy_mag, device=self.device, dtype=torch.float32).permute(0, 2, 1)   
                clean_mag = torch.tensor(clean_mag, device=self.device, dtype=torch.float32).permute(0, 2, 1)

                optimizer.zero_grad()
                output = model(noisy_mag)
                loss = criterion(output, clean_mag)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                pbar.set_postfix({
                    'avg_loss': avg_loss / (i + 1)
                })
            losses.append(avg_loss / (len(train_loader)))
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        return model, losses


    def test(self, model, test_loader):
        model.eval()
        set_snr(test_loader, 0) # load only snr 0
        set_load_waveforms(test_loader, True) # load waveforms
        with torch.no_grad():
            before_pesq = {snr: {} for snr in self.snr_levels}
            after_pesq = {snr: {} for snr in self.snr_levels}
            before_stoi = {snr: {} for snr in self.snr_levels}
            after_stoi = {snr: {} for snr in self.snr_levels}
            for snr in self.snr_levels:
                set_snr(test_loader, snr)
                pbar = tqdm(test_loader, desc="Testing", unit="batch")
                for batch in pbar:
                    noisy_batch, clean_batch, env, recsit, cut_id, snip_id, extra, snrs = batch
                    noisy_list = []
                    clean_list = []
                    for i in range(len(noisy_batch)):
                        key = (env[i], recsit[i], cut_id[i], snip_id[i])
                        noisy_list.append(noisy_batch[i])
                        clean_list.append(clean_batch[i])
                    # convert to numpy
                    noisy_list = np.array(noisy_list, dtype=np.float32)
                    clean_list = np.array(clean_list, dtype=np.float32)

                    noisy_mag,noisy_phase = librosa.magphase(librosa.stft(noisy_list, n_fft=320, win_length=320, hop_length=160))
                    clean_mag = librosa.magphase(librosa.stft(clean_list, n_fft=320, win_length=320, hop_length=160))[0]
                    # transpose to [B,T,F]
                    noisy_mag = torch.tensor(noisy_mag, device=self.device, dtype=torch.float32).permute(0, 2, 1)   
                    clean_mag = torch.tensor(clean_mag, device=self.device, dtype=torch.float32).permute(0, 2, 1)

                    enhanced_mag = model(noisy_mag)
                    # if batch dimension is not present, add it
                    if enhanced_mag.ndim == 2:
                        enhanced_mag = enhanced_mag[None, :, :]
                    enhanced_mag = enhanced_mag.permute(0, 2, 1).detach().cpu().numpy() # [B, T, F] => [B, F, T]

                    enhanced = librosa.istft(enhanced_mag * noisy_phase, hop_length=160, win_length=320, length=160000)
                    
                    # iterate through each sample
                    for i in range(len(noisy_batch)):
                        if env[i] not in before_pesq[snr]:
                            before_pesq[snr][env[i]] = []
                            before_stoi[snr][env[i]] = []
                            after_pesq[snr][env[i]] = []
                            after_stoi[snr][env[i]] = []
                        before_pesq[snr][env[i]].append(pesq(16000, clean_list[i], noisy_list[i]))
                        before_stoi[snr][env[i]].append(stoi(clean_list[i], noisy_list[i], 16000, extended=True))
                        after_pesq[snr][env[i]].append(pesq(16000, clean_list[i], enhanced[i]))
                        after_stoi[snr][env[i]].append(stoi(clean_list[i], enhanced[i], 16000, extended=True))

                    # save waveforms
                    # sf.write(f"noisy_{key}.wav", noisy_list[i], 16000)
                    # sf.write(f"clean_{key}.wav", clean_list[i], 16000)
                    # sf.write(f"enhanced_{key}.wav", enhanced[i], 16000)

                    
                    mean_pesq = {env: [] for env in before_pesq[snr]}
                    mean_stoi = {env: [] for env in before_stoi[snr]}
                    for env in before_pesq[snr]:
                        mean_pesq[env].append(np.mean(before_pesq[snr][env]))
                        mean_stoi[env].append(np.mean(before_stoi[snr][env]))
                    for env in after_pesq[snr]:
                        mean_pesq[env].append(np.mean(after_pesq[snr][env]))
                        mean_stoi[env].append(np.mean(after_stoi[snr][env]))
                    # total mean pesq and stoi (average over all environments)
                    total_before_pesq = {env: [] for env in before_pesq[snr]}
                    total_before_stoi = {env: [] for env in before_stoi[snr]}
                    total_after_pesq = {env: [] for env in after_pesq[snr]}
                    total_after_stoi = {env: [] for env in after_stoi[snr]}
                    for env in before_pesq[snr]:
                        total_before_pesq[env].append(np.mean(mean_pesq[env]))
                        total_before_stoi[env].append(np.mean(mean_stoi[env]))
                    for env in after_pesq[snr]:
                        total_after_pesq[env].append(np.mean(mean_pesq[env]))
                        total_after_stoi[env].append(np.mean(mean_stoi[env]))
                    
                    pbar.set_postfix(
                        {
                            'avg_pesq': f'{np.mean(list(total_before_pesq.values())):.2f} -> {np.mean(list(total_after_pesq.values())):.2f}',
                            'avg_stoi': f'{np.mean(list(total_before_stoi.values())):.2f} -> {np.mean(list(total_after_stoi.values())):.2f}'
                        }
                    )
        return before_pesq, after_pesq, before_stoi, after_stoi
    def run(self):
        os.makedirs(self.base_dir, exist_ok=True)
        train_loader, test_loader = self.get_loaders()
        model, losses = self.train(train_loader)
        torch.save(model.state_dict(), f"{self.base_dir}/model.pth")
        with open(f"{self.base_dir}/losses.csv", "w") as f:
            f.write(f"Epoch,Loss\n")
            for i, loss in enumerate(losses):
                    f.write(f"{i},{loss}\n")
        

            before_pesq, after_pesq, before_stoi, after_stoi = self.test(model, test_loader)
        #     for snr in self.snr_levels:
        #         print(f"SNR: {snr}")
        #         print(f"PESQ: {np.mean(list(before_pesq[snr])):.2f} -> {np.mean(list(after_pesq[snr])):.2f}")
        #         print(f"STOI: {np.mean(list(before_stoi[snr])):.2f} -> {np.mean(list(after_stoi[snr])):.2f}")
        #     print("-"*100)
        #     # save results
            with open(f"{self.base_dir}/results.pkl", "wb") as f:
                pickle.dump({
                    "before_pesq": before_pesq,
                    "after_pesq": after_pesq,
                    "before_stoi": before_stoi,
                    "after_stoi": after_stoi
                }, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_no", type=int)
    parser.add_argument("--cuda", action='store_true', default=True)
    # parser.add_argument("--no_augment", action='store_true', default=False,
                        # help="If set, don't use data augmentation and use cached features")
    args = parser.parse_args()

    # if arg is not provided, default to 1
    # but warn 
    if args.experiment_no is None:
        print("No experiment number provided. Defaulting to 1.")
        experiment_no = 5
    else:
        experiment_no = args.experiment_no
    cuda = args.cuda

    experiment = SpeechEnhancementExperiment(experiment_no, cuda)
    experiment.run()
