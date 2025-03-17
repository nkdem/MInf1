import collections
import csv
import logging
import os
import torch
import numpy as np
import pickle
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import soundfile as sf
import sys 
from pesq import pesq
from pystoi import stoi
from train_enhance import SpeechEnhanceAdamEarlyStopTrainer
sys.path.append(os.path.abspath(os.path.join('.')))
from heards_dataset import BackgroundDataset, MixedAudioDataset, SpeechDataset, split_background_dataset, split_speech_dataset
from helpers import compute_average_logmel, linear_to_waveform, logmel_to_linear
from models import CNNSpeechEnhancer
from expr_baseline import BaseExperiment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerEnvironmentSpeechEnhancementExperiment(BaseExperiment):
    """
    A variant of the speech enhancement experiment that trains separate models for each environment.
    """
    def __init__(
        self,
        batch_size: int = 16,
        cuda: bool = False,
        experiment_no: int = 1,
        use_splits: bool = True,
    ):
        self.heards_dir = '/Users/nkdem/Downloads/HEAR-DS' if not cuda else '/disk/scratch/s2203859/minf-1/HEAR-DS'
        self.speech_dir = '/Volumes/SSD/Datasets/CHiME3/CHiME3-Isolated-DEV/dt05_bth' if not cuda else '/disk/scratch/s2203859/minf-1/dt05_bth/'
        self.experiment_no = experiment_no
        self.use_splits = use_splits
        
        # Initialize with empty loaders first
        super().__init__(batch_size=batch_size, cuda=cuda)
        
        if use_splits:
            self.setup_data_from_splits()
        else:
            self.setup_data_from_scratch(self.heards_dir, self.speech_dir)
            
        logger.info("PerEnvironmentSpeechEnhancementExperiment initialized.")

    def _get_environment_specific_loaders(self, train_mixed_ds, test_mixed_ds, environment):
        """Create environment-specific data loaders."""
        
        def get_indices_for_environment(dataset, target_env):
            indices = []
            for i in range(len(dataset)):
                _, _, env, _, _, _, _ = dataset[i]
                if env == f"SpeechIn_{target_env}":
                    indices.append(i)
            return indices

        # Get indices for the specific environment
        train_indices = get_indices_for_environment(train_mixed_ds, environment)
        test_indices = get_indices_for_environment(test_mixed_ds, environment)

        # Create subsets for the specific environment
        train_env_ds = Subset(train_mixed_ds, train_indices)
        test_env_ds = Subset(test_mixed_ds, test_indices)

        def speech_enh_collate_fn(batch, ignore=True):
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

        # Create data loaders for the environment-specific subsets
        train_loader = DataLoader(
            train_env_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=speech_enh_collate_fn
        )
        test_loader = DataLoader(
            test_env_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=speech_enh_collate_fn
        )

        return train_loader, test_loader

    def run(self):
        """
        Run the experiment by training separate models for each environment.
        """
        logger.info("Running per-environment speech enhancement experiment...")

        # List of environments (excluding CocktailParty and InterfereringSpeakers)
        environments = [
            'InTraffic',
            'InVehicle',
            'Music',
            'QuietIndoors',
            'ReverberantEnvironment',
            'WindTurbulence'
        ]

        # Load the split data
        split_file = f'splits/split_{self.experiment_no - 1}.pkl'
        with open(split_file, 'rb') as f:
            split = pickle.load(f)
            train_mixed_ds = split['subsets']['train']['mixed']
            test_mixed_ds = split['subsets']['test']['mixed']

        # Train and evaluate a model for each environment
        for env in environments:
            logger.info(f"\nTraining model for environment: {env}")
            
            # Create environment-specific directory
            base_dir = self.create_experiment_dir(f"speech_enhance_{env}", self.experiment_no)
            
            # Get environment-specific data loaders
            env_train_loader, env_test_loader = self._get_environment_specific_loaders(
                train_mixed_ds, test_mixed_ds, env
            )
            
            # Initialize trainer for this environment
            adam = SpeechEnhanceAdamEarlyStopTrainer(
                base_dir=base_dir,
                num_epochs=240,  # You might want to adjust this
                train_loader=env_train_loader,
                batch_size=self.batch_size,
                cuda=self.cuda,
                initial_lr=1e-3,
                early_stop_threshold=1e-4,
                patience=5,
            )

            # Train the model
            adam.train()
            
            logger.info(f"Training completed for {env}. Starting evaluation...")

            # Load the trained model
            model_path = os.path.join(base_dir, 'model.pth')
            cnn = CNNSpeechEnhancer().to(self.device)
            cnn.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))

            # Test the model
            self.test(base_dir=base_dir, test_loader=env_test_loader, model=cnn, trainer=adam)
            
            logger.info(f"Evaluation completed for {env}")

        logger.info("Per-environment experiment completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_no", type=int)
    parser.add_argument("--cuda", action='store_true', default=False)
    parser.add_argument("--no_splits", action='store_true', default=False, 
                        help="If set, don't use pre-generated splits")
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
    exp = PerEnvironmentSpeechEnhancementExperiment(
        batch_size=16,
        cuda=cuda,
        experiment_no=experiment_no,
        use_splits=use_splits
    )
    exp.run()
    logger.info("Done.") 