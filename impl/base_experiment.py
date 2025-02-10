# base_experiment.py
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from pesq import pesq
# from pystoi import stoi
import seaborn as sns
from abc import ABC, abstractmethod

import torchaudio
from constants import MODELS
from train import train
from test import test


class BaseExperiment(ABC):
    def __init__(self, dataset, cuda=False):
        self.dataset = dataset
        self.cuda = cuda
        
    def create_experiment_dir(self, experiment_name, i):
        base_dir = f'models/{experiment_name}/{i}'
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def initialize_result_containers(self):
        return {
            'confusion_matrices': {model: [] for model in MODELS.keys()},
            'accuracies': {model: [] for model in MODELS.keys()},
            'losses': {model: [] for model in MODELS.keys()},
            'training_times': {model: [] for model in MODELS.keys()}
        }

    def validate_data_consistency(self, training_data, testing_data):
        for j in range(len(MODELS) - 1):
            assert training_data[j] == training_data[j + 1], 'Training data is not the same across models'
            assert testing_data[j] == testing_data[j + 1], 'Testing data is not the same across models'

    def collect_experiment_data(self, base_dir):
        training_data = [[] for _ in range(len(MODELS))]
        testing_data = [[] for _ in range(len(MODELS))]

        for j, (model, _) in enumerate(MODELS.items()):
            root_dir = os.path.join(base_dir, model)
            if os.path.exists(root_dir):
                train_data = os.path.join(root_dir, 'train_files.txt')
                test_data = os.path.join(root_dir, 'test_files.txt')
                if os.path.exists(train_data) and os.path.exists(test_data):
                    with open(train_data, 'r') as f:
                        training_data[j] = f.readlines()
                    with open(test_data, 'r') as f:
                        testing_data[j] = f.readlines()
        
        return training_data, testing_data

    def collect_model_results(self, base_dir, model, results):
        root_dir = os.path.join(base_dir, model)
        if os.path.exists(root_dir):
            # Get model architecture parameters
            cnn1_channels, cnn2_channels, fc_neurons = MODELS[model]
            
            # Get confusion matrix and accuracy
            confusion_matrix, accuracy = test(
                base_dir=root_dir,
                root_dir=root_dir,
                dataset=self.dataset, 
                model_name=model, 
                cnn1_channels=cnn1_channels,
                cnn2_channels=cnn2_channels,
                fc_neurons=fc_neurons,
                cuda=self.cuda
            )
            results['confusion_matrices'][model].append(confusion_matrix)
            results['accuracies'][model].append(accuracy)
            
            # Get losses
            loss_file = os.path.join(root_dir, 'losses.txt')
            if os.path.exists(loss_file):
                with open(loss_file, 'r') as f:
                    losses = [float(line.strip()) for line in f.readlines()]
                    results['losses'][model].append(losses)
            
            # Get training time
            duration_file = os.path.join(root_dir, 'duration.txt')
            if os.path.exists(duration_file):
                with open(duration_file, 'r') as f:
                    duration = float(f.readline().strip())
                    results['training_times'][model].append(duration / 60)  # Convert to minutes

    def plot_std_confusion_matrix(self, model, all_confusion_matrices, base_dir, output_dir, experiment_name):
        if all_confusion_matrices:
            std_conf_matrix = np.std(all_confusion_matrices, axis=0)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(std_conf_matrix, 
                       annot=False,
                       cmap='YlOrBr_r',
                       vmin=0.0,
                       vmax=1.0,
                       square=False,
                       linewidths=0.5,
                       linecolor='black',
                       cbar_kws={
                           'label': '',
                           'orientation': 'horizontal',
                           'pad': 0.2,
                           'ticks': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                       })

            # Read int_to_label mapping
            with open(os.path.join(base_dir, model, 'int_to_label.txt'), 'r') as f:
                int_to_label = dict(line.strip().split() for line in f)
                int_to_label = {int(k): v for k, v in int_to_label.items()}

            num_classes = len(int_to_label)

            plt.xticks(np.arange(num_classes) + 0.5, 
                      [int_to_label[i] for i in range(num_classes)], 
                      rotation=45,
                      ha='right',
                      rotation_mode='anchor')

            plt.yticks(np.arange(num_classes) + 0.5, 
                      [int_to_label[i] for i in range(num_classes)], 
                      rotation=0)
            
            plt.ylabel('True Scene')
            plt.title(f"Standard Deviation of Confusion Matrix - {model} [{experiment_name}]")

            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)

            ax = plt.gca()
            ax.set_xlabel('Estimated Scene')
            colorbar = ax.collections[0].colorbar
            colorbar.ax.set_position([
                colorbar.ax.get_position().x0, 
                colorbar.ax.get_position().y0 - 0.05,
                colorbar.ax.get_position().width,
                colorbar.ax.get_position().height
            ])

            plt.savefig(
                os.path.join(output_dir, f'{model}_std_confusion_matrix.png'),
                bbox_inches='tight',
                dpi=300
            )
            plt.close()

    def plot_classwise_accuracy(self, model, all_confusion_matrices, base_dir, output_dir, experiment_name):
        if all_confusion_matrices:
            class_accuracies_per_exp = []
            for conf_matrix in all_confusion_matrices:
                acc = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
                class_accuracies_per_exp.append(acc)
            
            class_accuracies_array = np.array(class_accuracies_per_exp)
            mean_accuracies = np.mean(class_accuracies_array, axis=0)
            std_accuracies = np.std(class_accuracies_array, axis=0)
            
            plt.figure(figsize=(12, 6))
            x = range(len(mean_accuracies))
            plt.bar(x, mean_accuracies, yerr=std_accuracies, capsize=5)
            
            with open(os.path.join(base_dir, model, 'int_to_label.txt'), 'r') as f:
                int_to_label = dict(line.strip().split() for line in f)
                int_to_label = {int(k): v for k, v in int_to_label.items()}
            
            plt.xticks(x, [int_to_label[i] for i in range(len(mean_accuracies))], 
                      rotation=45, ha='right')
            plt.title(f'Classwise Accuracy - {model} [{experiment_name}]')
            plt.xlabel('Class')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model}_classwise_accuracy.png'))
            plt.close()

    def plot_model_accuracies(self, all_accuracies, output_dir, experiment_name):
        plt.figure(figsize=(12, 6))
        for model in MODELS.keys():
            if all_accuracies[model]:
                mean_acc = np.mean(all_accuracies[model])
                std_acc = np.std(all_accuracies[model])
                plt.errorbar(model, mean_acc, yerr=std_acc, fmt='o', capsize=5)

        plt.title(f'Model Accuracies with Standard Deviation [{experiment_name}]')
        plt.xlabel('Model')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_accuracies.png'))
        plt.close()

    def plot_training_losses(self, all_losses, output_dir, experiment_name):
        plt.figure(figsize=(12, 6))
        for model in MODELS.keys():
            if all_losses[model]:
                # pad losses with zeros to ensure equal length
                # Find maximum length
                max_len = max(len(losses) for losses in all_losses[model])
                # Pad each loss list with NaNs (or another value)
                padded_losses = [losses + [np.nan]*(max_len - len(losses)) for losses in all_losses[model]]
                losses_array = np.array(padded_losses)
                mean_losses = np.mean(losses_array, axis=0)
                std_losses = np.std(losses_array, axis=0)
                epochs = range(1, len(mean_losses) + 1)
                
                plt.plot(epochs, mean_losses, label=model)
                plt.fill_between(
                    epochs, 
                    mean_losses - std_losses,
                    mean_losses + std_losses,
                    alpha=0.2
                )

        plt.title(f'Train Loss against Epochs [{experiment_name}]')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_losses.png'))
        plt.close()

    def plot_training_times(self, all_training_times, output_dir, experiment_name):
        plt.figure(figsize=(10, 6))
        for model in MODELS.keys():
            if all_training_times[model]:
                mean_time = np.mean(all_training_times[model])
                std_time = np.std(all_training_times[model])
                plt.bar(model, mean_time, yerr=std_time, capsize=5)

        plt.title(f'Training Time Analysis [{experiment_name}]')
        plt.xlabel('Model')
        plt.ylabel('Time (minutes)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_times.png'))
        plt.close()

    def generate_plots(self, results, base_dir, output_dir, experiment_name):
        for model in MODELS.keys():
            self.plot_std_confusion_matrix(
                model, 
                results['confusion_matrices'][model],
                base_dir,
                output_dir,
                experiment_name
            )
            self.plot_classwise_accuracy(
                model,
                results['confusion_matrices'][model],
                base_dir,
                output_dir,
                experiment_name
            )
        
        self.plot_model_accuracies(results['accuracies'], output_dir, experiment_name)
        self.plot_training_losses(results['losses'], output_dir, experiment_name)
        self.plot_training_times(results['training_times'], output_dir, experiment_name)

    def compare_snr_performance(self):
        """
        For each environment that contains a Speech folder, this function computes
        (and prints) average PESQ and STOI at each SNR level.
        
        Assumptions:
        - Each environment’s Speech folder contains subfolders named after SNR levels (e.g. "21", "-21", etc.).
        - Inside each SNR folder are the noisy .wav files.
        - For each noisy .wav file, the corresponding JSON mapping key is reconstructed as:
                "<noisy_filename>_<snr>"
            For example, if the file is "06_204_13_015_ITC_R_16kHz.wav" in the folder "21",
            then the mapping key would be "06_204_13_015_ITC_R_16kHz.wav_21".
        - The JSON mapping file (located at the dataset root as "speech_mapping.json") contains entries like:
                "06_204_13_015_ITC_R_16kHz.wav_-21": {
                    "concatenated_speech": "WindTurbulence_06_204_13_015_ITC_-21_concat.wav",
                    "used_files": [...]
                }
        - The clean (concatenated) speech files are located in the "concatenated_speech" folder at the dataset root.
        
        The printed table for each environment will have the following structure:
        
        Environment: WindTurbulence
        SNR    PESQ     STOI
        -21  1.25     0.45
        ...
        21  2.80     0.92
        """
        dataset_dir = self.dataset.root_dir

        # Load the speech mapping JSON file from the dataset root.
        mapping_file = os.path.join(dataset_dir, "speech_mapping.json")
        if not os.path.exists(mapping_file):
            print("Error: speech_mapping.json not found in dataset root directory.")
            return
        with open(mapping_file, "r") as f:
            speech_mapping = json.load(f)
            
        # Identify environments that contain a Speech folder.
        environments = []
        for env in os.listdir(dataset_dir):
            env_path = os.path.join(dataset_dir, env)
            if os.path.isdir(env_path) and os.path.exists(os.path.join(env_path, "Speech")):
                environments.append(env)
                    
        # Define the SNR levels we wish to evaluate.
        snr_levels = [-21, -18, -15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18, 21]
        
        # Store the results in a nested dictionary: { environment: {snr: {pesq: val, stoi: val} } }
        table_data = {}
        
        for env in environments:
            table_data[env] = {}
            speech_dir = os.path.join(dataset_dir, env, "Speech")
            # The clean concatenated files are expected in the "concatenated_speech" folder at the dataset root.
            clean_dir = os.path.join(dataset_dir, "concatenated_speech")
            
            for snr in snr_levels:
                pesq_vals = []
                stoi_vals = []
                # Build the path for the SNR folder.
                snr_folder = os.path.join(speech_dir, str(snr))
                if not os.path.exists(snr_folder):
                    table_data[env][snr] = {"pesq": None, "stoi": None}
                    continue

                for file in os.listdir(snr_folder):
                    # Process only .wav (or .wav.zst if needed) files.
                    if not (file.endswith('.wav') or file.endswith('.wav.zst')):
                        continue

                    # Construct the mapping key as: "<file>_<snr>"
                    key = f"{file}_{snr}"
                    if key not in speech_mapping:
                        continue
                    concat_file = speech_mapping[key].get("concatenated_speech")
                    if not concat_file:
                        continue

                    noisy_path = os.path.join(snr_folder, file)
                    clean_path = os.path.join(clean_dir, concat_file)
                    if os.path.exists(noisy_path) and os.path.exists(clean_path):
                        try:
                            noisy, sr = torchaudio.load(noisy_path)
                            clean, sr_clean = torchaudio.load(clean_path)

                            # Ensure that the sampling rates match
                            if sr != sr_clean:
                                continue

                            # Squeeze to remove extra dimensions (assuming they are mono signals)
                            noisy_array = noisy.numpy().squeeze()
                            clean_array = clean.numpy().squeeze()

                            # Crop signals to the same length
                            min_length = min(len(clean_array), len(noisy_array))
                            clean_array = clean_array[:min_length]
                            noisy_array = noisy_array[:min_length]

                            # Compute PESQ (using wideband mode) and STOI
                            p_val = pesq(sr, clean_array, noisy_array, mode='wb')
                            s_val = stoi(clean_array, noisy_array, sr, extended=False)
                            pesq_vals.append(p_val)
                            stoi_vals.append(s_val)
                        except Exception as e:
                            print(f"Error processing file {file} in environment {env} at SNR {snr}: {e}")
                # Compute average values if any were collected.
                avg_pesq = np.mean(pesq_vals) if pesq_vals else None
                avg_stoi = np.mean(stoi_vals) if stoi_vals else None
                table_data[env][snr] = {"pesq": avg_pesq, "stoi": avg_stoi}
        
        # Print out the summary table for each environment.
        for env in environments:
            print(f"\nEnvironment: {env}")
            print(f"{'SNR':>5} | {'PESQ':>5} | {'STOI':>5}")
            print("-" * 25)
            for snr in snr_levels:
                pesq_val = table_data[env][snr]["pesq"]
                stoi_val = table_data[env][snr]["stoi"]
                pesq_str = f"{pesq_val:.2f}" if pesq_val is not None else "NA"
                stoi_str = f"{stoi_val:.2f}" if stoi_val is not None else "NA"
                print(f"{snr:5} | {pesq_str:>5} | {stoi_str:>5}")
        
        # Optionally, format the results in a DataFrame for easier handling.
        df_dict = {}
        for env in environments:
            df_env = pd.DataFrame({
                "PESQ": [table_data[env][snr]["pesq"] for snr in snr_levels],
                "STOI": [table_data[env][snr]["stoi"] for snr in snr_levels]
            }, index=snr_levels)
            df_dict[env] = df_env
        for env, df in df_dict.items():
            print(f"\nDetailed results for {env}:")
            print(df)
        
        return table_data



    @abstractmethod
    def run(self):
        """
        This method should be implemented by each specific experiment class.
        It should contain the main experiment logic.
        """
        pass
