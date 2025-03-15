# There's a folder called models 
# and each has a subfolder for each experiment.
# two experiments we currently have are adam-early-stop and fixed-lr-sgd and they're suffixed with the experiment number.
# so the path to the experiment is models/adam-early-stop-1 and models/fixed-lr-sgd-1 etc.
# each of those folders has a results.pkl file  which is 
#             {
#             'class_accuracies': {model: [] for model in MODELS.keys()},
#             'total_accuracies': {model: [] for model in MODELS.keys()},
#             'confusion_matrix_raw': {model: [] for model in MODELS.keys()},
#             'trained_models': {model: [] for model in MODELS.keys()}, 
#             'losses': {model: [] for model in MODELS.keys()},
#             'duration': {model: [] for model in MODELS.keys()},
#             'learning_rates': {model: [] for model in MODELS.keys()}
#             }

import os 
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from constants import MODELS
output_dir = 'output'

def get_results():
    results = {} # key is experiment name without the number, value is a list of results (index 0 is experiment 1, index 1 is experiment 2 etc)
    for folder in os.listdir('models'):
        if os.path.isdir(os.path.join('models', folder)):
            base_folder = folder.split('-')[0]
            if base_folder not in results:
                results[base_folder] = []
            for experiment in os.listdir(f'models/{folder}'):
                results[base_folder].append({})
                with open(f'models/{folder}/{experiment}/results.pkl', 'rb') as f:
                    results[base_folder][-1] = pickle.load(f)
    return results

def plot_model_accuracies(output_dir, results):
    plt.figure(figsize=(12, 6))
    models = list(MODELS.keys())
    mean_accs = []
    std_accs = []
    
    # Calculate mean and std for each model across experiments
    for model in models:
        model_accs = [exp['total_accuracies'][model] for exp in results if model in exp['total_accuracies']]
        mean_acc = np.mean(model_accs)
        std_acc = np.std(model_accs)
        mean_accs.append(mean_acc)
        std_accs.append(std_acc)
    
    # Plotting
    plt.plot(models, mean_accs, marker='o')
    plt.errorbar(models, mean_accs, yerr=std_accs, fmt='none', capsize=5, ecolor='black')
    
    plt.title(f'Model Accuracies with Standard Deviation')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_accuracies_with_std.png'))
    plt.close()

results = get_results()
for experiment, experiment_results in results.items():
    os.makedirs(os.path.join(output_dir, experiment), exist_ok=True)
    plot_model_accuracies(output_dir, experiment_results)
