import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List
# from constants import MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from pesq import pesq
from pystoi import stoi

def load_experiment_results_HEARDS(base_folder: str = 'models', max_experiments: int = 5) -> Dict:
    """
    Load results from all experiments for different model configurations.
    Returns a dictionary with experiment types as keys and lists of results as values.
    """
    results = {}
    for folder in os.listdir(base_folder):
        if not os.path.isdir(os.path.join(base_folder, folder)):
            continue
            
        for exp_type in ['fixed-lr-sgd-AUG-', 'adam-early-stop-', 'FIXED-fixed-lr-sgd-']:
            if folder.startswith(exp_type):
                base_name = exp_type.rstrip('-')
                if base_name not in results:
                    results[base_name] = []
                
                experiment_number = int(folder.split('-')[-1])
                if experiment_number <= max_experiments:
                    with open(f'{base_folder}/{folder}/{experiment_number}/results2.pkl', 'rb') as f:
                        results[base_name].append(pickle.load(f))
    
    return results

def load_experiment_results_TUT(base_folder: str = 'models/full_adam_240epochs_32batchsize_TUT') -> Dict:
    """
    Load results from all experiments for different model configurations.
    Returns a dictionary with experiment types as keys and lists of results as values.
    """
    results = {'TUT-experiment': []}
    for exp_type in ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']:
        for fold in ['fold1', 'fold2', 'fold3', 'fold4']:
            with open(os.path.join(base_folder, exp_type, f'results-{fold}.pkl'), 'rb') as f:
                # as the pickle files were created before refactoring the keys need to be renamed
                # namely confusion_matrix_raw -> naive_confusion_matrix_raw
                # class_accuracies -> naive_class_accuracies
                # total_accuracies -> naive_total_accuracies
                
                obj = pickle.load(f)
                obj['naive_confusion_matrix_raw'] = obj['confusion_matrix_raw']
                obj['naive_class_accuracies'] = obj['class_accuracies']
                obj['naive_total_accuracies'] = obj['total_accuracies']
                results['TUT-experiment'].append(obj)

    return results

def load_experiment_results_SE_HEARDS(base_folder: str = 'experiments/') -> Dict:
    """
    Load results from SE experiments for different model configurations.
    Returns a dictionary with experiment types as keys and lists of results as values.
    Each result contains losses and metrics for each environment.
    """
    environments = [
        'Music', 'ReverberantEnvironment', 'QuietIndoors',
        'InVehicle', 'WindTurbulence', 'InTraffic'
    ]
    results = {env: [] for env in environments}
    
    # Look for SE experiment folders
    for folder in os.listdir(base_folder):
        if not os.path.isdir(os.path.join(base_folder, folder)):
            continue
            
        # Check if this is an SE experiment folder
        if folder.startswith('hear-ds-speech-enh-exp-'):
            # Load results for each environment
            for env in environments:
                obj = {}
                # losses_InTraffic.csv etc
                losses_file = os.path.join(base_folder, folder, f'losses_{env}.csv')
                with open(losses_file, 'rb') as f:
                    # csv file read
                    losses = pd.read_csv(f)
                    # epoch, loss
                    # lets convert to list of losses
                    obj['losses'] = losses['Loss'].tolist()
                
                # results_InTraffic.pkl
                results_file = os.path.join(base_folder, folder, f'results_{env}.pkl')
                try:
                    with open(results_file, 'rb') as f:
                        obj['results'] = pickle.load(f)
                except Exception as e:
                    print(f"Error loading results for {env}: {e}")
                results[env].append(obj)
    
    return results

def load_class_labels(base_folder: str, exp_type: str, model_name: str) -> Dict[int, str]:
    """
    Load class labels from int_to_label.txt file for a specific model.
    Args:
        base_folder: Base directory containing experiment folders
        exp_type: Type of experiment (e.g., 'fixed-lr-sgd' or 'adam-early-stop')
        model_name: Name of the model (e.g., 'net-8', 'net-20', etc.)
    """
    # Look through experiment folders to find the label file
    for folder in os.listdir(base_folder):
        if folder.startswith(f"{exp_type}-"):
            exp_num = folder.split('-')[-1]
            # Look in the experiment directory, not the model directory
            label_file = os.path.join(base_folder, folder, exp_num, 'int_to_label.txt')
            if os.path.exists(label_file):
                labels = {}
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            label, idx = line.strip().split(' ')
                            labels[int(idx)] = label
                return labels
    return {}  # Return empty dict if no labels found

def plot_training_losses(results: Dict, output_dir: str):
    """Plot training losses for all models across experiments."""
    # Define a color palette for different models
    colors = sns.color_palette("husl", len(MODELS))
    
    for exp_type, exp_results in results.items():
        plt.figure(figsize=(6, 4))  # Reduced from (12, 6)
        
        for model_idx, model_name in enumerate(MODELS.keys()):
            # Collect losses from all experiments for this model
            all_losses = []
            for exp in exp_results:
                if model_name in exp['losses']:
                    losses = exp['losses'][model_name]
                    # Handle the case where losses is a list of lists
                    if isinstance(losses, list):
                        if losses and isinstance(losses[0][0], list):
                            all_losses.extend(losses[0])  # Extend if we have a list of lists
                        else:
                            all_losses.append(losses[0])  # Append if we have a single list
            
            if not all_losses:  # Skip if no data for this model
                continue
                
            # Find the maximum length
            # flatten the list if needed
            max_len = max(len(loss) for loss in all_losses)
            
            # Pad shorter sequences with NaN
            padded_losses = []
            for loss in all_losses:
                if len(loss) < max_len:
                    padded = np.pad(loss, (0, max_len - len(loss)), 
                                  mode='constant', constant_values=np.nan)
                else:
                    padded = loss
                padded_losses.append(padded)
            
            # Stack arrays and compute statistics
            losses_array = np.stack(padded_losses)
            mean_losses = np.nanmean(losses_array, axis=0)
            std_losses = np.nanstd(losses_array, axis=0)
            
            # Plot mean with confidence interval
            epochs = range(1, max_len + 1)
            plt.plot(epochs, mean_losses, label=model_name, color=colors[model_idx])
            plt.fill_between(epochs, 
                           mean_losses - std_losses,
                           mean_losses + std_losses,
                           alpha=0.2, color=colors[model_idx])
    
            # y ticks 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25
            plt.yticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25])
            plt.xlim(0, 121)
            plt.title(f'Training Loss Over Time ({exp_type})', fontsize=10)
            plt.xlabel('Epoch', fontsize=9)
            plt.ylabel('Loss', fontsize=9)
            plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # Save training losses plot in the main output directory since it compares all models
        plt.savefig(os.path.join(output_dir, f'{exp_type}_training_losses.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()

def plot_classwise_accuracies(results: Dict, output_dir: str):
    """Plot classwise accuracies for each model and experiment."""
    # Reference data from the paper
    paper_accuracies = {
        'Music': 82,
        'CocktailParty': 57,
        'ReverberantEnvironment': 40,
        'QuietIndoors': 75,
        'InVehicle': 90,
        'WindTurbulence': 63,
        'InTraffic': 85,
        'InterfereringSpeakers': 77,
        'SpeechIn_Music': 80,
        'SpeechIn_ReverberantEnvironment': 40,
        'SpeechIn_QuietIndoors': 84,
        'SpeechIn_InVehicle': 74,
        'SpeechIn_WindTurbulence': 63,
        'SpeechIn_InTraffic': 80, 
    }
    
    paper_spreads = {
        'Music': 10,
        'CocktailParty': 30,
        'ReverberantEnvironment': 23,
        'QuietIndoors': 25,
        'InVehicle': 3,
        'WindTurbulence': 17,
        'InTraffic': 3,
        'InterfereringSpeakers': 10,
        'SpeechIn_Music': 3,
        'SpeechIn_ReverberantEnvironment': 25,
        'SpeechIn_QuietIndoors': 5,
        'SpeechIn_InVehicle': 17,
        'SpeechIn_WindTurbulence': 20,
        'SpeechIn_InTraffic': 4,
    }

    for exp_type, exp_results in results.items():
        for model_name in MODELS.keys():
            # Create model-specific and experiment-type directory
            exp_dir = os.path.join(output_dir, model_name, exp_type)
            os.makedirs(exp_dir, exist_ok=True)
            
            # Load class labels
            class_labels = load_class_labels('models', exp_type, model_name)
            
            # Collect classwise accuracies for all experiments
            all_accuracies = []
            for exp in exp_results:
                if model_name in exp['naive_class_accuracies']:
                    acc = exp['naive_class_accuracies'][model_name]
                    if isinstance(acc, (list, np.ndarray)):
                        if isinstance(acc[0], (list, np.ndarray)):
                            acc_list = [np.array(a) * 100 if np.max(a) <= 1 else np.array(a) for a in acc]
                            all_accuracies.extend(acc_list)
                        else:
                            acc_array = np.array(acc) * 100 if np.max(acc) <= 1 else np.array(acc)
                            all_accuracies.append(acc_array)
            
            if not all_accuracies:  # Skip if no data
                continue
                
            # Convert to numpy array
            all_accuracies = np.array(all_accuracies)
            mean_accuracies = np.mean(all_accuracies, axis=0)
            std_accuracies = np.std(all_accuracies, axis=0)
            
            # Calculate overall accuracy statistics
            mean_total = np.mean(mean_accuracies)
            std_total = np.std(mean_accuracies)
            
            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [0.3, 4]}, figsize=(15, 6))
            
            # Set the same y-limits for both plots
            y_min = 0  # Accuracies should never be negative
            y_max = max(100, np.max(mean_accuracies + std_accuracies))
            ax1.set_ylim(y_min, y_max)
            ax2.set_ylim(y_min, y_max)
            
            # Plot overall accuracy on the left
            if model_name == 'net-20':
                # Our model
                ax1.bar(0.3, mean_total, yerr=std_total, capsize=5, width=0.3,
                       color='#1f77b4', label='_nolegend_')
                # Reference model
                ax1.bar(0.7, 77, yerr=6, capsize=5, width=0.3,
                       color='gray', alpha=0.7, label='_nolegend_')
                ax1.set_xlim(0, 1)
                ax1.set_xticks([0.5])
            else:
                # Just our model
                ax1.bar(0.5, mean_total, yerr=std_total, capsize=5, width=0.4,
                       color='#1f77b4', label='_nolegend_')
                ax1.set_xlim(0, 1)
                ax1.set_xticks([0.5])
            
            ax1.set_xticklabels(['Total'])
            ax1.set_title('Total Accuracy', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylabel('Accuracy (%)', fontsize=9)
            
            # Plot classwise accuracies on the right
            x = np.arange(len(mean_accuracies))
            
            # if not tut
            if model_name == 'net-20' and not exp_type.startswith('TUT-experiment'):
                # Plot both our model and baseline
                width = 0.35
                # Our model bars
                rects1 = ax2.bar(x - width/2, mean_accuracies, width, yerr=std_accuracies,
                               capsize=5, color='#1f77b4', label='Our Model')
                
                # Reference model bars
                paper_acc_list = [paper_accuracies[label] for label in class_labels.values()]
                paper_spread_list = [paper_spreads[label] for label in class_labels.values()]
                rects2 = ax2.bar(x + width/2, paper_acc_list, width, yerr=paper_spread_list,
                               capsize=5, color='gray', alpha=0.7, label='HEAR-DS CNN Baseline')
            else:
                # Just plot our model
                width = 0.6
                ax2.bar(x, mean_accuracies, width, yerr=std_accuracies,
                       capsize=5, color='#1f77b4', label='Our Model')
            
            # Set labels
            labels = [class_labels.get(i, str(i)) for i in range(len(mean_accuracies))]
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            ax2.set_title('Classwise Accuracy', fontsize=10)
            
            # Add grid
            ax2.grid(True, alpha=0.3)
            
            # Add legend only for net-20
            if model_name == 'net-20':
                handles, labels = ax2.get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99))
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(exp_dir, 'classwise_accuracies.png'),
                        bbox_inches='tight', dpi=300)
            plt.close()

def plot_training_times(results: Dict, output_dir: str):
    """Plot training times comparison across models."""
    for exp_type, exp_results in results.items():
        model_times = {model: [] for model in MODELS.keys()}
        
        # Collect training times for each model
        for exp in exp_results:
            for model_name in MODELS.keys():
                if model_name in exp['duration']:
                    duration = exp['duration'][model_name]
                    if isinstance(duration, (int, float)):  # Ensure we're dealing with a number
                        model_times[model_name].append(duration)
                    elif isinstance(duration, list):  # Handle list of durations
                        model_times[model_name].extend(duration)
        
        # Calculate statistics
        mean_times = []
        std_times = []
        model_names = []
        
        for model_name, times in model_times.items():
            if times:  # Only include if we have data
                model_names.append(model_name)
                mean_times.append(np.mean(times))
                std_times.append(np.std(times))
        
        if not model_names:  # Skip if no data
            continue
            
        # Create bar plot
        plt.figure(figsize=(6, 4))  # Reduced from (12, 6)
        x = range(len(model_names))
        plt.bar(x, mean_times, yerr=std_times, capsize=5, alpha=0.7)
        
        plt.title(f'Training Times Comparison ({exp_type})', fontsize=10)
        plt.xlabel('Model', fontsize=9)
        plt.ylabel('Time (seconds)', fontsize=9)
        plt.xticks(x, model_names, rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        # Save training times plot in the main output directory since it compares all models
        plt.savefig(os.path.join(output_dir, f'{exp_type}_training_times.png'),
                   bbox_inches='tight', dpi=300)
        plt.close()

def plot_confusion_matrices(results: Dict, output_dir: str):
    """Plot mean confusion matrices for each model and experiment."""
    for exp_type, exp_results in results.items():
        for model_name in MODELS.keys():
            # Create model-specific and experiment-type directory
            exp_dir = os.path.join(output_dir, model_name, exp_type)
            
            # Load class labels
            class_labels = load_class_labels('models', exp_type, model_name)
            
            # Collect confusion matrices from all experiments
            all_matrices = []
            for exp in exp_results:
                if model_name in exp['naive_confusion_matrix_raw']:
                    matrix = exp['naive_confusion_matrix_raw'][model_name]
                    if isinstance(matrix, (list, np.ndarray)):
                        matrix = np.array(matrix)
                        # Remove extra dimension if present
                        if matrix.ndim > 2:
                            matrix = np.squeeze(matrix)
                        # Ensure values are between 0 and 1
                        if matrix.max() > 1.0:
                            matrix = matrix / 100.0
                        all_matrices.append(matrix)
            
            if not all_matrices:  # Skip if no data
                continue
                
            # Ensure all matrices have the same shape
            shapes = [m.shape for m in all_matrices]
            if not all(s == shapes[0] for s in shapes):
                print(f"Warning: Inconsistent matrix shapes for {model_name} in {exp_type}")
                continue
                
            # Convert to numpy array and compute mean
            matrices_array = np.stack(all_matrices)
            mean_matrix = np.mean(matrices_array, axis=0)
            
            # Create figure
            fig = plt.figure(figsize=(6, 4))
            gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.05], wspace=0.05)
            
            # Create axes
            ax = fig.add_subplot(gs[0])
            cax = fig.add_subplot(gs[1])  # Colorbar axis
            
            # Create custom colormap from yellow to purple
            colors = ['#ffff00', '#ff9900', '#ff3300', '#990099', '#330099']
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list('custom', colors)
            
            # Plot mean confusion matrix
            im = ax.imshow(mean_matrix, cmap=cmap, vmin=0, vmax=1.0)
            
            # Add black borders to each cell
            for i in range(mean_matrix.shape[0]):
                for j in range(mean_matrix.shape[1]):
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', linewidth=0.5)
                    ax.add_patch(rect)
            
            ax.set_title('Confusion Matrix', fontsize=9)
            
            # Set labels
            labels = [class_labels.get(i, str(i)) for i in range(len(mean_matrix))]
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels([])  # Remove x-axis labels
            ax.set_yticklabels(labels)
            
            # Add colorbar
            plt.colorbar(im, cax=cax)
            
            # Add labels
            ax.set_ylabel('True Scene', fontsize=9)
            ax.set_xlabel('Estimated Scene', fontsize=9)
            
            # Create descriptive title
            depth = model_name.split('-')[1]  # Extract depth number from model name
            # Determine optimizer type based on experiment type
            if exp_type.startswith('fixed-lr-sgd-AUG'):
                optimizer_type = 'SGD with Augmentation'
            elif exp_type.startswith('adam-early-stop'):
                optimizer_type = 'Adam with Early Stopping'
            elif exp_type.startswith('FIXED-fixed-lr-sgd'):
                optimizer_type = 'SGD without Augmentation'
            else:
                optimizer_type = exp_type
            title = f'Scene Classification Performance - {depth}-Layer Network ({optimizer_type})'
            fig.suptitle(title, y=1.05, fontsize=10, fontweight='bold')
            
            # Adjust bottom margin to make room for x-axis label and top margin for title
            plt.subplots_adjust(bottom=0.15, top=0.85)
            
            # Save figure in experiment-type directory
            plt.savefig(os.path.join(exp_dir, 'confusion_matrix.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()

def calculate_metrics(confusion_matrix: np.ndarray) -> tuple:
    """Calculate precision, recall, and F1 score for each class from a confusion matrix."""
    # Initialize arrays for metrics
    num_classes = confusion_matrix.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        # True positives
        tp = confusion_matrix[i, i]
        
        # False positives (sum of column i excluding diagonal)
        fp = np.sum(confusion_matrix[:, i]) - tp
        
        # False negatives (sum of row i excluding diagonal)
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        # Calculate metrics
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    return precision, recall, f1

def plot_class_metrics(results: Dict, output_dir: str):
    """Plot precision, recall, and F1 score for each class."""
    for exp_type, exp_results in results.items():
        for model_name in MODELS.keys():
            # Create model-specific and experiment-type directory
            exp_dir = os.path.join(output_dir, model_name, exp_type)
            
            # Load class labels
            class_labels = load_class_labels('models', exp_type, model_name)
            
            # Collect confusion matrices from all experiments
            all_matrices = []
            for exp in exp_results:
                if model_name in exp['naive_confusion_matrix_raw']:
                    matrix = exp['naive_confusion_matrix_raw'][model_name]
                    if isinstance(matrix, (list, np.ndarray)):
                        matrix = np.array(matrix)
                        # Remove extra dimension if present
                        if matrix.ndim > 2:
                            matrix = np.squeeze(matrix)
                        # Ensure values are between 0 and 1
                        if matrix.max() > 1.0:
                            matrix = matrix / 100.0
                        all_matrices.append(matrix)
            
            if not all_matrices:  # Skip if no data
                continue
                
            # Ensure all matrices have the same shape
            shapes = [m.shape for m in all_matrices]
            if not all(s == shapes[0] for s in shapes):
                print(f"Warning: Inconsistent matrix shapes for {model_name} in {exp_type}")
                continue
                
            # Convert to numpy array and compute mean
            matrices_array = np.stack(all_matrices)
            mean_matrix = np.mean(matrices_array, axis=0)
            
            # Calculate metrics
            precision, recall, f1 = calculate_metrics(mean_matrix)
            
            # Create figure with adjusted width to accommodate legend
            fig, ax = plt.subplots(figsize=(12, 4))
            
            # Set up the bar positions
            x = np.arange(len(precision))
            width = 0.25
            
            # Plot bars for each metric
            ax.bar(x - width, precision, width, label='Precision', alpha=0.7)
            ax.bar(x, recall, width, label='Recall', alpha=0.7)
            ax.bar(x + width, f1, width, label='F1 Score', alpha=0.7)
            
            # Add labels and title
            ax.set_xlabel('Class', fontsize=9)
            ax.set_ylabel('Score', fontsize=9)
            ax.set_title('Class-wise Performance Metrics', fontsize=10)
            
            # Set x-axis ticks and labels
            ax.set_xticks(x)
            ax.set_xticklabels([class_labels.get(i, str(i)) for i in range(len(precision))], 
                             rotation=45, ha='right')
            
            # Set y-axis limits
            ax.set_ylim(0, 1.0)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Move legend outside the plot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            # Create descriptive title
            depth = model_name.split('-')[1]
            # Determine optimizer type based on experiment type
            if exp_type.startswith('fixed-lr-sgd-AUG'):
                optimizer_type = 'SGD with Augmentation'
            elif exp_type.startswith('adam-early-stop'):
                optimizer_type = 'Adam with Early Stopping'
            elif exp_type.startswith('FIXED-fixed-lr-sgd'):
                optimizer_type = 'SGD without Augmentation'
            else:
                optimizer_type = exp_type
            fig.suptitle(f'Performance Metrics - {depth}-Layer Network ({optimizer_type})', 
                        y=1.05, fontsize=10, fontweight='bold')
            
            # Adjust layout to prevent legend overlap
            plt.tight_layout()
            
            # Save figure in experiment-type directory
            plt.savefig(os.path.join(exp_dir, 'class_metrics.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()

def plot_extreme_snr_performance(results: Dict, output_dir: str):
    """Plot performance metrics for extreme SNRs (-21 and 21)."""
    for exp_type, exp_results in results.items():
        for model_name in MODELS.keys():
            # Create model-specific and experiment-type directory
            exp_dir = os.path.join(output_dir, model_name, exp_type)
            
            # Load class labels
            class_labels = load_class_labels('models', exp_type, model_name)
            
            # Collect metrics for extreme SNRs
            snr_metrics = {}
            for exp in exp_results:
                if model_name in exp.get('per_snr_metrics', {}):
                    for snr, metrics in exp['per_snr_metrics'][model_name].items():
                        if int(snr) in [-21, 0, 21]:  # Only process extreme SNRs
                            if snr not in snr_metrics:
                                snr_metrics[snr] = []
                            snr_metrics[snr].append(metrics)
            
            if not snr_metrics:  # Skip if no data
                continue
            
            # Process each extreme SNR
            for snr, metrics_list in snr_metrics.items():
                # Calculate mean confusion matrix
                mean_conf_matrix = np.mean([m['confusion_matrix'] for m in metrics_list], axis=0)
                
                # Calculate mean class metrics
                num_classes = len(metrics_list[0]['class_metrics'])
                mean_precision = np.zeros(num_classes)
                mean_recall = np.zeros(num_classes)
                mean_f1 = np.zeros(num_classes)
                
                for class_idx in range(num_classes):
                    # Extract metrics for each class across all experiments
                    class_precisions = [m['class_metrics'][class_idx]['precision'] for m in metrics_list]
                    class_recalls = [m['class_metrics'][class_idx]['recall'] for m in metrics_list]
                    class_f1s = [m['class_metrics'][class_idx]['f1'] for m in metrics_list]
                    
                    # Calculate means
                    mean_precision[class_idx] = np.mean(class_precisions)
                    mean_recall[class_idx] = np.mean(class_recalls)
                    mean_f1[class_idx] = np.mean(class_f1s)
                
                # Create a figure with two subplots arranged vertically
                fig = plt.figure(figsize=(8, 10))
                gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
                
                # 1. Confusion Matrix (top)
                ax1 = fig.add_subplot(gs[0])
                # Create custom colormap from yellow to purple
                colors = ['#ffff00', '#ff9900', '#ff3300', '#990099', '#330099']
                from matplotlib.colors import LinearSegmentedColormap
                cmap = LinearSegmentedColormap.from_list('custom', colors)
                
                im = ax1.imshow(mean_conf_matrix, cmap=cmap, vmin=0, vmax=1.0)
                
                # Add black borders to each cell
                for i in range(mean_conf_matrix.shape[0]):
                    for j in range(mean_conf_matrix.shape[1]):
                        rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', linewidth=0.5)
                        ax1.add_patch(rect)
                
                ax1.set_title('Confusion Matrix', fontsize=9)
                ax1.set_xticks(np.arange(len(class_labels)))
                ax1.set_yticks(np.arange(len(class_labels)))
                ax1.set_xticklabels([])
                ax1.set_yticklabels([class_labels.get(i, str(i)) for i in range(len(class_labels))])
                ax1.set_ylabel('True Scene', fontsize=9)
                ax1.set_xlabel('Estimated Scene', fontsize=9)
                plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
                
                # 2. Class Metrics (bottom) - Horizontal bars
                ax3 = fig.add_subplot(gs[1])
                # Only plot speech classes (8-13)
                speech_classes = range(8, 14)
                speech_precision = mean_precision[8:14]
                speech_recall = mean_recall[8:14]
                speech_f1 = mean_f1[8:14]
                y = np.arange(len(speech_classes))
                width = 0.25
                
                # Plot horizontal bars
                ax3.barh(y - width, speech_precision, width, label='Precision', alpha=0.7)
                ax3.barh(y, speech_recall, width, label='Recall', alpha=0.7)
                ax3.barh(y + width, speech_f1, width, label='F1', alpha=0.7)
                
                ax3.set_title('Class Metrics', fontsize=9)
                ax3.set_yticks(y)
                ax3.set_yticklabels([class_labels.get(i, str(i)) for i in speech_classes])
                ax3.set_xlim(0, 1.0)
                # Move legend outside the plot
                ax3.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.0, 0.5))
                ax3.grid(True, alpha=0.3)
                
                # Add overall title
                depth = model_name.split('-')[1]
                # Determine optimizer type based on experiment type
                if exp_type.startswith('fixed-lr-sgd-AUG'):
                    optimizer_type = 'SGD with Augmentation'
                elif exp_type.startswith('adam-early-stop'):
                    optimizer_type = 'Adam with Early Stopping'
                elif exp_type.startswith('FIXED-fixed-lr-sgd'):
                    optimizer_type = 'SGD without Augmentation'
                else:
                    optimizer_type = exp_type
                title = f'Performance at SNR {snr}dB - {depth}-Layer Network ({optimizer_type})'
                fig.suptitle(title, y=1.05, fontsize=10, fontweight='bold')
                
                # Adjust layout to make room for the legend
                plt.tight_layout()
                
                # Save figure in experiment-type directory
                plt.savefig(os.path.join(exp_dir, f'snr_{snr}_performance.png'),
                          bbox_inches='tight', dpi=300)
                plt.close()

def plot_average_accuracies(results: Dict, output_dir: str):
    """Plot average accuracies with polynomial regression fitting for each experiment type."""
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    
    # Define colors for each model
    colors = plt.cm.rainbow(np.linspace(0, 1, len(MODELS)))
    
    # Paper reference points
    paper_depths = np.array([8, 20, 32])
    paper_accuracies = np.array([44, 70, 80])
    paper_errors = np.array([
        [6, 12],  # [lower, upper] for net-8: 35-55
        [5, 5],   # net-20: 65-75
        [3, 2],   # net-32: 77-82
    ])

    # TUT reference points
    tut_single_no_aug = 73.12
    tut_single_aug = 83.19
    tut_multi_no_aug = 74.30
    tut_multi_no_aug_std = 4.81
    tut_multi_aug = 87.29
    tut_multi_aug_std = 2.02
    
    # Create a plot for each experiment type
    for exp_type, exp_results in results.items():
        # Determine if this is a TUT experiment
        is_tut_experiment = exp_type.startswith('TUT-experiment')
        
        # Create figure - two subplots for TUT, one for others
        if is_tut_experiment:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        # Collect accuracies and model depths for our models
        model_data = {}  # Dict to store accuracies for each model
        
        # Collect accuracies from all experiments
        for exp in exp_results:  # Loop through all experiments
            for model_name in MODELS.keys():
                if model_name in exp['naive_total_accuracies']:
                    if model_name not in model_data:
                        model_data[model_name] = []
                    
                    accuracy = exp['naive_total_accuracies'][model_name]
                    if isinstance(accuracy, (int, float)):
                        accuracy = accuracy * 100 if accuracy <= 1 else accuracy
                    elif isinstance(accuracy, list):
                        accuracy = accuracy[-1] * 100 if accuracy[-1] <= 1 else accuracy[-1]
                    
                    model_data[model_name].append(accuracy)
        
        # Prepare data for plotting our models
        X = []  # Model depths
        y = []  # Mean accuracies
        yerr = []  # Standard deviations for error bars
        
        for model_name, accuracies in model_data.items():
            if accuracies:  # Only process if we have data
                depth = int(model_name.split('-')[1])
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies) if len(accuracies) > 1 else 0
                
                X.append(depth)
                y.append(mean_acc)
                yerr.append(std_acc)
        
        if not X:  # Skip if no data for this experiment type
            continue
            
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        
        # Fit polynomial regression for our models
        poly_reg = make_pipeline(PolynomialFeatures(3), LinearRegression())
        poly_reg.fit(X, y)
        
        # Generate points for smooth curve
        X_smooth = np.linspace(min(X), max(X), 100).reshape(-1, 1)
        y_smooth = poly_reg.predict(X_smooth)
        
        # Plot our models
        for i, (x, mean_acc, std_acc) in enumerate(zip(X.flatten(), y, yerr)):
            model_name = f'net-{int(x)}'
            ax1.errorbar(x, mean_acc, yerr=std_acc, fmt='o', capsize=5, markersize=8, 
                        color=colors[i], label=model_name, zorder=2)
        
        ax1.plot(X_smooth, y_smooth, 'k-', linewidth=2, label=None, zorder=1)
        
        # Plot paper reference points with dashed line
        if not is_tut_experiment:
            yerr_paper = [paper_errors[:, 0], paper_errors[:, 1]]
            ax1.errorbar(paper_depths, paper_accuracies, yerr=yerr_paper, fmt='s', 
                        color='gray', capsize=5, markersize=8, label='HEAR-DS CNN Baseline',
                        zorder=3)
            ax1.plot(paper_depths, paper_accuracies, 'k--', linewidth=2, zorder=2)
        
        # Customize first subplot/main plot
        ax1.set_xlabel('Model Depth', fontsize=12)
        ax1.set_ylabel('Accuracy [%]', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_xticks(X.flatten())
        ax1.set_xticklabels([f'net-{int(x)}' for x in X.flatten()])
        ax1.set_ylim(0, 100)
        
        if is_tut_experiment:
            ax1.set_title('Our Models', fontsize=12)
            
            # Plot TUT results on the right subplot
            x = np.array([1, 2])  # x positions for bars
            width = 0.35  # width of bars
            
            # Plot no augmentation results
            ax2.bar(x[0] - width/2, tut_single_no_aug, width, label='Single Resolution', 
                   color='#8B4513')  # Saddle Brown
            ax2.bar(x[1] - width/2, tut_multi_no_aug, width, yerr=tut_multi_no_aug_std,
                   capsize=5, label='Multi Resolution', color='#4B0082')  # Indigo
            
            # Plot augmentation results
            ax2.bar(x[0] + width/2, tut_single_aug, width, 
                   color='#8B4513', alpha=0.6, hatch='//')
            ax2.bar(x[1] + width/2, tut_multi_aug, width, yerr=tut_multi_aug_std,
                   capsize=5, color='#4B0082', alpha=0.6, hatch='//')

            # Add text annotations for the values
            ax2.text(x[0] - width/2, tut_single_no_aug + 1, f'{tut_single_no_aug:.1f}%', 
                    ha='center', va='bottom')
            ax2.text(x[0] + width/2, tut_single_aug + 1, f'{tut_single_aug:.1f}%', 
                    ha='center', va='bottom')
            ax2.text(x[1] - width/2, tut_multi_no_aug + 1, f'{tut_multi_no_aug:.1f}%', 
                    ha='center', va='bottom')
            ax2.text(x[1] + width/2, tut_multi_aug + 1, f'{tut_multi_aug:.1f}%', 
                    ha='center', va='bottom')

            # Customize right subplot
            ax2.set_ylabel('Accuracy [%]', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(x)
            ax2.set_xticklabels(['No Augmentation', 'With Augmentation'])
            ax2.set_ylim(0, 100)
            ax2.set_title('TUT Models', fontsize=12)
            ax2.text(0.5, -0.15, 
                    'Note: As per the paper, Single Resolution results show grouped predictions\n(averaging all predictions per file), hence no error bars.',
                    ha='center', va='center', transform=ax2.transAxes, 
                    style='italic', fontsize=9, color='#555555')

            # Create custom legend handles
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#8B4513', label='Single Resolution'),
                Patch(facecolor='#4B0082', label='Multi Resolution'),
            ]
            ax2.legend(handles=legend_elements, fontsize=10)

        # Add overall title based on experiment type
        if exp_type.startswith('fixed-lr-sgd-AUG'):
            title = 'Average Accuracy - SGD with Augmentation'
        elif exp_type.startswith('adam-early-stop'):
            title = 'Average Accuracy - Adam with Early Stopping'
        elif exp_type.startswith('FIXED-fixed-lr-sgd'):
            title = 'Average Accuracy - SGD without Augmentation'
        elif exp_type.startswith('TUT-experiment'):
            title = 'Average Accuracy Comparison with TUT Models'
        else:
            title = f'Average Accuracy - {exp_type}'
        
        plt.suptitle(title, fontsize=14, y=1.05)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Create experiment-specific filename
        if exp_type.startswith('fixed-lr-sgd-AUG'):
            filename = 'average_accuracies_sgd_aug.png'
        elif exp_type.startswith('adam-early-stop'):
            filename = 'average_accuracies_adam.png'
        elif exp_type.startswith('FIXED-fixed-lr-sgd'):
            filename = 'average_accuracies_sgd_no_aug.png'
        elif exp_type.startswith('TUT-experiment'):
            filename = 'average_accuracies_tut_comparison.png'
        else:
            filename = f'average_accuracies_{exp_type}.png'
        
        plt.savefig(os.path.join(output_dir, filename),
                    bbox_inches='tight', dpi=300)
        plt.close()

def plot_se_training_losses(results: Dict, output_dir: str):
    """Plot training losses for all environments across experiments."""
    # Define a color palette for different environments
    colors = sns.color_palette("husl", len(results))
    
    plt.figure(figsize=(12, 6))
    
    for env_idx, (env_name, env_results) in enumerate(results.items()):
        # Collect losses from all experiments for this environment
        all_losses = []
        for exp in env_results:
            if 'losses' in exp:
                losses = exp['losses']
                if isinstance(losses, list):
                    all_losses.append(losses)
        
        if not all_losses:  # Skip if no data for this environment
            continue
            
        # Find the maximum length
        max_len = max(len(loss) for loss in all_losses)
        
        # Pad shorter sequences with NaN
        padded_losses = []
        for loss in all_losses:
            if len(loss) < max_len:
                padded = np.pad(loss, (0, max_len - len(loss)), 
                              mode='constant', constant_values=np.nan)
            else:
                padded = loss
            padded_losses.append(padded)
        
        # Stack arrays and compute statistics
        losses_array = np.stack(padded_losses)
        mean_losses = np.nanmean(losses_array, axis=0)
        std_losses = np.nanstd(losses_array, axis=0)
        
        # Plot mean with confidence interval
        epochs = range(1, max_len + 1)
        plt.plot(epochs, mean_losses, label=env_name, color=colors[env_idx])
        plt.fill_between(epochs, 
                       mean_losses - std_losses,
                       mean_losses + std_losses,
                       alpha=0.2, color=colors[env_idx])
    
    plt.title('Training Loss Over Time (HEAR-DS Dataset)', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'hear-ds_training_losses.png'),
               bbox_inches='tight', dpi=300)
    plt.close()

def calculate_hear_ds_metrics(results: Dict):
    """
    Calculate and print the STOI and PESQ metrics for both baseline and processed audio.
    Results are formatted as LaTeX tables.
    """
    snrs = [-10, -5, 0, 5, 10]
    environments = ['Music', 'WindTurbulence', 'InTraffic', 'InVehicle', 'QuietIndoors', 'ReverberantEnvironment']
    
    # Initialize dictionaries to store metrics
    baseline_metrics = {env: {'stoi': {snr: [] for snr in snrs}, 
                            'pesq': {snr: [] for snr in snrs}} for env in environments}
    processed_metrics = {env: {'stoi': {snr: [] for snr in snrs}, 
                             'pesq': {snr: [] for snr in snrs}} for env in environments}
    
    # Collect metrics from all experiments
    for env, env_results in results.items():
        for exp in env_results:
            if 'results' in exp:
                results_data = exp['results']
                # Process baseline metrics
                if 'before_pesq' in results_data and 'before_stoi' in results_data:
                    for snr in snrs:
                        if snr in results_data['before_pesq']:
                            if results_data['before_pesq'][snr] is not None:
                                baseline_metrics[env]['pesq'][snr].append(results_data['before_pesq'][snr])
                        if snr in results_data['before_stoi']:
                            if results_data['before_stoi'][snr] is not None:
                                baseline_metrics[env]['stoi'][snr].append(results_data['before_stoi'][snr])
                
                # Process enhanced metrics
                if 'after_pesq' in results_data and 'after_stoi' in results_data:
                    for snr in snrs:
                        if snr in results_data['after_pesq']:
                            if results_data['after_pesq'][snr] is not None:
                                processed_metrics[env]['pesq'][snr].append(results_data['after_pesq'][snr])
                        if snr in results_data['after_stoi']:
                            if results_data['after_stoi'][snr] is not None:
                                processed_metrics[env]['stoi'][snr].append(results_data['after_stoi'][snr])
    
    # Calculate averages
    def calculate_averages(metrics_dict):
        avg_dict = {}
        for env, env_metrics in metrics_dict.items():
            avg_dict[env] = {'stoi': {}, 'pesq': {}}
            for metric in ['stoi', 'pesq']:
                for snr in snrs:
                    values = env_metrics[metric][snr]
                    # values is a list of lists, so we need to flatten it
                    flattened_values = [item for sublist in values for item in sublist]
                    avg_dict[env][metric][snr] = np.mean(flattened_values) if flattened_values else 0
        return avg_dict
    
    baseline_avg = calculate_averages(baseline_metrics)
    processed_avg = calculate_averages(processed_metrics)
    
    # Print LaTeX tables
    def print_latex_table(metrics, title):
        print(f"\\begin{{table}}[h]")
        print(f"   \\centering")
        print(f"   \\begin{{tabular}}{{l|{'c'*5}|{'c'*5}}}")
        print(f"      \\toprule")
        print(f"      Environment & \\multicolumn{{5}}{{c|}}{{STOI}} & \\multicolumn{{5}}{{c}}{{PESQ}} \\\\")
        print(f"      \\cmidrule(lr){{2-6}} \\cmidrule(lr){{7-11}}")
        print(f"      SNR (dB) & -10 & -5 & 0 & 5 & 10 & -10 & -5 & 0 & 5 & 10 \\\\")
        print(f"      \\midrule")
        
        for env in environments:
            stoi_values = [f"{metrics[env]['stoi'][snr]:.2f}" for snr in snrs]
            pesq_values = [f"{metrics[env]['pesq'][snr]:.2f}" for snr in snrs]
            print(f"      {env} & {' & '.join(stoi_values)} & {' & '.join(pesq_values)} \\\\")
        
        print(f"      \\bottomrule")
        print(f"   \\end{{tabular}}")
        print(f"   \\caption{{{title}}}")
        print(f"   \\label{{tab:hear-ds-{'baseline' if title.startswith('Baseline') else 'processed'}}}")
        print(f"\\end{{table}}")
    
    print("\nBaseline Metrics:")
    print_latex_table(baseline_avg, "Baseline (unprocessed) STOI and PESQ scores for different environments at various SNR levels.")
    
    print("\nProcessed Metrics:")
    print_latex_table(processed_avg, "Processed STOI and PESQ scores for different environments at various SNR levels.")

def plot_audio_spectrograms(sample_dir: str, output_dir: str, env: str):
    """
    Plot spectrograms of clean, noisy, and enhanced audio files side by side.
    Args:
        sample_dir: Directory containing the audio files and metrics
        output_dir: Directory to save the plots
    """
    import librosa
    import librosa.display
    
    # Get all sample directories
    sample_dirs = [d for d in os.listdir(sample_dir) if d.startswith('pesq_top_') or d.startswith('stoi_top_')]
    done = set()
    
    for sample in sample_dirs:
        # Construct file paths
        base = "_".join(sample.split('_')[0:4])
        if base in done:
            continue
        done.add(base)
        clean_path = os.path.join(sample_dir, base + '_clean.wav')  
        noisy_path = os.path.join(sample_dir, base + '_noisy.wav')
        enhanced_path = os.path.join(sample_dir, base + '_enhanced.wav')
        metrics_path = os.path.join(sample_dir, base + '_metrics.txt')

        # Read metrics
        with open(metrics_path, 'r') as f:
            metrics = f.read()
        
        # Read audio files
        y_clean, sr = librosa.load(clean_path, sr=None)
        y_noisy, _ = librosa.load(noisy_path, sr=None)
        y_enhanced, _ = librosa.load(enhanced_path, sr=None)
        
        # Calculate metrics for clean and noisy
        pesq_clean = pesq(sr, y_clean, y_clean, 'wb')
        stoi_clean = stoi(y_clean, y_clean, sr, extended=True)
        
        pesq_noisy = pesq(sr, y_clean, y_noisy, 'wb')
        stoi_noisy = stoi(y_clean, y_noisy, sr, extended=True)

        # Calculate metrics for enhanced
        pesq_enh = pesq(sr, y_clean, y_enhanced, 'wb')
        stoi_enh = stoi(y_clean, y_enhanced, sr, extended=True)
        
        # Create figure with 3 subplots and a metrics text box
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.2])
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        ax_metrics = fig.add_subplot(gs[3])
        
        # Plot spectrograms
        def plot_spectrogram(y, ax, title, pesq_val, stoi_val):
            # Use higher n_fft and hop_length for better resolution
            n_fft = 2048
            hop_length = 512
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
            img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, 
                                         hop_length=hop_length, ax=ax)
            ax.set_title(f'{title}\nPESQ: {pesq_val:.2f} | STOI: {stoi_val:.2f}', pad=10)
            # Remove all axis elements
            ax.axis('off')
            return img
        
        # Plot each spectrogram with metrics in title
        plot_spectrogram(y_clean, ax1, 'Clean Audio', pesq_clean, stoi_clean)
        plot_spectrogram(y_noisy, ax2, 'Noisy Audio', pesq_noisy, stoi_noisy)
        plot_spectrogram(y_enhanced, ax3, 'Enhanced Audio', pesq_enh, stoi_enh)
        
        # Add metrics as text in a separate subplot
        ax_metrics.axis('off')
        ax_metrics.text(0.5, 0.5, metrics, ha='center', va='center', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot with higher DPI
        os.makedirs(os.path.join(output_dir, env), exist_ok=True)
        plt.savefig(os.path.join(output_dir, env, f'spectrograms_{base}.png'),
                   bbox_inches='tight', dpi=600)
        plt.close()

        # Create log-mel spectrograms
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.2])
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        ax_metrics = fig.add_subplot(gs[3])
        
        # Plot log-mel spectrograms
        def plot_logmel_spectrogram(y, ax, title, pesq_val, stoi_val):
            # Use higher n_fft and hop_length for better resolution
            n_fft = 2048
            hop_length = 512
            n_mels = 128
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, 
                                                    hop_length=hop_length, n_mels=n_mels)
            # Convert to log scale
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Display the spectrogram
            img = librosa.display.specshow(log_mel_spec, y_axis='mel', x_axis='time', 
                                         sr=sr, hop_length=hop_length, ax=ax)
            ax.set_title(f'{title}\nPESQ: {pesq_val:.2f} | STOI: {stoi_val:.2f}', pad=10)
            # Remove all axis elements
            ax.axis('off')
            return img
        
        # Plot each log-mel spectrogram with metrics in title
        plot_logmel_spectrogram(y_clean, ax1, 'Clean Audio', pesq_clean, stoi_clean)
        plot_logmel_spectrogram(y_noisy, ax2, 'Noisy Audio', pesq_noisy, stoi_noisy)
        plot_logmel_spectrogram(y_enhanced, ax3, 'Enhanced Audio', pesq_enh, stoi_enh)
        
        # Add metrics as text in a separate subplot
        ax_metrics.axis('off')
        ax_metrics.text(0.5, 0.5, metrics, ha='center', va='center', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save log-mel spectrogram plot
        plt.savefig(os.path.join(output_dir, env, f'spectrograms_logmel_{base}.png'),
                   bbox_inches='tight', dpi=600)
        plt.close()

        # Create magnitude spectrograms
        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.2])
        
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        ax_metrics = fig.add_subplot(gs[3])
        
        # Plot magnitude spectrograms
        def plot_magnitude_spectrogram(y, ax, title, pesq_val, stoi_val):
            # Use higher n_fft and hop_length for better resolution
            n_fft = 2048
            hop_length = 512
            
            # Compute magnitude spectrogram
            D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
            
            # Display the spectrogram
            img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, 
                                         hop_length=hop_length, ax=ax)
            ax.set_title(f'{title}\nPESQ: {pesq_val:.2f} | STOI: {stoi_val:.2f}', pad=10)
            # Remove all axis elements
            ax.axis('off')
            return img
        
        # Plot each magnitude spectrogram with metrics in title
        plot_magnitude_spectrogram(y_clean, ax1, 'Clean Audio', pesq_clean, stoi_clean)
        plot_magnitude_spectrogram(y_noisy, ax2, 'Noisy Audio', pesq_noisy, stoi_noisy)
        plot_magnitude_spectrogram(y_enhanced, ax3, 'Enhanced Audio', pesq_enh, stoi_enh)
        
        # Add metrics as text in a separate subplot
        ax_metrics.axis('off')
        ax_metrics.text(0.5, 0.5, metrics, ha='center', va='center', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save magnitude spectrogram plot
        plt.savefig(os.path.join(output_dir, env, f'magnitude_spectra_{base}.png'),
                   bbox_inches='tight', dpi=600)
        plt.close()

def load_voice_enhancement_results(base_folder: str = 'experiments') -> Dict:
    """
    Load results from voice enhancement experiments.
    Returns a dictionary with experiment numbers as keys and results as values.
    """
    results = {}
    for folder in os.listdir(base_folder):
        if folder.startswith('voicebank-enh-exp-'):
            exp_num = int(folder.split('-')[-1])
            results[exp_num] = {}
            
            
            # Load metrics
            metrics_file = os.path.join(base_folder, folder, 'results.pkl')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'rb') as f:
                    metrics = pickle.load(f)
                    results[exp_num]['metrics'] = metrics
    
    return results

def plot_voice_enhancement_results(results: Dict, output_dir: str):
    """Plot training curves and metrics for voice enhancement experiments."""
    # Plot training curves
    plt.figure(figsize=(10, 6))
    
    # Collect all losses
    all_losses = []
    for exp_num, exp_data in results.items():
        if 'metrics' in exp_data and 'losses' in exp_data['metrics']:
            losses = exp_data['metrics']['losses']
            all_losses.append(losses)
    
    if all_losses:
        # Find maximum length
        max_len = max(len(losses) for losses in all_losses)
        
        # Pad shorter sequences with NaN
        padded_losses = []
        for losses in all_losses:
            if len(losses) < max_len:
                padded = np.pad(losses, (0, max_len - len(losses)), 
                              mode='constant', constant_values=np.nan)
            else:
                padded = losses
            padded_losses.append(padded)
        
        # Stack arrays and compute statistics
        losses_array = np.stack(padded_losses)
        mean_losses = np.nanmean(losses_array, axis=0)
        min_losses = np.nanmin(losses_array, axis=0)
        max_losses = np.nanmax(losses_array, axis=0)
        
        # Plot mean with spread region
        epochs = range(1, max_len + 1)
        plt.plot(epochs, mean_losses, label='Mean Loss', color='blue')
        plt.fill_between(epochs, min_losses, max_losses,
                       alpha=0.2, color='blue', label='_nolegend_')
    
    plt.title('Training Loss over Time (VOICEBANK + DEMAND)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'voice_enhancement_training_curves.png'),
               bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create metrics table
    metrics_data = []
    for exp_num, exp_data in results.items():
        if 'metrics' in exp_data:
            metrics = exp_data['metrics']
            row = {
                'Experiment': f'Exp {exp_num}',
                'STOI (Noisy)': metrics.get('before_stoi', 0),
                'STOI (Enhanced)': metrics.get('after_stoi', 0),
                'PESQ (Noisy)': metrics.get('before_pesq', 0),
                'PESQ (Enhanced)': metrics.get('after_pesq', 0)
            }
            metrics_data.append(row)
    
    if metrics_data:
        df = pd.DataFrame(metrics_data)
        # print("\nVoice Enhancement Metrics:")
        # print(df.to_string(index=False))
        
        # Save metrics to CSV
        df.to_csv(os.path.join(output_dir, 'voice_enhancement_metrics.csv'), index=False)
        
        # Generate LaTeX table
        print("\nLaTeX Table:")
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\begin{tabular}{l|cc}")
        print("\\toprule")
        print("Condition & STOI & PESQ \\\\")
        print("\\midrule")
        
        # Calculate mean values across experiments
        mean_stoi_noisy = np.mean([list(row['STOI (Noisy)'].values()) for row in metrics_data])
        mean_stoi_enh = np.mean([list(row['STOI (Enhanced)'].values()) for row in metrics_data])
        mean_pesq_noisy = np.mean([list(row['PESQ (Noisy)'].values()) for row in metrics_data])
        mean_pesq_enh = np.mean([list(row['PESQ (Enhanced)'].values()) for row in metrics_data])
        
        print(f"Noisy & {mean_stoi_noisy:.2f} & {mean_pesq_noisy:.2f} \\\\")
        print(f"SpecMix & X & X \\\\")
        print(f"Our Method & {mean_stoi_enh:.2f} & {mean_pesq_enh:.2f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\caption{Average STOI and PESQ scores for noisy and enhanced speech}")
        print("\\label{tab:voice_enhancement_metrics}")
        print("\\end{table}")

def generate_all_plots(output_dir: str = 'output/visualizations'):
    """Generate all visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and plot voice enhancement results
    # voice_results = load_voice_enhancement_results()
    # plot_voice_enhancement_results(voice_results, output_dir)
    
    # Load and plot SE results
    # results_se = load_experiment_results_SE_HEARDS()
    # calculate_hear_ds_metrics(results_se)
    
    # Plot spectrograms for top samples
    sample_dirs = [
        # ('WindTurbulence', '/workspace/experiments/hear-ds-speech-enh-exp-2/top_samples_WindTurbulence'),
        # ('InTraffic', '/workspace/experiments/hear-ds-speech-enh-exp-2/top_samples_InTraffic'),
        ('InVehicle', '/workspace/experiments/hear-ds-speech-enh-exp-2/top_samples_InVehicle'),
        ('Music', '/workspace/experiments/hear-ds-speech-enh-exp-2/top_samples_Music'),
        # ('QuietIndoors', '/workspace/experiments/hear-ds-speech-enh-exp-2/top_samples_QuietIndoors'),
        # ('ReverberantEnvironment', '/workspace/experiments/hear-ds-speech-enh-exp-2/top_samples_ReverberantEnvironment')
    ]
    for env, sample_dir in sample_dirs:
        plot_audio_spectrograms(sample_dir, output_dir, env)


if __name__ == "__main__":
    generate_all_plots() 