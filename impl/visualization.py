import os 
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List
from constants import MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def load_experiment_results(base_folder: str = 'models', max_experiments: int = 3) -> Dict:
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
            # x axis max 120
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
            
            if model_name == 'net-20':
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
    
    # Create a plot for each experiment type
    for exp_type, exp_results in results.items():
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Collect accuracies and model depths
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
        
        # Prepare data for plotting
        X = []  # Model depths
        y = []  # Mean accuracies
        yerr = []  # Standard deviations for error bars
        
        for model_name, accuracies in model_data.items():
            if accuracies:  # Only process if we have data
                depth = int(model_name.split('-')[1])
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies) if len(accuracies) > 1 else 0
                
                print(f"{exp_type} - {model_name}: mean={mean_acc:.2f}, std={std_acc:.2f}, n={len(accuracies)}")  # Debug print
                
                X.append(depth)
                y.append(mean_acc)
                yerr.append(std_acc)
        
        if not X:  # Skip if no data for this experiment type
            continue
            
        X = np.array(X).reshape(-1, 1)
        y = np.array(y)
        
        # Fit polynomial regression
        poly_reg = make_pipeline(PolynomialFeatures(3), LinearRegression())
        poly_reg.fit(X, y)
        
        # Generate points for smooth curve
        X_smooth = np.linspace(min(X), max(X), 100).reshape(-1, 1)
        y_smooth = poly_reg.predict(X_smooth)
        
        # Plot with error bars and smooth curve
        for i, (x, mean_acc, std_acc) in enumerate(zip(X.flatten(), y, yerr)):
            model_name = f'net-{int(x)}'
            plt.errorbar(x, mean_acc, yerr=std_acc, fmt='o', capsize=5, markersize=8, 
                        color=colors[i], label=model_name, zorder=2)
        
        plt.plot(X_smooth, y_smooth, 'k-', linewidth=2, label=None, zorder=1)
        
        # Plot paper reference points with dashed line
        yerr_paper = [paper_errors[:, 0], paper_errors[:, 1]]  # Split into lower and upper errors
        plt.errorbar(paper_depths, paper_accuracies, yerr=yerr_paper, fmt='s', 
                    color='gray', capsize=5, markersize=8, label='HEAR-DS CNN Baseline',
                    zorder=3)
        plt.plot(paper_depths, paper_accuracies, 'k--', linewidth=2, zorder=2)
        
        # Customize plot
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Accuracy [%]', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Move legend to the right side outside the plot
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set x-ticks to model names
        plt.xticks(X.flatten(), [f'net-{int(x)}' for x in X.flatten()])
        
        # Set y-axis limits from 0 to 100
        plt.ylim(0, 100)
        
        # Add title based on experiment type
        if exp_type.startswith('fixed-lr-sgd-AUG'):
            title = 'Average Accuracy - SGD with Augmentation'
        elif exp_type.startswith('adam-early-stop'):
            title = 'Average Accuracy - Adam with Early Stopping'
        elif exp_type.startswith('FIXED-fixed-lr-sgd'):
            title = 'Average Accuracy - SGD without Augmentation'
        else:
            title = f'Average Accuracy - {exp_type}'
        
        plt.title(title, fontsize=12, pad=20)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Create experiment-specific filename
        if exp_type.startswith('fixed-lr-sgd-AUG'):
            filename = 'average_accuracies_sgd_aug.png'
        elif exp_type.startswith('adam-early-stop'):
            filename = 'average_accuracies_adam.png'
        elif exp_type.startswith('FIXED-fixed-lr-sgd'):
            filename = 'average_accuracies_sgd_no_aug.png'
        else:
            filename = f'average_accuracies_{exp_type}.png'
        
        plt.savefig(os.path.join(output_dir, filename),
                    bbox_inches='tight', dpi=300)
        plt.close()

def generate_all_plots(output_dir: str = 'output/visualizations'):
    """Generate all visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    results = load_experiment_results()
    
    # Create subdirectories for each model and experiment type
    for model_name in MODELS.keys():
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        for exp_type in results.keys():
            exp_dir = os.path.join(model_dir, exp_type)
            os.makedirs(exp_dir, exist_ok=True)
    
    # plot_training_losses(results, output_dir)
    # plot_classwise_accuracies(results, output_dir)
    # plot_training_times(results, output_dir)
    # plot_confusion_matrices(results, output_dir)
    # plot_class_metrics(results, output_dir)
    plot_extreme_snr_performance(results, output_dir)
    # plot_average_accuracies(results, output_dir)

if __name__ == "__main__":
    generate_all_plots() 