import os 
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List
from constants import MODELS

def load_experiment_results(base_folder: str = 'models', max_experiments: int = 5) -> Dict:
    """
    Load results from all experiments for different model configurations.
    Returns a dictionary with experiment types as keys and lists of results as values.
    """
    results = {}
    for folder in os.listdir(base_folder):
        if not os.path.isdir(os.path.join(base_folder, folder)):
            continue
            
        for exp_type in ['fixed-lr-sgd-', 'adam-early-stop-']:
            if folder.startswith(exp_type):
                base_name = exp_type.rstrip('-')
                if base_name not in results:
                    results[base_name] = []
                
                experiment_number = int(folder.split('-')[-1])
                if experiment_number <= max_experiments:
                    with open(f'{base_folder}/{folder}/{experiment_number}/results.pkl', 'rb') as f:
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
        plt.figure(figsize=(12, 6))
        
        for model_idx, model_name in enumerate(MODELS.keys()):
            # Collect losses from all experiments for this model
            all_losses = []
            for exp in exp_results:
                if model_name in exp['losses']:
                    losses = exp['losses'][model_name]
                    # Handle the case where losses is a list of lists
                    if isinstance(losses, list):
                        if losses and isinstance(losses[0], list):
                            all_losses.extend(losses)  # Extend if we have a list of lists
                        else:
                            all_losses.append(losses)  # Append if we have a single list
            
            if not all_losses:  # Skip if no data for this model
                continue
                
            # Convert lists to numpy arrays for processing
            loss_arrays = [np.array(loss) for loss in all_losses]
            
            # Find the maximum length
            max_len = max(len(loss) for loss in loss_arrays)
            
            # Pad shorter sequences with NaN
            padded_losses = []
            for loss in loss_arrays:
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
    
        plt.title(f'Training Loss Over Time ({exp_type})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{exp_type}_training_losses.png'))
        plt.close()

def plot_classwise_accuracies(results: Dict, output_dir: str):
    """Plot classwise accuracies for each model and experiment."""
    for exp_type, exp_results in results.items():
        for model_name in MODELS.keys():
            # Load model-specific class labels
            class_labels = load_class_labels('models', exp_type, model_name)
            
            # Collect classwise accuracies for all experiments
            all_accuracies = []
            all_total_accuracies = []  # For overall accuracy
            for exp in exp_results:
                if model_name in exp['class_accuracies']:
                    acc = exp['class_accuracies'][model_name]
                    if isinstance(acc, (list, np.ndarray)):
                        if isinstance(acc[0], (list, np.ndarray)):
                            acc_list = [np.array(a) * 100 if np.max(a) <= 1 else np.array(a) for a in acc]
                            all_accuracies.extend(acc_list)
                        else:
                            acc_array = np.array(acc) * 100 if np.max(acc) <= 1 else np.array(acc)
                            all_accuracies.append(acc_array)
                            
                if model_name in exp['total_accuracies']:
                    total_acc = exp['total_accuracies'][model_name]
                    if isinstance(total_acc, (list, np.ndarray)):
                        total_acc_list = [a * 100 if a <= 1 else a for a in total_acc]
                        all_total_accuracies.extend(total_acc_list)
                    else:
                        total_acc_val = total_acc * 100 if total_acc <= 1 else total_acc
                        all_total_accuracies.append(total_acc_val)
            
            if not all_accuracies:  # Skip if no data
                continue
                
            # Convert to numpy array
            all_accuracies = np.array(all_accuracies)
            mean_accuracies = np.mean(all_accuracies, axis=0)
            std_accuracies = np.std(all_accuracies, axis=0)
            
            # Calculate overall accuracy statistics
            mean_total = np.mean(all_total_accuracies) if all_total_accuracies else 0
            std_total = np.std(all_total_accuracies) if all_total_accuracies else 0
            
            # Create figure with two subplots side by side
            # Make the first subplot just wide enough for the bar (0.3) plus padding (0.2 on each side)
            # So total width of first plot is 0.7 relative to second plot's 4
            fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [0.3, 4]}, figsize=(15, 6))
            
            # Set the same y-limits for both plots
            y_min = 0  # Accuracies should never be negative
            y_max = max(100, np.max(mean_accuracies + std_accuracies), mean_total + std_total)
            ax1.set_ylim(y_min, y_max)
            ax2.set_ylim(y_min, y_max)
            
            # Remove the box around the plots
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            # Plot overall accuracy on the left with thinner bar
            ax1.bar(x=0.5, height=mean_total, yerr=std_total, capsize=0,  # Remove error bar caps
                   alpha=0.7, color='#1f77b4', width=0.8)  # Extremely thin bar
            ax1.set_xlim(0, 1)  # Set explicit x-axis limits
            ax1.set_xticks([0.5])  # Center the "Total" label
            ax1.set_xticklabels(['Total'])
            ax1.set_title('Total Accuracy')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylabel('Accuracy (%)')
            
            # Plot classwise accuracies on the right
            x = np.arange(len(mean_accuracies))  # Use numpy arange instead of range
            ax2.bar(x, mean_accuracies, yerr=std_accuracies, capsize=0,  # Remove error bar caps
                   alpha=0.7, color='#1f77b4')  # Default width
            ax2.set_title('Classwise Accuracy')
            
            # Set class labels
            labels = [class_labels.get(i, str(i)) for i in range(len(mean_accuracies))]
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, rotation=45, ha='right')
            
            # Remove y-ticks and label from right plot
            ax2.set_yticklabels([])
            ax2.set_ylabel('')
            
            # Add grid to both plots
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)
            
            # Remove subplot titles and add a single title for the figure
            fig.suptitle(f'Accuracies - {model_name} ({exp_type})', y=1.02)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{exp_type}_{model_name}_classwise_accuracies.png'),
                       bbox_inches='tight')
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
        plt.figure(figsize=(12, 6))
        x = range(len(model_names))
        plt.bar(x, mean_times, yerr=std_times, capsize=5, alpha=0.7)
        
        plt.title(f'Training Times Comparison ({exp_type})')
        plt.xlabel('Model')
        plt.ylabel('Time (seconds)')
        plt.xticks(x, model_names, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{exp_type}_training_times.png'))
        plt.close()

def plot_confusion_matrices(results: Dict, output_dir: str):
    """Plot mean and standard deviation confusion matrices for each model and experiment."""
    for exp_type, exp_results in results.items():
        for model_name in MODELS.keys():
            # Load class labels
            class_labels = load_class_labels('models', exp_type, model_name)
            
            # Collect confusion matrices from all experiments
            all_matrices = []
            for exp in exp_results:
                if model_name in exp['confusion_matrix_raw']:
                    matrix = exp['confusion_matrix_raw'][model_name]
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
                
            # Convert to numpy array and compute statistics
            matrices_array = np.stack(all_matrices)
            mean_matrix = np.mean(matrices_array, axis=0)
            std_matrix = np.std(matrices_array, axis=0)
            
            # Create figure and gridspec
            fig = plt.figure(figsize=(12, 5))
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.05)
            
            # Create axes
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            cax = fig.add_subplot(gs[2])  # Colorbar axis
            
            # Create custom colormap from yellow to purple
            colors = ['#ffff00', '#ff9900', '#ff3300', '#990099', '#330099']
            from matplotlib.colors import LinearSegmentedColormap
            cmap = LinearSegmentedColormap.from_list('custom', colors)
            
            # Plot mean confusion matrix
            im1 = ax1.imshow(mean_matrix, cmap=cmap, vmin=0, vmax=1.0)
            ax1.set_title('Mean of Confusion Matrix')
            
            # Plot standard deviation matrix
            im2 = ax2.imshow(std_matrix, cmap=cmap, vmin=0, vmax=np.max(std_matrix))
            ax2.set_title('Standard Deviation')
            
            # Set labels
            labels = [class_labels.get(i, str(i)) for i in range(len(mean_matrix))]
            
            # Set ticks and labels for both matrices
            for ax in [ax1, ax2]:
                ax.set_xticks(np.arange(len(labels)))
                ax.set_yticks(np.arange(len(labels)))
                ax.set_xticklabels([])  # Remove x-axis labels from both matrices
                ax.set_yticklabels([]) if ax == ax2 else ax.set_yticklabels(labels)  # Only show y labels on first plot
            
            # Add colorbar without label
            plt.colorbar(im1, cax=cax)
            
            # Add labels
            ax1.set_ylabel('True Scene')
            
            # Add shared x-axis label
            fig.text(0.5, 0.02, 'Estimated Scene', ha='center')
            
            # Create descriptive title
            depth = model_name.split('-')[1]  # Extract depth number from model name
            optimizer_type = 'SGD' if exp_type == 'fixed-lr-sgd' else 'Adam with Early Stopping'
            title = f'Scene Classification Performance - {depth}-Layer Network ({optimizer_type})'
            fig.suptitle(title, y=1.05, fontsize=12, fontweight='bold')
            
            # Adjust bottom margin to make room for shared x-axis label and top margin for title
            plt.subplots_adjust(bottom=0.15, top=0.85)
            
            # Save figure
            plt.savefig(os.path.join(output_dir, f'{exp_type}_{model_name}_confusion_matrices.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()

def generate_all_plots(output_dir: str = 'output/visualizations'):
    """Generate all visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    results = load_experiment_results()
    
    plot_training_losses(results, output_dir)
    plot_classwise_accuracies(results, output_dir)
    plot_training_times(results, output_dir)
    plot_confusion_matrices(results, output_dir)

if __name__ == "__main__":
    generate_all_plots() 