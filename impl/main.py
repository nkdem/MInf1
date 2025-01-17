import os
import numpy as np
import matplotlib.pyplot as plt
from constants import MODELS
from hear_ds import HEARDS
from train import train
from test import test
import seaborn as sns

 
def run_experiment(num_epochs, batch_size, number_of_experiments, max_lr=None, learning_rates=None, cuda=False):
    experiment_name = f'{num_epochs}_epochs_{batch_size}_batch'
    for i in range(1, number_of_experiments + 1):
        base_dir = f'models/{experiment_name}/{i}'
        os.makedirs(base_dir, exist_ok=True)
        train(dataset=dataset, base_dir=base_dir, num_epochs=num_epochs, batch_size=batch_size, max_lr=max_lr, learning_rates=learning_rates, cuda=cuda)

    # Initialize data structures to store results across all experiments
    all_confusion_matrices = {model: [] for model in MODELS.keys()}
    all_accuracies = {model: [] for model in MODELS.keys()}
    all_losses = {model: [] for model in MODELS.keys()}
    all_training_times = {model: [] for model in MODELS.keys()}


    for i in range(1, number_of_experiments + 1):
        base_dir = f'models/{experiment_name}/{i}'
        training_data = [[] for _ in range(len(MODELS))]
        testing_data = [[] for _ in range(len(MODELS))]

        for j, (model, (cnn1_channels, cnn2_channels, fc_neurons)) in enumerate(MODELS.items()):
            root_dir = os.path.join(base_dir, model)
            if os.path.exists(root_dir):
                # Get training and testing data
                train_data = os.path.join(root_dir, 'train_files.txt')
                test_data = os.path.join(root_dir, 'test_files.txt')
                if os.path.exists(train_data) and os.path.exists(test_data):
                    with open(train_data, 'r') as f:
                        training_data[j] = f.readlines()
                    with open(test_data, 'r') as f:
                        testing_data[j] = f.readlines()
        
        # Check if training and testing data are the same across models
        for j in range(len(MODELS) - 1):
            assert training_data[j] == training_data[j + 1], 'Training data is not the same across models'
            assert testing_data[j] == testing_data[j + 1], 'Testing data is not the same across models'

    # Collect data from all experiments
    for i in range(1, number_of_experiments + 1):
        base_dir = f'models/{experiment_name}/{i}'
        
        # Collect results for each model
        for model, (cnn1_channels, cnn2_channels, fc_neurons) in MODELS.items():
            root_dir = os.path.join(base_dir, model)
            if os.path.exists(root_dir):
                # Get confusion matrix and accuracy
                confusion_matrix, accuracy = test(dataset, root_dir, model, cnn1_channels, cnn2_channels, fc_neurons, cuda=cuda)
                all_confusion_matrices[model].append(confusion_matrix)
                all_accuracies[model].append(accuracy)
                
                # Get losses
                loss_file = os.path.join(root_dir, 'losses.txt')
                if os.path.exists(loss_file):
                    with open(loss_file, 'r') as f:
                        losses = [float(line.strip()) for line in f.readlines()]
                        all_losses[model].append(losses)
                
                # Get training time
                duration_file = os.path.join(root_dir, 'duration.txt')
                if os.path.exists(duration_file):
                    with open(duration_file, 'r') as f:
                        duration = float(f.readline().strip())
                        all_training_times[model].append(duration / 60)  # Convert to minutes

    # Create output directory for combined results
    output_dir = f'models/{experiment_name}/combined_results'
    os.makedirs(output_dir, exist_ok=True)

    # Standard deviation confusion matrix
    for model in MODELS.keys():
        if all_confusion_matrices[model]:
            std_conf_matrix = np.std(all_confusion_matrices[model], axis=0)
            
            # Create figure with specific size ratio and extra space at bottom
            plt.figure(figsize=(12, 10))

            # Create the heatmap
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

            # Read int_to_label mapping from any experiment (they should be the same)
            base_dir = f'models/{experiment_name}/1/{model}'
            with open(os.path.join(base_dir, 'int_to_label.txt'), 'r') as f:
                int_to_label = dict(line.strip().split() for line in f)
                int_to_label = {int(k): v for k, v in int_to_label.items()}

            num_classes = len(int_to_label)

            # Add x-axis labels with better formatting
            plt.xticks(np.arange(num_classes) + 0.5, [int_to_label[i] for i in range(num_classes)], 
                    rotation=45,
                    ha='right',
                    rotation_mode='anchor')

            # Adjust y-axis labels
            plt.yticks(np.arange(num_classes) + 0.5, [int_to_label[i] for i in range(num_classes)], rotation=0)
            plt.ylabel('True Scene')

            # Set title
            plt.title(f"Standard Deviation of Confusion Matrix - {model} [{experiment_name}]")

            # Adjust layout with more bottom space
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)

            # Move colorbar down
            ax = plt.gca()
            ax.set_xlabel('Estimated Scene')
            colorbar = ax.collections[0].colorbar
            colorbar.ax.set_position([colorbar.ax.get_position().x0, 
                                    colorbar.ax.get_position().y0 - 0.05,
                                    colorbar.ax.get_position().width,
                                    colorbar.ax.get_position().height])

            plt.savefig(os.path.join(output_dir, f'{model}_std_confusion_matrix.png'),
                    bbox_inches='tight',
                    dpi=300)
            plt.close()

    # Classwise Accuracy
    for model in MODELS.keys():
        if all_confusion_matrices[model]:
            # Calculate mean and std of class accuracies across experiments
            class_accuracies_per_exp = []
            for conf_matrix in all_confusion_matrices[model]:
                acc = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
                class_accuracies_per_exp.append(acc)
            
            class_accuracies_array = np.array(class_accuracies_per_exp)
            mean_accuracies = np.mean(class_accuracies_array, axis=0)
            std_accuracies = np.std(class_accuracies_array, axis=0)
            
            plt.figure(figsize=(12, 6))
            x = range(len(mean_accuracies))
            plt.bar(x, mean_accuracies, yerr=std_accuracies, capsize=5)
            
            # Read class labels
            with open(os.path.join(base_dir, 'int_to_label.txt'), 'r') as f:
                int_to_label = dict(line.strip().split() for line in f)
                int_to_label = {int(k): v for k, v in int_to_label.items()}
            
            plt.xticks(x, [int_to_label[i] for i in range(len(mean_accuracies))], rotation=45, ha='right')
            plt.title(f'Classwise Accuracy - {model} [{experiment_name}]')
            plt.xlabel('Class')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model}_classwise_accuracy.png'))
            plt.close()


    # Model Accuracies with Standard Deviation
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

    # Loss against epochs
    plt.figure(figsize=(12, 6))
    for model in MODELS.keys():
        if all_losses[model]:
            losses_array = np.array(all_losses[model])
            mean_losses = np.mean(losses_array, axis=0)
            std_losses = np.std(losses_array, axis=0)
            epochs = range(1, len(mean_losses) + 1)
            
            plt.plot(epochs, mean_losses, label=model)
            plt.fill_between(epochs, 
                            mean_losses - std_losses,
                            mean_losses + std_losses,
                            alpha=0.2)

    plt.title(f'Train Loss against Epochs [{experiment_name}]')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_losses.png'))
    plt.close()

    # Training Time Analysis
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

if __name__ == '__main__':
    cuda = False
    root_dir = '/Users/nkdem/Downloads/HEAR-DS' if not cuda else '/home/s2203859/minf-1/dataset'
    dataset = HEARDS(root_dir=root_dir, cuda=cuda)
    run_experiment(num_epochs=1, batch_size=16, number_of_experiments=2, learning_rates=[0.05,0.01], cuda=cuda)