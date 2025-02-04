import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from constants import MODELS
from hear_ds import HEARDS
from train import train
from test import test
import seaborn as sns

from pesq import pesq
from pystoi import stoi
import soundfile as sf
def base_dir_name(num_epochs, batch_size, aug):
    return f'models/{num_epochs}_epochs_{batch_size}_batch_{"aug" if aug else "no_aug"}'
 
def run_experiment(num_epochs, batch_size, number_of_experiments, max_lr=None, learning_rates=None, cuda=False):
    experiment_name = f'{num_epochs}_epochs_{batch_size}_batch'
    
    # Initialize metrics storage for speech quality
    metrics = {
        'no_aug': {snr: {'pesq': [], 'stoi': []} for snr in dataset.snr_levels},
        'aug': {snr: {'pesq': [], 'stoi': []} for snr in dataset.snr_levels}
    }
    
    # Initialize data structures for accuracy and performance metrics
    all_confusion_matrices = {model: [] for model in MODELS.keys()}
    all_accuracies = {model: [] for model in MODELS.keys()}
    all_losses = {model: [] for model in MODELS.keys()}
    all_training_times = {model: [] for model in MODELS.keys()}

    for i in range(1, number_of_experiments + 1):
        # Run without augmentation
        dataset.augmentation = False
        base_dir = base_dir_name(num_epochs, batch_size, False)
        train_data, test_data = train(dataset=dataset, base_dir=base_dir, num_epochs=num_epochs, 
                                    batch_size=batch_size, max_lr=max_lr, learning_rates=learning_rates, cuda=cuda)
        
        # Calculate speech metrics for non-augmented test data
        for test_sample in test_data:
            # ['/Users/nkdem/Downloads/HEAR-DS/Music/Speech/-6/10_103_43_002_ITC_L_16kHz.wav', '/Users/nkdem/Downloads/HEAR-DS/Music/Speech/-6/10_103_43_002_ITC_R_16kHz.wav']
            if test_sample[3] is not None:
# test_sample[0] = '/Users/nkdem/Downloads/HEAR-DS/Music/Speech/-6/10_103_43_002_ITC_L_16kHz.wav'
                snr =  int(test_sample[3])
                clean_signal, _ = sf.read(test_sample[0][0].replace('_noisy', '_clean'))
                noisy_signal, _ = sf.read(test_sample[0][0])
                
                pesq_score, stoi_score = calculate_metrics(clean_signal, noisy_signal)
                if pesq_score and stoi_score:
                    metrics['no_aug'][snr]['pesq'].append(pesq_score)
                    metrics['no_aug'][snr]['stoi'].append(stoi_score)

        # Collect performance metrics for non-augmented
        collect_performance_metrics(base_dir, all_confusion_matrices, all_accuracies, all_losses, all_training_times, cuda)

        # Run with augmentation
        dataset.augmentation = True
        base_dir = base_dir_name(num_epochs, batch_size, True)
        train_data, test_data = train(dataset=dataset, base_dir=base_dir, num_epochs=num_epochs, 
                                    batch_size=batch_size, max_lr=max_lr, learning_rates=learning_rates, cuda=cuda)
        
        # Calculate speech metrics for augmented test data
        for test_sample in test_data:
            if test_sample[3] is not None:
                snr = int(test_sample[3])
                clean_signal, _ = sf.read(test_sample[0][0].replace('_noisy', '_clean'))
                noisy_signal, _ = sf.read(test_sample[0][0])
                
                pesq_score, stoi_score = calculate_metrics(clean_signal, noisy_signal)
                if pesq_score and stoi_score:
                    metrics['aug'][snr]['pesq'].append(pesq_score)
                    metrics['aug'][snr]['stoi'].append(stoi_score)

        # Collect performance metrics for augmented
        collect_performance_metrics(base_dir, all_confusion_matrices, all_accuracies, all_losses, all_training_times, cuda)

    # Create output directory
    output_dir = f'models/{experiment_name}/combined_results'
    os.makedirs(output_dir, exist_ok=True)

    # Generate speech metrics table
    generate_speech_metrics_table(metrics, output_dir)

    # Generate performance visualizations
    generate_confusion_matrices(all_confusion_matrices, experiment_name, output_dir, dataset)
    generate_classwise_accuracy(all_confusion_matrices, experiment_name, output_dir, dataset)  # Add dataset parameter here too
    generate_model_accuracies(all_accuracies, experiment_name, output_dir)
    generate_loss_plots(all_losses, experiment_name, output_dir)
    generate_training_times(all_training_times, experiment_name, output_dir)

def collect_performance_metrics(base_dir, all_confusion_matrices, all_accuracies, all_losses, all_training_times, cuda):
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

def generate_speech_metrics_table(metrics, output_dir):
    """Modified to handle empty metrics"""
    table_data = []
    for snr in sorted(metrics['no_aug'].keys(), reverse=True):
        no_aug_pesq = metrics['no_aug'][snr]['pesq']
        no_aug_stoi = metrics['no_aug'][snr]['stoi']
        aug_pesq = metrics['aug'][snr]['pesq']
        aug_stoi = metrics['aug'][snr]['stoi']
        
        row = {
            'SNR': snr,
            'No Aug PESQ': np.mean(no_aug_pesq) if no_aug_pesq else 'N/A',
            'No Aug STOI': np.mean(no_aug_stoi) if no_aug_stoi else 'N/A',
            'Aug PESQ': np.mean(aug_pesq) if aug_pesq else 'N/A',
            'Aug STOI': np.mean(aug_stoi) if aug_stoi else 'N/A'
        }
        table_data.append(row)
    
    # Add average row (only for non-empty values)
    avg_row = {
        'SNR': 'AVE',
        'No Aug PESQ': np.mean([np.mean(metrics['no_aug'][snr]['pesq']) 
                               for snr in metrics['no_aug'] 
                               if metrics['no_aug'][snr]['pesq']]) if any(metrics['no_aug'][snr]['pesq'] for snr in metrics['no_aug']) else 'N/A',
        'No Aug STOI': np.mean([np.mean(metrics['no_aug'][snr]['stoi']) 
                               for snr in metrics['no_aug'] 
                               if metrics['no_aug'][snr]['stoi']]) if any(metrics['no_aug'][snr]['stoi'] for snr in metrics['no_aug']) else 'N/A',
        'Aug PESQ': np.mean([np.mean(metrics['aug'][snr]['pesq']) 
                            for snr in metrics['aug'] 
                            if metrics['aug'][snr]['pesq']]) if any(metrics['aug'][snr]['pesq'] for snr in metrics['aug']) else 'N/A',
        'Aug STOI': np.mean([np.mean(metrics['aug'][snr]['stoi']) 
                            for snr in metrics['aug'] 
                            if metrics['aug'][snr]['stoi']]) if any(metrics['aug'][snr]['stoi'] for snr in metrics['aug']) else 'N/A'
    }
    table_data.append(avg_row)

    # Save to CSV and print formatted table
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(output_dir, 'speech_metrics.csv'), index=False)
    
    print("\nSpeech Quality Metrics Comparison")
    print("=" * 80)
    print(f"{'SNR':^6} | {'No Augmentation':^30} | {'Augmentation':^30}")
    print(f"{'':^6} | {'PESQ':^14} {'STOI':^14} | {'PESQ':^14} {'STOI':^14}")
    print("-" * 80)
    
    for row in table_data:
        pesq_no_aug = f"{row['No Aug PESQ']:.3f}" if isinstance(row['No Aug PESQ'], float) else row['No Aug PESQ']
        stoi_no_aug = f"{row['No Aug STOI']:.3f}" if isinstance(row['No Aug STOI'], float) else row['No Aug STOI']
        pesq_aug = f"{row['Aug PESQ']:.3f}" if isinstance(row['Aug PESQ'], float) else row['Aug PESQ']
        stoi_aug = f"{row['Aug STOI']:.3f}" if isinstance(row['Aug STOI'], float) else row['Aug STOI']
        
        print(f"{row['SNR']:^6} | {pesq_no_aug:^14} {stoi_no_aug:^14} | {pesq_aug:^14} {stoi_aug:^14}")

        
        
def calculate_metrics(clean_signal, noisy_signal, fs=16000):
    """Calculate PESQ and STOI scores"""
    try:
        pesq_score = pesq(fs, clean_signal, noisy_signal, 'wb')
        stoi_score = stoi(clean_signal, noisy_signal, fs, extended=False)
        return pesq_score, stoi_score
    except:
        return None, None

def generate_confusion_matrices(all_confusion_matrices, experiment_name, output_dir, dataset):
    """Modified to use dataset's label mapping instead of reading from file"""
    for model in MODELS.keys():
        if all_confusion_matrices[model]:
            std_conf_matrix = np.std(all_confusion_matrices[model], axis=0)
            
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

            int_to_label = dataset.int_to_label
            num_classes = len(int_to_label)

            plt.xticks(np.arange(num_classes) + 0.5, [int_to_label[i] for i in range(num_classes)], 
                    rotation=45,
                    ha='right',
                    rotation_mode='anchor')
            plt.yticks(np.arange(num_classes) + 0.5, [int_to_label[i] for i in range(num_classes)], rotation=0)
            plt.ylabel('True Scene')
            plt.title(f"Standard Deviation of Confusion Matrix - {model} [{experiment_name}]")
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.2)

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


def generate_classwise_accuracy(all_confusion_matrices, experiment_name, output_dir, dataset):
    """Modified to use dataset's label mapping instead of reading from file"""
    for model in MODELS.keys():
        if all_confusion_matrices[model]:
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
            
            # Use dataset's label mapping
            int_to_label = dataset.int_to_label
            
            plt.xticks(x, [int_to_label[i] for i in range(len(mean_accuracies))], rotation=45, ha='right')
            plt.title(f'Classwise Accuracy - {model} [{experiment_name}]')
            plt.xlabel('Class')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{model}_classwise_accuracy.png'))
            plt.close()


def generate_model_accuracies(all_accuracies, experiment_name, output_dir):
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

def generate_loss_plots(all_losses, experiment_name, output_dir):
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

def generate_training_times(all_training_times, experiment_name, output_dir):
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


    # # Initialize data structures to store results across all experiments
    # all_confusion_matrices = {model: [] for model in MODELS.keys()}
    # all_accuracies = {model: [] for model in MODELS.keys()}
    # all_losses = {model: [] for model in MODELS.keys()}
    # all_training_times = {model: [] for model in MODELS.keys()}


    # for i in range(1, number_of_experiments + 1):
    #     base_dir = f'models/{experiment_name}/{i}'
    #     training_data = [[] for _ in range(len(MODELS))]
    #     testing_data = [[] for _ in range(len(MODELS))]

        # for j, (model, (cnn1_channels, cnn2_channels, fc_neurons)) in enumerate(MODELS.items()):
    #         root_dir = os.path.join(base_dir, model)
    #         if os.path.exists(root_dir):
    #             # Get training and testing data
    #             train_data = os.path.join(root_dir, 'train_files.txt')
    #             test_data = os.path.join(root_dir, 'test_files.txt')
    #             if os.path.exists(train_data) and os.path.exists(test_data):
    #                 with open(train_data, 'r') as f:
    #                     training_data[j] = f.readlines()
    #                 with open(test_data, 'r') as f:
    #                     testing_data[j] = f.readlines()
        
    #     # Check if training and testing data are the same across models
    #     for j in range(len(MODELS) - 1):
    #         assert training_data[j] == training_data[j + 1], 'Training data is not the same across models'
    #         assert testing_data[j] == testing_data[j + 1], 'Testing data is not the same across models'

    # # Collect data from all experiments
    # for i in range(1, number_of_experiments + 1):
    #     base_dir = f'models/{experiment_name}/{i}'
        
    #     # Collect results for each model
    #     for model, (cnn1_channels, cnn2_channels, fc_neurons) in MODELS.items():
    #         root_dir = os.path.join(base_dir, model)
    #         if os.path.exists(root_dir):
    #             # Get confusion matrix and accuracy
    #             confusion_matrix, accuracy = test(dataset, root_dir, model, cnn1_channels, cnn2_channels, fc_neurons, cuda=cuda)
    #             all_confusion_matrices[model].append(confusion_matrix)
    #             all_accuracies[model].append(accuracy)
                
    #             # Get losses
    #             loss_file = os.path.join(root_dir, 'losses.txt')
    #             if os.path.exists(loss_file):
    #                 with open(loss_file, 'r') as f:
    #                     losses = [float(line.strip()) for line in f.readlines()]
    #                     all_losses[model].append(losses)
                
    #             # Get training time
    #             duration_file = os.path.join(root_dir, 'duration.txt')
    #             if os.path.exists(duration_file):
    #                 with open(duration_file, 'r') as f:
    #                     duration = float(f.readline().strip())
    #                     all_training_times[model].append(duration / 60)  # Convert to minutes

    # # Create output directory for combined results
    # output_dir = f'models/{experiment_name}/combined_results'
    # os.makedirs(output_dir, exist_ok=True)

    # # Standard deviation confusion matrix
    # for model in MODELS.keys():
    #     if all_confusion_matrices[model]:
    #         std_conf_matrix = np.std(all_confusion_matrices[model], axis=0)
            
    #         # Create figure with specific size ratio and extra space at bottom
    #         plt.figure(figsize=(12, 10))

    #         # Create the heatmap
    #         sns.heatmap(std_conf_matrix, 
    #                     annot=False,
    #                     cmap='YlOrBr_r',
    #                     vmin=0.0,
    #                     vmax=1.0,
    #                     square=False,
    #                     linewidths=0.5,
    #                     linecolor='black',
    #                     cbar_kws={
    #                         'label': '',
    #                         'orientation': 'horizontal',
    #                         'pad': 0.2,
    #                         'ticks': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    #                     })

    #         # Read int_to_label mapping from any experiment (they should be the same)
    #         base_dir = f'models/{experiment_name}/1/{model}'
    #         with open(os.path.join(base_dir, 'int_to_label.txt'), 'r') as f:
    #             int_to_label = dict(line.strip().split() for line in f)
    #             int_to_label = {int(k): v for k, v in int_to_label.items()}

    #         num_classes = len(int_to_label)

    #         # Add x-axis labels with better formatting
    #         plt.xticks(np.arange(num_classes) + 0.5, [int_to_label[i] for i in range(num_classes)], 
    #                 rotation=45,
    #                 ha='right',
    #                 rotation_mode='anchor')

    #         # Adjust y-axis labels
    #         plt.yticks(np.arange(num_classes) + 0.5, [int_to_label[i] for i in range(num_classes)], rotation=0)
    #         plt.ylabel('True Scene')

    #         # Set title
    #         plt.title(f"Standard Deviation of Confusion Matrix - {model} [{experiment_name}]")

    #         # Adjust layout with more bottom space
    #         plt.tight_layout()
    #         plt.subplots_adjust(bottom=0.2)

    #         # Move colorbar down
    #         ax = plt.gca()
    #         ax.set_xlabel('Estimated Scene')
    #         colorbar = ax.collections[0].colorbar
    #         colorbar.ax.set_position([colorbar.ax.get_position().x0, 
    #                                 colorbar.ax.get_position().y0 - 0.05,
    #                                 colorbar.ax.get_position().width,
    #                                 colorbar.ax.get_position().height])

    #         plt.savefig(os.path.join(output_dir, f'{model}_std_confusion_matrix.png'),
    #                 bbox_inches='tight',
    #                 dpi=300)
    #         plt.close()

    # # Classwise Accuracy
    # for model in MODELS.keys():
    #     if all_confusion_matrices[model]:
    #         # Calculate mean and std of class accuracies across experiments
    #         class_accuracies_per_exp = []
    #         for conf_matrix in all_confusion_matrices[model]:
    #             acc = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    #             class_accuracies_per_exp.append(acc)
            
    #         class_accuracies_array = np.array(class_accuracies_per_exp)
    #         mean_accuracies = np.mean(class_accuracies_array, axis=0)
    #         std_accuracies = np.std(class_accuracies_array, axis=0)
            
    #         plt.figure(figsize=(12, 6))
    #         x = range(len(mean_accuracies))
    #         plt.bar(x, mean_accuracies, yerr=std_accuracies, capsize=5)
            
    #         # Read class labels
    #         with open(os.path.join(base_dir, 'int_to_label.txt'), 'r') as f:
    #             int_to_label = dict(line.strip().split() for line in f)
    #             int_to_label = {int(k): v for k, v in int_to_label.items()}
            
    #         plt.xticks(x, [int_to_label[i] for i in range(len(mean_accuracies))], rotation=45, ha='right')
    #         plt.title(f'Classwise Accuracy - {model} [{experiment_name}]')
    #         plt.xlabel('Class')
    #         plt.ylabel('Accuracy')
    #         plt.ylim(0, 1)
    #         plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    #         plt.tight_layout()
    #         plt.savefig(os.path.join(output_dir, f'{model}_classwise_accuracy.png'))
    #         plt.close()


    # # Model Accuracies with Standard Deviation
    # plt.figure(figsize=(12, 6))
    # for model in MODELS.keys():
    #     if all_accuracies[model]:
    #         mean_acc = np.mean(all_accuracies[model])
    #         std_acc = np.std(all_accuracies[model])
    #         plt.errorbar(model, mean_acc, yerr=std_acc, fmt='o', capsize=5)

    # plt.title(f'Model Accuracies with Standard Deviation [{experiment_name}]')
    # plt.xlabel('Model')
    # plt.ylabel('Accuracy (%)')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, 'model_accuracies.png'))
    # plt.close()

    # # Loss against epochs
    # plt.figure(figsize=(12, 6))
    # for model in MODELS.keys():
    #     if all_losses[model]:
    #         losses_array = np.array(all_losses[model])
    #         mean_losses = np.mean(losses_array, axis=0)
    #         std_losses = np.std(losses_array, axis=0)
    #         epochs = range(1, len(mean_losses) + 1)
            
    #         plt.plot(epochs, mean_losses, label=model)
    #         plt.fill_between(epochs, 
    #                         mean_losses - std_losses,
    #                         mean_losses + std_losses,
    #                         alpha=0.2)

    # plt.title(f'Train Loss against Epochs [{experiment_name}]')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, 'training_losses.png'))
    # plt.close()

    # # Training Time Analysis
    # plt.figure(figsize=(10, 6))
    # for model in MODELS.keys():
    #     if all_training_times[model]:
    #         mean_time = np.mean(all_training_times[model])
    #         std_time = np.std(all_training_times[model])
    #         plt.bar(model, mean_time, yerr=std_time, capsize=5)

    # plt.title(f'Training Time Analysis [{experiment_name}]')
    # plt.xlabel('Model')
    # plt.ylabel('Time (minutes)')
    # plt.xticks(rotation=45)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, 'training_times.png'))
    # plt.close()

if __name__ == '__main__':
    cuda = False
    root_dir = '/Users/nkdem/Downloads/HEAR-DS' if not cuda else '/home/s2203859/minf-1/dataset/abc'
    dataset = HEARDS(root_dir=root_dir, cuda=cuda, augmentation=True)
    run_experiment(num_epochs=1, batch_size=16, number_of_experiments=1, learning_rates=[0.05], cuda=cuda)