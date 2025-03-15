import os
from constants import MODELS
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_results(base_dir, experiment_no):
    results_path = os.path.join(base_dir, str(experiment_no), "results.pkl")
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    return results

def plot_confusion_matrix(conf_matrix, labels, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_training_loss(losses, title, save_path):
    plt.figure()
    for model, loss in losses.items():
        print(f"Model: {model}, Loss Data: {loss}")

        # Flatten the loss data if necessary
        if isinstance(loss, list) and len(loss) > 0 and isinstance(loss[0], list):
            loss = loss[0]

        # Check if loss is iterable and contains valid data
        if isinstance(loss, (list, np.ndarray)) and len(loss) > 0 and not np.isnan(loss).any():
            plt.plot(loss, label=model)
        else:
            print(f"Warning: Loss data for model {model} is not in the expected format or is empty.")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_classwise_accuracy(class_accuracies, labels, title, save_path):
    plt.figure(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.2  # Width of each bar

    for idx, (model, accuracies) in enumerate(class_accuracies.items()):
        print(f"Model: {model}, Accuracies Type: {type(accuracies)}, Accuracies Data: {accuracies}")

        # Ensure accuracies is iterable and contains valid data
        if isinstance(accuracies, (list, np.ndarray)) and len(accuracies) > 0:
            # Flatten the accuracies if they are stored as lists of arrays
            if isinstance(accuracies[0], np.ndarray):
                accuracies_array = np.array(accuracies)
                mean_accuracies = np.nanmean(accuracies_array, axis=0)
            else:
                mean_accuracies = np.nanmean(accuracies, axis=0)

            # Plot each model's classwise accuracy in a separate position
            plt.bar(x + idx * width, mean_accuracies, width, label=model)
        else:
            print(f"Warning: Accuracies for model {model} are not in the expected format or are empty.")

    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks(x + width * (len(class_accuracies) - 1) / 2, labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    base_dir = "models/full_adam_2epochs_32batchsize"
    num_experiments = 2
    int_to_label_path = os.path.join(base_dir, "int_to_label.txt")
    labels = ['SpeechIn_QuietIndoors', 'InterfereringSpeakers', 'Music', 'WindTurbulence', 'SpeechIn_InTraffic', 'SpeechIn_WindTurbulence', 'SpeechIn_ReverberantEnvironment', 'InTraffic', 'SpeechIn_Music', 'QuietIndoors', 'ReverberantEnvironment', 'CocktailParty', 'SpeechIn_InVehicle', 'InVehicle']


    # Aggregate results over experiments
    aggregated_conf_matrices = {model: np.zeros((len(labels), len(labels))) for model in MODELS}
    aggregated_losses = {model: [] for model in MODELS}
    aggregated_class_accuracies = {model: [] for model in MODELS}

    for experiment_no in range(1, num_experiments + 1):
        results = load_results(base_dir, experiment_no)

        for model in results['confusion_matrix_raw']:
            aggregated_conf_matrices[model] += np.sum(results['confusion_matrix_raw'][model], axis=0)
            aggregated_losses[model].append(results['losses'][model])
            aggregated_class_accuracies[model].append(results['class_accuracies'][model])

            # Individual experiment plots
            plot_confusion_matrix(
                np.sum(results['confusion_matrix_raw'][model], axis=0),
                labels,
                f'Confusion Matrix - {model} - Experiment {experiment_no}',
                f'{base_dir}/confusion_matrix_{model}_exp_{experiment_no}.png'
            )
            plot_training_loss(
                results['losses'],
                f'Training Loss - {model} - Experiment {experiment_no}',
                f'{base_dir}/training_loss_{model}_exp_{experiment_no}.png'
            )
            plot_classwise_accuracy(
                results['class_accuracies'],
                labels,
                f'Classwise Accuracy - {model} - Experiment {experiment_no}',
                f'{base_dir}/classwise_accuracy_{model}_exp_{experiment_no}.png'
            )

    # Average plots
    for model in aggregated_conf_matrices:
        plot_confusion_matrix(
            aggregated_conf_matrices[model] / num_experiments,
            labels,
            f'Average Confusion Matrix - {model}',
            f'{base_dir}/average_confusion_matrix_{model}.png'
        )

    plot_training_loss(
        {model: np.mean(losses, axis=0) for model, losses in aggregated_losses.items()},
        'Average Training Loss',
        f'{base_dir}/average_training_loss.png'
    )

    plot_classwise_accuracy(
        {model: np.mean(accuracies, axis=0) for model, accuracies in aggregated_class_accuracies.items()},
        labels,
        'Average Classwise Accuracy',
        f'{base_dir}/average_classwise_accuracy.png'
    )

if __name__ == "__main__":
    main()
