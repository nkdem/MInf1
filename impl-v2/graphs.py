import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_results(base_dir):
    results = {}
    for file in os.listdir(base_dir):
        if file.startswith('results-fold') and file.endswith('.pkl'):
            fold = file.split('-')[1].split('.')[0]
            file_path = os.path.join(base_dir, file)
            with open(file_path, 'rb') as f:
                results[fold] = pickle.load(f)
    return results

# def plot_classwise_accuracy(results, base_dir):
#     models = list(results['fold1']['class_accuracies'].keys())
#     folds = sorted(results.keys())
    
#     for fold in folds:
#         fig, ax = plt.subplots(figsize=(12, 8))
#         width = 0.8 / len(models)
#         x = np.arange(len(models))
        
#         for i, model in enumerate(models):
#             accuracies = results[fold]['class_accuracies'][model]
#             ax.bar(x + i*width, accuracies, width, label=model)
        
#         ax.set_xlabel('Class')
#         ax.set_ylabel('Accuracy')
#         ax.set_title(f'Classwise Accuracy for Fold {fold}')
#         ax.set_xticks(x + width * (len(models) - 1) / 2)
#         ax.set_xticklabels(range(len(accuracies)))
#         ax.legend()
        
#         plt.savefig(os.path.join(base_dir, f'classwise_accuracy_fold{fold}.png'))
#         plt.close()

def plot_overall_accuracy(results, base_dir):
    models = list(results['fold1']['total_accuracies'].keys())
    folds = sorted(results.keys())
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model in models:
        accuracies = [results[fold]['total_accuracies'][model] for fold in folds]
        ax.plot(folds, accuracies, marker='o', label=model)
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('Overall Accuracy')
    ax.set_title('Overall Accuracy Across All Folds')
    ax.legend()
    
    plt.savefig(os.path.join(base_dir, 'overall_accuracy_all_folds.png'))
    plt.close()

def plot_loss(results, base_dir):
    models = list(results['fold1']['losses'].keys())
    folds = sorted(results.keys())
    
    for fold in folds:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for model in models:
            losses = results[fold]['losses'][model]
            ax.plot(range(len(losses)), losses, marker='o', label=model)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss for Fold {fold}')
        ax.legend()
        
        plt.savefig(os.path.join(base_dir, f'loss_fold{fold}.png'))
        plt.close()

def create_graphs(base_dir):
    results = load_results(base_dir)
    
    if results:
        # plot_classwise_accuracy(results, base_dir)
        plot_overall_accuracy(results, base_dir)
        plot_loss(results, base_dir)
        print(f"Graphs saved in {base_dir}")
    else:
        print("No results found in the specified directory.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Create graphs from TUTBaselineExperiment results")
    parser.add_argument("--base_dir", type=str, required=False, help="Base directory containing fold results")
    args = parser.parse_args()
    if args.base_dir is None:
        print("No base directory provided.")
        base_dir = 'models/full_adam_240epochs_32batchsize_TUT/exp1'
    else:
        base_dir = args.base_dir
    
    create_graphs(base_dir)
