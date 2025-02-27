import collections
import gc
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm
from base_experiment import BaseExperiment
from helpers import compute_average_logmel
from models import AudioCNN
from classification.train import AdamEarlyStopTrainer
from constants import MODELS
from abc import ABC, abstractmethod

from classification.TUT.tut_dataset import get_datasets_for_fold, get_folds, TUTDataset

def collect_model_results(test_loader, model, no_classes, env_to_int):
    model.eval()
    total = 0
    correct = 0
    confusion_matrix = np.zeros((no_classes, no_classes))
    model.device = torch.device("mps")
    print(model.device)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            pair, envs, base = batch
            logmels = compute_average_logmel([(pair[0][i], pair[1][i]) for i in range(len(pair[0]))], model.device)
            labels = torch.tensor([env_to_int[env] for env in envs], dtype=torch.long).to(model.device)
            outputs = model(logmels)
            _, predicted = torch.max(outputs.data, 1)
            labels = labels.cpu()
            total += len(predicted)
            correct += (predicted == labels).sum().item()
            for i in range(len(predicted)):
                confusion_matrix[labels[i], predicted[i]] += 1
    classwise_accuracy = confusion_matrix.diagonal() / confusion_matrix.sum(axis=1)
    total_accuracy = correct / total
    return classwise_accuracy, total_accuracy, confusion_matrix
if __name__ == '__main__':
    root_dir = '/Users/nkdem/Downloads/TUT-acoustic-scenes-2017-evaluation'
    tut_dir = '/Users/nkdem/Downloads/TUT-acoustic-scenes-2017-evaluation-16k'
    
    # TUT-acoustic-scenes-2017-evaluation contains a folder called evaluation_setup and a file called evaluate.txt
    # which contains a mapping of the audio files to the labels
    audio_files = []
    with open(os.path.join(root_dir, 'evaluation_setup', 'evaluate.txt'), 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            # strip the audio/ prefix
            file_name = parts[0].split('/')[1]
            audio_files.append(os.path.join(tut_dir, file_name))
    dataset = TUTDataset(root_dir, audio_files=audio_files)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # from models/full_adam_240epochs_32batchsize_TUT/exp1/fold1  and each model folder contains a model.pth file
    # read the model and evaluate it on the test_loader
    base_dir = 'models/full_adam_240epochs_32batchsize_TUT/exp1/fold1'
    for model in MODELS.keys():
        cnn1_channels, cnn2_channels, fc_neurons = MODELS[model]
        cnn = AudioCNN(15, cnn1_channels, cnn2_channels, fc_neurons)
        model_path = os.path.join(base_dir, model, 'model.pth')
        cnn.load_state_dict(torch.load(model_path, weights_only=True))
        # env_to_int can be created by reading the int_to_label.txt file in fold1
        env_to_int = {env: i for i, env in enumerate(dataset.labels)}
        with open(os.path.join(base_dir, 'int_to_label.txt'), 'r') as f:
            for i, line in enumerate(f):
                env, value = line.strip().split(' ')
                env_to_int[env] = int(value)
        classwise_accuracy, total_accuracy, confusion_matrix = collect_model_results(test_loader=test_loader, model=cnn, no_classes=15, env_to_int=env_to_int)
        print(f"Model: {model}")
        print(f"Classwise accuracy: {classwise_accuracy}")
        print(f"Total accuracy: {total_accuracy}")
        print(f"Confusion matrix: {confusion_matrix}")
        print("\n")