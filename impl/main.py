import os

from constants import MODELS
from hear_ds import HEARDS
from train import train
from test import test
train_dataset = HEARDS('/Users/nkdem/Downloads/HEAR-DS')


for i in range(1,3):
    base_dir = f'models/210_epochs_256_batch/{i}'
    os.makedirs(base_dir, exist_ok=True)
    train(base_dir=base_dir, num_epochs=210, batch_size=256)
    # test
    for model, (cnn1_channels, cnn2_channels, fc_neurons) in MODELS.items():
        root_dir = os.path.join(base_dir, model)
        if os.path.exists(root_dir):
            test(train_dataset, root_dir, model, cnn1_channels, cnn2_channels, fc_neurons)

# test
# for i in range(1,10):
# base_dir = f'models/280_epochs_64_batch/'
# for model, (cnn1_channels, cnn2_channels, fc_neurons) in MODELS.items():
#     root_dir = os.path.join(base_dir, model)
#     if os.path.exists(root_dir):
#         test(dataset, root_dir, model, cnn1_channels, cnn2_channels, fc_neurons)