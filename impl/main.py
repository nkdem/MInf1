import os

from constants import MODELS
from train import train
from test import test


for i in range(1,10):
    base_dir = f'models/{i}'
    os.makedirs(base_dir, exist_ok=True)
    train(base_dir=base_dir)

# test
# for i in range(1,10):
#     base_dir = f'models/{i}'
#     for model, (cnn1_channels, cnn2_channels, fc_neurons) in MODELS.items():
#         root_dir = os.path.join(base_dir, model)
#         if os.path.exists(root_dir):
#             test(root_dir, model, cnn1_channels, cnn2_channels, fc_neurons)