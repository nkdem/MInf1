import torch
import sys
import torch.nn as nn
from tqdm import tqdm 
sys.path.append('/workspace')
from voicebank_dataset import get_loaders
import logging
import torch
from models import CRNN
import numpy as np
lr = 0.0006
beta1 = 0.9
beta2 = 0.999

epochs = 70

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cuda = torch.cuda.is_available()
batch_size = 32 if cuda else 4

train_loader, test_loader = get_loaders(cuda=cuda, batch_size=batch_size)
for batch in train_loader:
    noisy, clean, basename = batch
    print(noisy[0].shape)
    print(clean[0].shape)
    print(basename)
    break

model = CRNN()
if cuda:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

criterion = nn.MSELoss()
losses = []
# early stopping
best_loss = float('inf')   
patience = 5
patience_counter = 0

for epoch in range(epochs):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", total=len(train_loader))
    avg_loss = 0
    for i, batch in enumerate(pbar):
        noisy, clean, basename = batch
        noisy_mag, clean_mag = noisy[0], clean[0]
        noisy_mag = noisy_mag.cuda() # [T, F]
        clean_mag = clean_mag.cuda() # [T, F]

        optimizer.zero_grad()
        output = model(noisy_mag) # [T, F]
        loss = criterion(output, clean_mag)
        loss.backward()

        optimizer.step()
        avg_loss += loss.item()
        pbar.set_postfix({
            'avg_loss': avg_loss / (i + 1)
        })

    # every 10 epochs, save the model
    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"model_{epoch}.pth")

    avg_loss /= len(train_loader)
    losses.append(avg_loss)
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# save model
torch.save(model.state_dict(), f"model_{epoch}.pth")

