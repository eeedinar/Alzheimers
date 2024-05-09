import torch
import torchvision
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np


import sys, os, inspect, glob, h5py, json,copy
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir  = os.path.dirname(currentdir)
print(currentdir, parentdir)
sys.path.insert(0, os.path.join(parentdir, 'dataloader'))
sys.path.insert(0, os.path.join(parentdir, 'models'))

### File Locations
BNL_dir     = '/Volumes/HDD/BNL-Data/Mar-2023'      # '/Volumes/HDD/BNL-Data/Mar-2023'         '/scratch/bashit.a/BNL-Data/Mar-2023'
sub_dir     = "CSV_Conv-8-point"  # CSV_Conv-8-point  CSV


# loading dataset and model from directory
from AE import AutoEncoder
from bnl import get_dataloaders

# get dataset
qvalue_lower_bound = 0.1
qvalue_upper_bound = 1.46

qgrid2 = np.hstack([np.arange(0.005, 0.0499, 0.001), np.arange(0.05, 0.099, 0.002), np.arange(0.1, 3.2, 0.005)])
lidx = np.argmin(qgrid2 < qvalue_lower_bound)   # qvalue = 0.7,  idx = 190
uidx = np.argmin(qgrid2 < qvalue_upper_bound)   # qvalue = 1.46, idx = 342
input_dim = (uidx - lidx)                              # (342-190) = 152

# create dataloader with dataset
training_loader, validation_loader = get_dataloaders(BNL_dir, sub_dir, lidx=lidx, uidx=uidx)


# load model
input_dim=input_dim   # change it accordingly
latent_dim=3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)

# loss and optimizer definition
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2, )  # optim.SGD(model.parameters(), lr=1e-3, momentum=0.8)


def train_one_epoch(epoch, tb_writer):

    running_loss = 0.
    last_loss    = 0.
    for batch_idx, inputs in enumerate(training_loader):

        inputs = inputs.view(-1,input_dim).to(device)

        optimizer.zero_grad()                          # Zero your gradients for every batch!
        x_reconstructed = model(inputs)    # Make predictions for this batch

        loss = loss_fn(inputs, x_reconstructed)  # Compute the loss 
        loss.backward()                  # Compute the gradients

        optimizer.step()                 # Adjust learning weights

        # Gather data and report
        running_loss += loss.item()     # add loss for every batch
        if batch_idx%1000 == 999 :
            last_loss = running_loss/1000  # average for every 500 iterations
            print('  batch {} loss: {}'.format(batch_idx + 1, last_loss))
            tb_writer.add_scalar('Loss/train', last_loss, epoch*len(training_loader)+batch_idx+1)
            running_loss = 0.
    return last_loss

# Perform training and validation by checking relative loss on a set of data that was not used for training, and report this loss
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                          # filename for tensorboard
tb_writer = SummaryWriter('../runs/training_{}'.format(timestamp))               # tensorboard object creation

EPOCHS      = 50                                                              # total number of epochs
best_vloss  = 10000. 
for epoch in range(EPOCHS):
    print('Epoch {}/{}'.format(epoch, EPOCHS))

    model.train()                                                         # Make sure gradient tracking is on, and do a pass over the training data
    avg_loss = train_one_epoch(epoch, tb_writer)                              

    model.eval()                                # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
    running_vloss = 0.
    with torch.no_grad() :                      # Disable gradient computation and reduce memory consumption.
        for i, inputs in enumerate(validation_loader):
            inputs = inputs.view(-1,input_dim).to(device)
            x_reconstructed= model(inputs)
            loss = loss_fn(inputs, x_reconstructed)
            running_vloss += loss.item()

    avg_vloss = running_vloss/(i+1)
    print('Loss - training: {} validation: {}'.format(avg_loss, avg_vloss))
    tb_writer.add_scalars('Training vs Validation Loss', {'Training' : avg_loss, 'Validation' : avg_vloss}, epoch+1)      # Log the running loss averaged for both training and validation
    tb_writer.flush()

    # Save model
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = '../runs/model_{}_{}'.format(timestamp, epoch)
        torch.save(model.state_dict(), model_path)

# load a model
model = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
PATH = '../runs/model_{}_{}'.format(timestamp, epoch)  # path where model is saved
model.load_state_dict(torch.load(PATH))

