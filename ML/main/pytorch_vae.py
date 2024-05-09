import torch
import torchvision
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import sys, copy, os

### specs
sys.path.insert(0,'../dataloader/')
sys.path.insert(0,'../models/')
directory= "/Volumes/HDD/BNL-Data/Mar-2023/"
sub_dir     = "CSV_Conv-8-point"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                          # filename for tensorboard
PATH = '../runs/model_{}_0'.format(timestamp)                         # path where model is saved

# loading dataset and model from directory
from VAE import VariationalAutoEncoder
from bnl import get_dataloaders

# get dataset
qvalue_lower_bound = 0.7
qvalue_upper_bound = 1.46

qgrid2 = np.hstack([np.arange(0.005, 0.0499, 0.001), np.arange(0.05, 0.099, 0.002), np.arange(0.1, 3.2, 0.005)])
lidx = np.argmin(qgrid2 < qvalue_lower_bound)   # qvalue = 0.7,  idx = 190
uidx = np.argmin(qgrid2 < qvalue_upper_bound)   # qvalue = 1.46, idx = 342
input_dim = (uidx - lidx)                              # (342-190) = 152
print(f'input dimension: {input_dim}')

# create dataloader with dataset
training_loader, validation_loader = get_dataloaders(directory, sub_dir, lidx=lidx, uidx=uidx)


# load model
input_dim=input_dim   # change it accordingly
hidden_dim=64
latent_dim=16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VariationalAutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

# loss and optimizer definition
def loss_fn(inputs, x_reconstructed, mu, logvar):
    BCE = F.binary_cross_entropy(x_reconstructed, inputs, reduction='mean')
    # https://arxiv.org/abs/1312.6114
    #      0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KL  = -0.5 * torch.sum(1 + logvar - torch.square(mu) - torch.exp(logvar))
    return BCE + KL

optimizer = optim.Adam(model.parameters(), lr=1e-4, )  # optim.SGD(model.parameters(), lr=1e-3, momentum=0.8)


def train_one_epoch(epoch, tb_writer):

    running_loss = 0.
    last_loss    = 0.
    for batch_idx, inputs in enumerate(training_loader):

        inputs = inputs.view(-1,input_dim).to(device)

        optimizer.zero_grad()    # Zero your gradients for every batch!
        x_reconstructed, mu, logvar = model(inputs)    # Make predictions for this batch

        loss = loss_fn(inputs, x_reconstructed, mu, logvar)  # Compute the loss 
        loss.backward()                  # Compute the gradients

        optimizer.step()                 # Adjust learning weights

        # Gather data and report
        running_loss += loss.item()     # add loss for every batch
        if batch_idx%1000 == 999 :
            last_loss = running_loss/1000  # average for every 500 iterations
            print('  batch {} loss: {}'.format(batch_idx + 1, last_loss))
            tb_writer.add_scalar('Loss/train', last_loss, epoch*len(training_loader)+batch_idx+1)
            running_loss = 0.
    return running_loss/(batch_idx+1)

# Perform training and validation by checking relative loss on a set of data that was not used for training, and report this loss
tb_writer = SummaryWriter('../runs/training_{}'.format(timestamp))               # tensorboard object creation

EPOCHS      = 500                                                              # total number of epochs
best_vloss  = 10000. 



for epoch in range(EPOCHS):
    print('Epoch {}/{}'.format(epoch+1, EPOCHS))

    model.train()                                                         # Make sure gradient tracking is on, and do a pass over the training data
    avg_loss = train_one_epoch(epoch, tb_writer)                              

    model.eval()                                # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization.
    running_vloss = 0.
    with torch.no_grad() :                      # Disable gradient computation and reduce memory consumption.
        for i, inputs in enumerate(validation_loader):
            inputs = inputs.view(-1,input_dim).to(device)
            x_reconstructed, mu, logvar = model(inputs)
            loss = loss_fn(inputs, x_reconstructed, mu, logvar)
            running_vloss += loss.item()

    avg_vloss = running_vloss/(i+1)
    print('Loss - training: {} validation: {}'.format(avg_loss, avg_vloss))
    tb_writer.add_scalars('Training vs Validation Loss', {'Training' : avg_loss, 'Validation' : avg_vloss}, epoch+1)      # Log the running loss averaged for both training and validation
    tb_writer.flush()

    # Save model
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        best_model_state = copy.deepcopy( model.state_dict() )

        try:
            os.remove(PATH)
        except:
            pass
        PATH = PATH[:PATH.rfind('_')] + f'_{epoch+1}'
        print('writing model to : {}'.format(PATH))
        torch.save(best_model_state, PATH)

# load a model
model = VariationalAutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load(PATH, map_location=device))
