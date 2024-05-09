import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datetime import datetime
import numpy as np
import sys
from torch.utils.tensorboard import SummaryWriter
import seaborn as sn
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pylab as plt

sys.path.insert(1,'../dataloader/')
sys.path.insert(1,'../dataset/')
sys.path.insert(1,'../models/')

# loading dataset and model from directory
from SimpleClassifier import MyClassifier
from lesions import get_dataloaders

# get dataset
qvalue_lower_bound = 0.7
qvalue_upper_bound = 1.46
classes = ['0','1']

# Perform training and validation by checking relative loss on a set of data that was not used for training, and report this loss
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                          # filename for tensorboard
PATH = '../runs/model_{}'.format(timestamp)             # path where model is saved

qgrid2 = np.hstack([np.arange(0.005, 0.0499, 0.001), np.arange(0.05, 0.099, 0.002), np.arange(0.1, 3.2, 0.005)])
lidx = np.argmin(qgrid2 < qvalue_lower_bound)   # qvalue = 0.7,  idx = 190
uidx = np.argmin(qgrid2 < qvalue_upper_bound)   # qvalue = 1.46, idx = 342
input_dim = (uidx - lidx)                              # (342-190) = 152



EPOCHS         = 10000      
learning_rates = [1e-4, 5e-5, 1e-4]
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(epoch, tb_writer):

    running_loss = 0.
    last_loss    = 0.
    epoch_loss   = 0.
    for batch_idx, inputs in enumerate(training_loader):

        X_batch = inputs[0]
        y_batch = inputs[1]
        X_batch = X_batch.view(-1,input_dim).to(device)
        y_batch = y_batch.to(device)

        ### Zero your gradients for every batch!
        optimizer.zero_grad()                          

        ### forward pass
        y_pred = model(X_batch)          # Make predictions for this batch
        loss   = loss_fn(y_pred, y_batch)  # Compute the loss 

        ### backward pass
        loss.backward()                  # Compute the gradients

        ### update weights
        optimizer.step()                 # Adjust learning weights

        # Gather data and report
        running_loss += loss.item()     # add loss for every batch

        # print(f'training accuracy on {batch_idx}: {(y_pred.round()==y_batch).float().mean()}')

        ### log data to tensorboard
        global_step = epoch*len(training_loader)+batch_idx+1
        writer.add_scalar('Training Loss', loss.item(), global_step = global_step)
        writer.add_scalar('Training Accuracy', (y_pred.round()==y_batch).float().mean() , global_step = global_step)
        # class_labels = [classes[label] for label in y_batch.numpy().ravel().astype(np.int32)]
        # writer.add_embedding(X_batch, metadata=class_labels, label_img = X_batch.unsqueeze(1).unsqueeze(1), global_step=global_step)
        # running_loss = 0.

        #### confusion matrix
        y_true = y_batch.to('cpu').detach().numpy()
        y_pred = y_pred.round().to('cpu').detach().numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp/(tp + fp)
        recall    = tp/(tp + fn)
        accuracy  = (tp + tn)/(tn + fp + fn + tp)
        f1        = 2*precision*recall/(precision + recall)
        df = pd.DataFrame(confusion_matrix(y_true, y_pred) , index = ['true_' + i for i in classes] , columns = ['pred_' + i for i in classes] )
        ax = plt.axes()
        ax.set_title('accuracy:{:0.3}, precision:{:0.3}, recall:{:0.3}, f1:{:0.3}'.format(accuracy, precision, recall, f1))
        writer.add_figure("confusion_matrix" , sn.heatmap(df, ax=ax, annot=True, fmt='g').get_figure(), global_step = global_step )


    return running_loss/(batch_idx+1)

# create dataloader with dataset
weights, training_loader, validation_loader = get_dataloaders(lidx=lidx, uidx=uidx)
weights = torch.from_numpy(weights).to(device)

for lr in learning_rates:

    writer = SummaryWriter('../runs/training_{}'.format(lr))               # tensorboard object creation
    # load model
    # input_dim=60   # change it accordingly         input_dim = 60                                  # brute force test case only
    model = MyClassifier(input_dim=input_dim).to(device)

    # loss and optimizer definition
    def BCELoss_class_weighted(weights):
        def loss(input, target):
            input = torch.clamp(input, min=1e-7, max=1-1e-7)
            bce = - (weights[1]*target*torch.log(input) + weights[0]*(1-target)*torch.log(1-input))
            return torch.mean(bce)
        return loss

                                      # total number of epochs
    loss_fn   = BCELoss_class_weighted(weights) # nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # optim.SGD(model.parameters(), lr=1e-3, momentum=0.8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=500, threshold=1e-4, min_lr=1e-6, verbose=True)


    best_vloss  = 1000     # init to None
    best_vacc   = -np.inf  # init to negative infinity
    for epoch in range(EPOCHS):
        print('Epoch {}/{}'.format(epoch, EPOCHS))

        model.train()                                                         # Make sure gradient tracking is on, and do a pass over the training data
        avg_loss = train_one_epoch(epoch, writer)  
        # scheduler.step(avg_loss)

        ### model evaluation
        model.eval()                                # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization. 
        X_val, y_val = next(iter(validation_loader))
        y_pred = model(X_val)
        avg_vloss = loss_fn(y_pred, y_val)

        avg_vacc = (y_pred.round()==y_val).float().mean()
        print('Loss - training: {} validation: {} Validation Accuracy: {}'.format(avg_loss, avg_vloss, avg_vacc))
        writer.add_scalars('Training vs Validation Loss', {'Training' : avg_loss, 'Validation' : avg_vloss}, epoch+1)      # Log the running loss averaged for both training and validation

        # Save model
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_vacc  = avg_vacc
            torch.save(model.state_dict(), PATH)

    writer.add_hparams({'lr': lr}, {'tr_loss': avg_loss, 'val_loss': best_vloss})


writer.flush()
print('Best Validation Accuracy : {}'.format(best_vacc))
# load a model
model = MyClassifier(input_dim=input_dim).to(device)
model.load_state_dict(torch.load(PATH))
