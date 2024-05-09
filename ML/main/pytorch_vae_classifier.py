import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import seaborn as sn
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pylab as plt


import sys, os, inspect, glob, h5py, json,copy
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir  = os.path.dirname(currentdir)
print(currentdir, parentdir)
sys.path.insert(0, os.path.join(parentdir, 'dataloader'))
sys.path.insert(0, os.path.join(parentdir, 'models'))

def h5File_h5Dir_csv_loc_by_h5file(file, BNL_dir, sub_dir):
    """ 
        h5File_h5Dir_csv_loc_by_h5file( file = "1948_HIPPO-roi1_0_0_masked_intp.h5",  
                                    BNL_dir     = '/Volumes/HDD/BNL-Data/Mar-2023' ,
                                    sub_dir     = "CSV_Conv-8-point")     
    """
    for file_search in glob.iglob(f'{BNL_dir}/**/*', recursive=True):
        if file_search.find(file) > -1 and file_search.endswith(".h5"):
            directory = os.path.dirname(file_search)
        if file_search.find(file) > -1 and f'/{sub_dir}/' in file_search:
            csv_file = file_search
            break
    return file, directory, csv_file


### snaking function
def snaking(Width, Height, X=None,):
    """
        img_orig = snaking(Width, Height, diff_patterns)
    """
    X = copy.deepcopy(X)    # deep copy
    if X is None:
        X_coor = np.arange(Height*Width).reshape(Height,Width)
    else:
        X_coor = X.reshape(Height,Width)            # input id numpy reshaped as height * width

    img_orig = np.flipud(X_coor)                                     # flipud matrix
    idx_even = np.arange(Height-2,-1,-2)                             # Get odd indices to leave to as it is
    img_orig[idx_even,:] = np.fliplr(img_orig[idx_even])             # flipping left to right 
    return img_orig

### snaking width_height extraction
## attrs in h5py is recorded in json format

def width_height(file, directory=None):
    """
        Width, Height = width_height(file)
    """
    directory = os.getcwd() if directory == None else directory
    with h5py.File(os.path.join(directory,file),'r') as hdf:
        dset = hdf.get(h5_top_group(file))
        header = json.loads(dset.attrs['start'])
        Height, Width = header['shape']
    return Width, Height

def h5_top_group(file):
    """ 
        input argument string of a h5py file name
        returns seperator
        seperator = h5_top_group(file)
    """
    return file.split('_masked')[0] if file.find('_masked') >= 0 else file.split('.')[0]

# assign snaking heapmap A to labels
def from_clusterFr_ceffs_to_matrix(A, cluster, coeffs):
    """
        cluster = [2066, 2067]
        coeffs  = [0.98, 0.99]
        A = np.array([np.zeros((Height,Width)),sna])                  # zero values matrix (A[0]=0) with frame numbers depth (A[1]=frames)
        
        funciton call: A = from_clusterFr_ceffs_to_matrix(A, cluster, coeffs)
    """
    for i, frame in enumerate(cluster):
        A[0][np.where(A[1]==frame)]=coeffs[i]

    B = A
    return B


### File Locations
column_names= {"Diffuse_Plaque":1. , "Tissue":0.}   # "Diffuse_Plaque"   "Neurofibrillary_Tangle_(tau)"    "Neuritic_Plaque"  "Tissue"
Excel_File  = "/Users/bashit.a/Documents/Alzheimer/Mar-2023/Mar-2023-Samples.xlsx"   # "/home/bashit.a/Codes/ML/Mar-2023-Samples.xlsx"   "/Users/bashit.a/Documents/Alzheimer/Mar-2023/Mar-2023-Samples.xlsx"
sheet       = 'Mar-2023-Samples'
BNL_dir     = '/Volumes/HDD/BNL-Data/Mar-2023'      # '/Volumes/HDD/BNL-Data/Mar-2023'         '/scratch/bashit.a/BNL-Data/Mar-2023'
sub_dir     = "CSV_Conv-8-point"  # CSV_Conv-8-point  CSV
val_files   = ["1948 V1-roi0_0_0_masked.h5"] # None ["1948_HIPPO-roi1_0_0_masked_intp.h5", "2428-roi1_0_0_masked_intp.h5"]


# loading dataset and model from directory
from SimpleClassifier import MyClassifier
from lesions import get_dataloaders_fixed_val_files
from VAE import VariationalAutoEncoder

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


EPOCHS         = 100      
learning_rates = [1e-4, 5e-5, 1e-4]
device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### VAE model 
hidden_dim=64
latent_dim=16

VAE_PATH = "/Users/bashit.a/Documents/Alzheimer/Codes/ML/runs/model_20231218_160842_38"
vae_model = VariationalAutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
vae_model.to(device)
vae_model.load_state_dict(torch.load(VAE_PATH, map_location=device))
vae_model.eval()

def train_one_epoch(epoch, tb_writer):

    running_loss = 0.
    last_loss    = 0.
    epoch_loss   = 0.
    for batch_idx, inputs in enumerate(training_loader):

        X_batch = inputs[0]
        y_batch = inputs[1]
        X_batch = X_batch.view(-1,input_dim)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        ### VAE model encoder
        _, mu, logvar = vae_model(X_batch)
        latent = vae_model.reparameterize(mu, logvar)

        ### Zero your gradients for every batch!
        optimizer.zero_grad()                          

        ### forward pass
        y_pred = model(latent)           # Make predictions for this batch    model(X_batch)   model(latent)
        loss   = loss_fn(y_pred, y_batch) # Compute the loss 

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
weights, training_loader, validation_loader = get_dataloaders_fixed_val_files(Excel_File, sheet, BNL_dir, sub_dir, column_names, val_files, lidx=lidx, uidx=uidx)
weights = torch.from_numpy(weights).to(device)

for lr in learning_rates:

    writer = SummaryWriter('../runs/training_{}'.format(lr))               # tensorboard object creation
    # load model
    # input_dim=60   # change it accordingly         input_dim = 60           input_dim=latent_dim for VAE                       # brute force test case only
    model = MyClassifier(input_dim=latent_dim)
    model.to(device)
    # print(f'model is in cuda ? {next(model.parameters()).is_cuda}')

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
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        # print(f'model is in cuda ? {next(model.parameters()).is_cuda}, inputs in cuda ? {X_val.is_cuda}')


        ### VAE model encoder
        _, mu, logvar = vae_model(X_val)
        latent = vae_model.reparameterize(mu, logvar)


        y_pred    = model(latent)
        avg_vloss = loss_fn(y_pred, y_val)

        avg_vacc = (y_pred.round()==y_val).float().mean()
        print('Loss - training: {} validation: {} Validation Accuracy: {}'.format(avg_loss, avg_vloss, avg_vacc))
        writer.add_scalars('Training vs Validation Loss', {'Training' : avg_loss, 'Validation' : avg_vloss}, epoch+1)      # Log the running loss averaged for both training and validation

        ax = plt.axes()
        file, directory, csv_file = h5File_h5Dir_csv_loc_by_h5file(val_files[0], BNL_dir, sub_dir)
        Width, Height = width_height(file, directory=directory)
        diff_patterns = y_pred.to('cpu').detach().numpy().round()    
        sna = snaking(Width, Height)  
        A = np.array([np.full((Height,Width), -1), sna]) 
        A = from_clusterFr_ceffs_to_matrix(A, validation_loader.dataset.frames, diff_patterns)
        img_orig = A[0]
        ax.set_title(f'Val-Accuracy:{avg_vacc}')
        writer.add_figure(f'Prediction:{file}', sn.heatmap(img_orig , ax=ax).get_figure(), global_step=epoch)

        # Save model
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_vacc  = avg_vacc
            torch.save(model.state_dict(), PATH)

    writer.add_hparams({'lr': lr}, {'tr_loss': avg_loss, 'val_loss': best_vloss})


writer.flush()
print('Best Validation Accuracy : {}'.format(best_vacc))
# load a model
model = MyClassifier(input_dim=latent_dim).to(device)
model.load_state_dict(torch.load(PATH))
