import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import seaborn as sn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_recall_fscore_support
import pandas as pd
import matplotlib.pylab as plt
import sys, os, inspect, glob, h5py, json,copy, yaml
import wandb
import random

### ensure deterministic behavior
set_seed = 40
torch.backends.cudnn.deterministic = True
random.seed(set_seed)
np.random.seed(set_seed)
torch.manual_seed(set_seed)
torch.cuda.manual_seed_all(set_seed)

### set python path
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir  = os.path.dirname(currentdir)
print(currentdir, parentdir)
sys.path.insert(0, os.path.join(parentdir, 'dataloader'))
sys.path.insert(0, os.path.join(parentdir, 'models'))

### loading dataset and model from directory
from utils        import *
from train_builds import *

device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.login()     # wandb

### file Locations
with  open('sweep_configuration.yaml') as file_sweep :    
    config = yaml.load(file_sweep, Loader=yaml.FullLoader)

yaml_file      = "/Users/bashit.a/Documents/Alzheimer/Codes/ML/dataloader/lesions.yaml"
yaml_model     = "/Users/bashit.a/Documents/Alzheimer/Codes/ML/models/transformer.yaml"
sonar_file     = "/Users/bashit.a/Documents/Alzheimer/Codes/ML/dataloader/sonar.csv"
dataset_source = "mar-2023"                # "sonar"    "mar-2023"
network        = "transformer"

# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')                          # filename for tensorboard
PATH = os.path.join(parentdir, 'runs/model')            # path where model is saved

### load yaml files
if dataset_source == "mar-2023":
    with open(yaml_file) as file:
        config_dataset = yaml.load(file, Loader=yaml.FullLoader)
    config["parameters"].update(config_dataset["parameters"])

if network == 'transformer':
    with open(yaml_model) as file:
        config_model = yaml.load(file, Loader=yaml.FullLoader)
    config["parameters"].update(config_model["parameters"])

# print(json.dumps(config,indent=2))

def train_epoch(epoch, input_dim, model, training_loader, optimizer, criterion, writer):

    running_loss = 0.
    for batch_idx, inputs in enumerate(training_loader):

        ### extract inputs from batches
        # X_batch = inputs[0].view(-1,input_dim)
        X_batch = inputs[0].to(device)
        y_batch = inputs[1].to(device)
        
        # anchor   = model(X_batch[:,             : input_dim]  , None)
        # positive = model(X_batch[:, input_dim   : 2*input_dim], None)
        # negative = model(X_batch[:, 2*input_dim :]            , None)

        ### Zero your gradients for every batch!
        optimizer.zero_grad()
        y_pred = model(X_batch, None)                             # forward pass- make predictions for this batch    model(X_batch)   model(latent)
        loss   = criterion(y_pred, torch.flatten(y_batch).long()) # Compute the loss
        # loss   = criterion(anchor, positive, negative)

        loss.backward()                  # backward pass - compute the gradients
        # # gradient clipping
        # clip_value = 10.0
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
        optimizer.step()                 # update weights - Adjust learning weights
        
        running_loss += loss.item()      # add loss for every batch

        ### accuracy
        y_pred    = torch.argmax(y_pred,1).type(torch.float32)
        avg_acc = performace_metrics(y_pred, y_batch)

        # wandb.log( {"loss_func_max": criterion.input.max(), "loss_func_min": criterion.input.min()} )
        
        ### log data to tensorboard
        # global_step = epoch*len(training_loader)+batch_idx+1
        # writer.add_scalar('train_loss', loss.item(), global_step = global_step)

        #### confusion matrix
        # y_true = y_batch.to('cpu').detach().numpy()
        # y_true = y_true.ravel()
        # y_pred = y_pred.round().to('cpu').detach().numpy()
        # y_pred = np.argmax(y_pred,1)

        # cm = confusion_matrix(y_true, y_pred).T
        # df_cm = pd.DataFrame(cm,   )  # index = list(column_names.keys()), columns =list(column_names.keys())
        # print(epoch, df_cm)

        # ax = plt.axes()
        # bal_accuracy = balanced_accuracy_score(y_true, y_pred)
        # precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
        # ax.set_title('bal_acc:{:0.1}, precision:{:0.1}, recall:{:0.1}, f1:{:0.1}'.format(bal_accuracy, precision, recall, f1))
        # writer.add_figure("confusion_matrix" , sn.heatmap(df_cm, ax=ax, annot=True, fmt='g').get_figure(), global_step = global_step )

    return running_loss/(batch_idx+1), avg_acc

def log_prediction(input_data, labels, outputs, predicted, log_table, log_counter):
    ### columns = ["id", "input_data", "guess", "truth"]
    log_scores = F.softmax(outputs, dim=1)
    log_scores = log_scores.to('cpu').detach().numpy()
    log_input_data = input_data.to('cpu').detach().numpy()
    log_labels     = labels.to('cpu').detach().numpy()
    log_predicted  = predicted.to('cpu').detach().numpy()

    _id=0
    for i, p, l, s in zip(log_input_data, log_predicted, log_labels, log_scores):
        
        log_table.add_data(str(_id), i.tolist(),p,l,*s)
        _id +=1

        if _id == 100:
            break

    
def train():

    with wandb.init():

        config    = wandb.config  #If called by wandb.agent, this config will be set by Sweep Controller
        epochs    = config.epochs
        lr        = config.lr
        optimizer = config.optimizer
        momentum  = config.momentum
        loss_fn   = config.loss_fn

        input_dim, output_dim, weights, training_loader, validation_loader = build_dataset(dataset_source, config, device, sonar_file=sonar_file)
        model     = build_model(network, yaml_model, config, device, input_dim, output_dim)
        optimizer = build_optimizer(optimizer, model, lr, momentum)
        es = EarlyStopping()
        criterion = build_loss(loss_fn, weights)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=200, threshold=1e-4, min_lr=1e-6, verbose=True)

        writer = SummaryWriter('../runs/training_{}'.format(lr))               # tensorboard object creation
        wandb.watch(model, optimizer, log="all", log_freq=10)

        best_loss  = 1000     # init to None
        best_acc   = -np.inf  # init to negative infinity
        
        epoch = 0
        done = False
        while epoch < epochs and not done:
            epoch +=1
            model.train()                                                         # Make sure gradient tracking is on, and do a pass over the training data
            avg_loss, avg_acc = train_epoch(epoch, input_dim, model, training_loader, optimizer, criterion, writer)  


            ### model evaluation
            model.eval()                                # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization. 
            with torch.no_grad():
                X_val, y_val = next(iter(validation_loader))
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                # print(f'model is in cuda ? {next(model.parameters()).is_cuda}, inputs in cuda ? {X_val.is_cuda}')

                out   = model(X_val,None)
                avg_vloss = criterion(out, torch.flatten(y_val).long())

                # anchor   = model(X_val[:,:input_dim]           , None)
                # positive = model(X_val[:,input_dim:2*input_dim], None)
                # negative = model(X_val[:,2*input_dim:]         , None)
                # avg_vloss   = criterion(anchor, positive, negative)

                # Note that step should be called after validate()
                scheduler.step(avg_vloss)  

                y_pred    = torch.argmax(out,1).type(torch.float32)
                avg_vacc = performace_metrics(y_pred, y_val)
            
            if es(model, avg_vloss.item(), epoch):
                done = True
                destination= PATH+'_'+f'{es.best_epoch}'+'_BEST_VLoss_'+f'{es.best_vloss:0.2f}'
                torch.save(es.best_model, destination)

            ### log data to weights and biases
            wandb.log( {"train_loss": avg_loss, "val_loss": avg_vloss.item() } )
            # metrics = {} # "train/epoch": global_step,
            # wandb.log(metrics)

            print(f'Epoch {epoch}/{epochs} : Loss - training: {avg_loss:0.6f} validation: {avg_vloss:0.6f} Validation Accuracy: {avg_vacc:0.4f} ES: {es.status}')
            writer.add_scalars('Training vs Validation Loss', {'Training' : avg_loss, 'Validation' : avg_vloss}, epoch+1)      # Log the running loss averaged for both training and validation            
            


            # ax = plt.axes()
            # file, directory, csv_file = h5File_h5Dir_csv_loc_by_h5file(val_files[0], BNL_dir, sub_dir)
            # Width, Height = width_height(file, directory=directory)
            # diff_patterns = y_pred.to('cpu').detach().numpy().round()    
            # sna = snaking(Width, Height)  
            # A = np.array([np.full((Height,Width), -1), sna]) 
            # A = from_clusterFr_ceffs_to_matrix(A, validation_loader.dataset.frames, diff_patterns)
            # img_orig = A[0]
            # ax.set_title(f'Val-Accuracy:{avg_vacc}')
            # writer.add_figure(f'Prediction:{file}', sn.heatmap(img_orig , ax=ax).get_figure(), global_step=epoch)

            # Save model
            if avg_loss < best_loss:   # avg_vacc > best_vacc     
                best_loss = avg_loss
                best_acc  = avg_acc

                # try: os.remove(destination)                    
                # except: pass
                
                destination= PATH+'_'+f'{epoch}'+'_'+f'{avg_loss:0.2f}'+'_'+'_'+f'{avg_acc:0.2f}'
                torch.save(model.state_dict(), destination)


            # log_table = wandb.Table(columns = ["id", "input_data", "guess", "truth", "Neurofibrillary_Tangle_(tau)", "Diffuse_Plaque", "Tissue"])
            # log_prediction(X_val, y_val, out, y_pred, log_table, log_counter=epoch)
            # wandb.log({"test_prediction":log_table})

        writer.add_hparams({'lr': config.lr}, {'tr_loss': best_loss, 'tr_acc': best_acc})

# Start sweep job
sweep_id = wandb.sweep(sweep=config, project="transformer-classifier")
wandb.agent(sweep_id, function=train, count=1)

# PATH_ITR = "0"

# # Mark the run as finished
# writer.flush()
# wandb.finish()

# print('Best Validation Accuracy : {}'.format(best_vacc))
# # load a model
# model = Transformer(config["n_heads"], config["seq_len"], config["embedding_dim"], config["hidden_dim"], config["N"], config["dropout"], n_classes)
# model.to(device)
# model.load_state_dict(torch.load(PATH))
