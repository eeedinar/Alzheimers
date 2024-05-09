import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import numpy as np
import os, glob


def interpolate_missing(A):
    # indx, A = interpolate_missing(Iq_M_WAXS[1])

    # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    ok = ~ np.isnan(A)
    xp = ok.ravel().nonzero()[0]
    fp = A[~ np.isnan(A)]
    x  = np.isnan(A).ravel().nonzero()[0]
    A[np.isnan(A)] = np.interp(x, xp, fp)

    return x, A


class XrayData(Dataset):

    def __init__(self, directory, sub_dir, lidx=0, uidx=690):
        # load data
        files = []
        for file in glob.iglob(os.path.join(directory,'**/*.csv'), recursive=True):  # '**/*.csv'
            if sub_dir in file:
                files.append(np.loadtxt(file, delimiter=",", dtype=np.float32, skiprows=1))

        x            = np.vstack([np.array(files[i]) for i in range(len(files))])
        _, x         = interpolate_missing(x)
        x = x/np.max(x,axis=1).reshape(-1,1) 
        self.dataset = x[:,lidx:uidx]


    def __getitem__(self, index):
        # dataset[0]
        return self.dataset[index]

    def __len__(self):
        # len(dataset)
        return self.dataset.shape[0]


def get_dataloaders(directory, sub_dir, lidx=0, uidx=690):

    batch_size  = 200000
    train_split = .8
    shuffle_dataset = True
    random_seed= 42

    dataset = XrayData(directory, sub_dir, lidx=lidx, uidx=uidx)
    
    train_size = int(0.8*len(dataset))
    val_size   = len(dataset) - train_size

    print(f'train size: {train_size} test size : {val_size}')

    # Creating data indices for training and validation splits:
    indices = np.arange(0,len(dataset))
    split   = int(train_split*len(dataset))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    training_loader   = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    validation_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0)
    print("training_loader size - {} ; validation_loader size {}".format(len(training_loader), len(validation_loader)))

    return training_loader, validation_loader


if __name__ == '__main__':

    directory   = "/Volumes/HDD/BNL-Data/Mar-2023/"
    sub_dir     = "CSV_Conv-8-point"

    training_loader, validation_loader = get_dataloaders(directory, sub_dir, lidx=190, uidx=352)
    train_dataiter = iter(training_loader)
    train_data = next(train_dataiter)
    print(train_data.shape, len(training_loader))

    val_dataiter = iter(validation_loader)
    val_data = next(val_dataiter)
    print(val_data.shape, len(validation_loader))
