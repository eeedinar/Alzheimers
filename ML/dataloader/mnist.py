import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def get_dataset():

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])

    # Create datasets for training & validation, download if necessary
    training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)

    return training_set, validation_set

if __name__ == '__main__':
    training_set, validation_set = get_dataset()
    print(training_set.data.shape, training_set.targets.shape)