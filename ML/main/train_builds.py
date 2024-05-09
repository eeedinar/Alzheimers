from lesions import *
from utils import *
from torch import nn, optim
import yaml, json
from sklearn.metrics import balanced_accuracy_score

# loss and optimizer definition
def BCELoss_class_weighted(weights):
    def loss(input, target):
        input = torch.clamp(input, min=1e-7, max=1-1e-7)
        bce = - (weights[1]*target*torch.log(input) + weights[0]*(1-target)*torch.log(1-input))
        return torch.mean(bce)
    return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weights=None):
        super(FocalLoss, self).__init__()
        self.gamma  = gamma
        self.weights = weights

    def forward(self, input, target):
        self.logpt = F.log_softmax(input, dim=1)
        self.pt    = torch.exp(self.logpt)
        prob  = ((1-self.pt)**self.gamma)*torch.log(self.pt)
        loss  = F.nll_loss(prob, target, self.weights)
        return loss

def build_optimizer(optimizer, model, lr, momentum=0.9):
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)  
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    return optimizer

# balanced accuracy as the performance metrics
def performace_metrics(y_pred, y_true):
    avg_vacc = balanced_accuracy_score(y_true, y_pred)
    return avg_vacc


def build_dataset(dataset_source, config, device, yaml_file=None, sonar_file=None):
    """
        read yaml_file, update config parameters, and return train and test dataloader, and weights to the device
        config, input_dim, output_dim, weights, training_loader, validation_loader = build_dataset("mar-2023", config, device, yaml_file=yaml_file)
        config, input_dim, output_dim, weights, training_loader, validation_loader = build_dataset("sonar", config, device, sonar_file=sonar_file)

    """
    if dataset_source == "mar-2023":
        qvalue_lower_bound = config.qvalue_lower_bound
        qvalue_upper_bound = config.qvalue_upper_bound
        Excel_File   = config.Excel_File
        sheet        = config.sheet
        BNL_dir      = config.BNL_dir
        sub_dir      = config.sub_dir
        column_names = config.column_names
        output_dim   = len(column_names)
        lidx, uidx, input_dim = idx_from_grid(qvalue_lower_bound, qvalue_upper_bound)
        val_files    = config.val_files
        # create dataloader with dataset
        # weights, training_loader, validation_loader = get_dataloaders_fixed_val_files(Excel_File, sheet, BNL_dir, sub_dir, column_names, val_files, lidx=lidx, uidx=uidx)
        # weights, training_loader, validation_loader = get_dataloaders_random_split(Excel_File, sheet, BNL_dir, sub_dir, column_names, lidx=lidx, uidx=uidx)

        weights, training_loader, validation_loader = get_dataloaders_fixed_val_files(Excel_File, sheet, BNL_dir, sub_dir, column_names, val_files, lidx, uidx)


    elif dataset_source == "sonar":

        weights, training_loader, validation_loader = get_sonar_dataloaders(sonar_file)
        input_dim = 60
        output_dim = 2

    config.update({"input_dim": input_dim, "output_dim": output_dim})
    
    return input_dim, output_dim, weights.to(device), training_loader, validation_loader


def build_model(network, yaml_file, config, device, input_dim, output_dim) :
    if network == 'transformer':
        from Transformer import Transformer

        ### Transformer model specs update
        seq_len = config.seq_len
        
        assert input_dim%seq_len ==0 , f'q value range is not divisible by {seq_len}'
        embedding_dim = input_dim//seq_len
        
        n_heads = config.n_heads 
        hidden_dim = config.hidden_dim
        N = config.N
        dropout = config.dropout

        model = Transformer(n_heads, seq_len, embedding_dim, hidden_dim, N, dropout, n_classes=output_dim)
        model.to(device)  # in place operation
        print(f'model is in cuda ? {next(model.parameters()).is_cuda}')

        config.update({"embedding_dim": embedding_dim})

    return model

def build_loss(loss_func, weights):
    if loss_func == 'cross_entropy':      
        return torch.nn.CrossEntropyLoss(weight=weights)  # weight=weights   
    elif loss_func == 'focal_loss':
        return FocalLoss(weights=weights)
    elif loss_func == 'triplet_margin_loss':
        return torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    return
