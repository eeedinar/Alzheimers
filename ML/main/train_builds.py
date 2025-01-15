from lesions import *
from utils import *
from torch import nn, optim
import yaml, json, copy
from sklearn.metrics import balanced_accuracy_score


def weighted_balanced_accuracy(y_pred, y_true, weights):
    """
    Calculate weighted balanced accuracy.

    Parameters:
    y_pred (torch.Tensor): Predicted labels.
    y_true (torch.Tensor): True labels.
    weights (torch.Tensor): Sample weights.

    Returns:
    balanced_accuracy (float): Weighted balanced accuracy.
    """
    # input cast to long
    y_pred = y_pred.long()
    y_true = y_true.long()

    # Normalize weights
    weights = weights / weights.sum()

    # Create a tensor with weights for each true label
    weights = torch.tensor([weights[label.item()] for label in y_true])

    # Get unique classes
    classes = torch.unique(y_true)

    weighted_recalls = []

    for cls in classes:
        # Mask for true class
        true_class_mask = (y_true == cls)

        # Calculate true positives and false negatives
        tp = torch.sum(torch.logical_and(y_true == cls, y_pred == cls) * weights)
        fn = torch.sum(torch.logical_and(y_true == cls, y_pred != cls) * weights)

        # Calculate recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calculate weighted recall
        weighted_recall = torch.sum(weights[true_class_mask]) * recall
        weighted_recalls.append(weighted_recall)

    # Calculate weighted sum
    weighted_sum = torch.sum(weights)

    # Calculate balanced accuracy
    balanced_accuracy = torch.sum(torch.tensor(weighted_recalls)) / weighted_sum
    return balanced_accuracy

# loss and optimizer definition
class BCELoss(nn.Module):
    def __init__(self, weights=torch.tensor([1.0,1.0], dtype=torch.float32)):
        super(BCELoss, self).__init__()
        self.weights = weights

    def forward(self, input, target):
        input = torch.clamp(input, min=1e-7, max=1-1e-7)
        bce = - (self.weights[1]*target*torch.log(input) + self.weights[0]*(1-target)*torch.log(1-input))    
        return torch.mean(bce)

class FocalLoss(nn.Module):
    def __init__(self, gamma=0., weights=None):
        super(FocalLoss, self).__init__()
        self.gamma  = gamma
        self.weights = weights

    def forward(self, input, target):
        self.input = input
        self.logpt = F.log_softmax(self.input, dim=1)
        self.pt    = torch.exp(self.logpt)
        self.prob  = ((1-self.pt)**self.gamma)*torch.log(self.pt)
        self.loss  = F.nll_loss(self.prob, target, self.weights)
        # print(self.input, self.logpt, self.pt   , self.prob , self.loss )
        return self.loss

def build_optimizer(optimizer, model, lr, momentum=0.9):
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    return optimizer

# balanced accuracy as the performance metrics
def performace_metrics(y_pred, y_true):
    return balanced_accuracy_score(y_true, y_pred)

def build_dataset(dataset_source, config, device, yaml_file=None, sonar_file=None):
    """
        read yaml_file, update config parameters, and return train and test dataloader, and weights to the device
        config, input_dim, output_dim, weights, training_loader, validation_loader = build_dataset("mar-2023", config, device, yaml_file=yaml_file)
        config, input_dim, output_dim, weights, training_loader, validation_loader = build_dataset("sonar", config, device, sonar_file=sonar_file)

    """
    if dataset_source == "mar-2023":
        qvalue_lower_bound = config.qvalue_lower_bound
        qvalue_upper_bound = config.qvalue_upper_bound
        mica_sub           = config.mica_sub
        scaling            = config.scaling

        Excel_File   = config.Excel_File
        sheet        = config.sheet
        BNL_dir      = config.BNL_dir
        sub_dir      = config.sub_dir
        column_names = config.column_names
        output_dim   = len(np.unique(list(column_names.values()), return_counts=False))
        lidx, uidx, input_dim = idx_from_grid(qvalue_lower_bound, qvalue_upper_bound)
        val_files    = config.val_files
        test_files   = config.test_files
        # create dataloader with dataset
        # weights, training_loader, validation_loader = get_dataloaders_fixed_val_files(Excel_File, sheet, BNL_dir, sub_dir, column_names, val_files, lidx=lidx, uidx=uidx)
        # weights, training_loader, validation_loader = get_dataloaders_random_split(Excel_File, sheet, BNL_dir, sub_dir, column_names, lidx=lidx, uidx=uidx)

        weights, training_loader, validation_loader, testing_loader = get_dataloaders_fixed_val_test_files(Excel_File, sheet, BNL_dir, sub_dir, column_names, val_files, test_files, lidx, uidx, mica_sub, scaling)


    elif dataset_source == "sonar":

        weights, training_loader, validation_loader = get_sonar_dataloaders(sonar_file)
        input_dim = 60
        output_dim = 2
        testing_loader = None

    config.update({"input_dim": input_dim, "output_dim": output_dim})
    
    return input_dim, output_dim, weights.to(device), training_loader, validation_loader, testing_loader


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


    elif network == 'simple-linear':

        class SimpleBinaryClassifier(nn.Module):
            def __init__(self, input_dim, dropout, n_classes):
                super().__init__()
                
                self.input_dim = input_dim


                in_channels = 1
                # self.layer1 = nn.Conv1d(in_channels, 16, kernel_size=1, padding=0, dilation=1, stride=1, bias=True)
                # self.act1   = nn.LeakyReLU(0.01)
                kernel_size = 5
                padding     = 0
                dilation    = 1
                stride      = 1
                self.layer2     = nn.Conv1d(in_channels, 1, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, bias=True)
                self.layer2_dim = (input_dim + 2*padding - dilation*(kernel_size-1) - 1)//stride + 1 
                self.act2       = nn.LeakyReLU(0.01) # nn.ReLU() # nn.LeakyReLU(0.01)
                # print(self.layer2_dim)
                # self.feature  = nn.Linear(self.layer2_dim, self.layer2_dim, bias=False)
                # self.const    = torch.diag( torch.ones(self.layer2_dim) )
                # self.feature.weight = nn.Parameter(self.const)

                self.layer3  = nn.Linear(self.layer2_dim, 2) # self.layer2_dim   input_dim
                self.act3    = nn.LeakyReLU(0.01) # nn.ReLU() # nn.LeakyReLU(0.01) 
                self.dropout = nn.Dropout(dropout)
                self.layer4  = nn.Linear(2,n_classes)
                # self.layer4  = nn.Linear(self.layer2_dim,n_classes)

            def forward(self,x, src_mask):
                x = x.view(x.shape[0], 1, self.input_dim)
                # x      = self.act1(self.layer1(x))
                x      = self.act2(self.layer2(x))
                
                # self.feature.weight.data.mul_(self.const)
                # x      = self.feature(x)

                x      = x.view(-1,x.size()[-1])
                x      = self.dropout(self.act3(self.layer3(x)))
                x      = self.layer4(x)
                return x

        model = SimpleBinaryClassifier(input_dim,  dropout=0.05, n_classes=1)
        model.to(device)  # in place operation
        print(f'model is in cuda ? {next(model.parameters()).is_cuda}')

    return model

def build_loss(loss_func, weights):
    if loss_func == 'cross_entropy':      
        return torch.nn.CrossEntropyLoss(weight=weights)  # weight=weights   
    elif loss_func == 'focal_loss':
        return FocalLoss(weights=weights)
    elif loss_func == 'triplet_margin_loss':
        return torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    elif loss_func == 'BCELoss':
        return BCELoss(weights=weights)    


class EarlyStopping:
    def __init__(self, patience=2000, min_delta=0, restore_best_model=True):
        self.patience  = patience
        self.min_delta = min_delta
        self.restore_best_model = restore_best_model
        self.counter = 0
        self.best_model = None
        self.best_vloss = None
        self.status = ""
        self.best_epoch = 0

    def __call__(self, model, val_loss, epoch):
        if self.best_vloss is None or (self.best_vloss - val_loss) >= self.min_delta:
            self.counter = 0
            self.status = f'Imrovement VLoss'
            self.best_vloss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
        else:
            self.counter += 1
            self.status = f'No Imrovement VLoss {self.counter}'
            if self.counter == self.patience:
                self.status = f'model is triggering after patience:{self.counter}, min_delta:{self.min_delta}'
                if self.restore_best_model:
                    self.status += f' Restoreing Epoch:{self.best_epoch}, VLoss:{self.best_vloss}'
                    model.load_state_dict(self.best_model)
                return True
        return False
