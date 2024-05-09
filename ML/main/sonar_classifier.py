import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import glob
import ast

# unequal level of list depth/nesting
def flatten(S):
    """
    l = [2,[[1,2]],1]
    list(flatten(l))
    """
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

def interpolate_missing(A):
    # indx, A = interpolate_missing(Iq_M_WAXS[1])

    # https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    ok = ~ np.isnan(A)
    xp = ok.ravel().nonzero()[0]
    fp = A[~ np.isnan(A)]
    x  = np.isnan(A).ravel().nonzero()[0]
    A[np.isnan(A)] = np.interp(x, xp, fp)

    return x, A

#### Sonar Data
# Read data
# data = pd.read_csv("sonar.csv", header=None)
# X = data.iloc[:, 0:60]
# y = data.iloc[:, 60]

# # Binary encoding of labels
# encoder = LabelEncoder()
# encoder.fit(y)
# y = encoder.transform(y)

# # Convert to 2D PyTorch tensors
# X = torch.tensor(X.values, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)


#### X-ray Data

qgrid2 = np.hstack([np.arange(0.005, 0.0499, 0.001), np.arange(0.05, 0.099, 0.002), np.arange(0.1, 3.2, 0.005)])
lidx = np.argmin(qgrid2 < 0.5)   # qvalue = 0.7,  idx = 190
uidx = np.argmin(qgrid2 < 1.46)   # qvalue = 1.46, idx = 342

input_dim = uidx-lidx

# load data
### specs : names - file , columns
file        = "/Users/bashit.a/Documents/Alzheimer/Mar-2023/Mar-2023-Samples.xlsx"
sheet       = 'Mar-2023-Samples'
column_names= {"Neuritic_Plaque":1 , "Tissue":0}   # "Diffuse_Plaque"   "Neurofibrillary_Tangle_(tau)"    "Neuritic_Plaque"  "Tissue"
BNL_dir     = '/Volumes/HDD/BNL-Data/Mar-2023'
sub_dir     = "CSV_Conv-8-point"  # CSV_Conv-8-point  CSV


### read dataframe
df = pd.read_excel(file, sheet_name=sheet)
df["File_Loc"] = pd.Series([], dtype=str)
for idx, file in zip(df.index, df['File']):
    for file_search in glob.iglob(f'{BNL_dir}/**/*', recursive=True):
        if file_search.find(file) > -1 and f'/{sub_dir}/' in file_search:
            df.at[idx, 'File_Loc'] = file_search
            break


Iq_values = []
labels    = []
files = []
frames_t = []
for column_name,label in column_names.items():
    ### indices where column values are located
    indices = df[column_name].dropna().index
    ### looping over columns
    for idx in indices:
        # print(df['File_Loc'][idx], '--> ', flatten(pd.eval(df[column_name].dropna()[idx])), '-->', ast.literal_eval(df['bkg'][idx]) if type(df['bkg'][idx]) is str else 'NaN' )
    
        df_temp = pd.read_csv(df['File_Loc'][idx], delimiter=",")
    
        if type(ast.literal_eval(df[column_name][idx])) is dict:
            for k, v in (ast.literal_eval(df[column_name][idx])).items():
                # print(k,v)
                pass
        elif type(ast.literal_eval(df[column_name][idx])) is tuple:
            # print(  flatten(list(ast.literal_eval(df[column_name][idx]))) )
            pass
        
        elif type(ast.literal_eval(df[column_name][idx])) is list:
            # print( list(ast.literal_eval(df[column_name][idx])))
            pass

        frames = flatten( list(ast.literal_eval(df[column_name][idx])) )
        
        Iq_values.append( df_temp.iloc[frames].values )
        labels.append([label]*len(frames))
        files.append(df['File_Loc'][idx])
        frames_t.append(frames)

x = np.vstack(Iq_values)
x = x[:,lidx:uidx].astype(np.float32)
assert np.any(np.isnan(x))==False, "X contains NaN values"
_, x = interpolate_missing(x)
x = x/np.max(x,axis=1).reshape(-1,1)

y = np.array(flatten(labels) ,dtype=np.float32).reshape(-1,1)
n_samples = len(y)

# Convert to 2D PyTorch tensors
X = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)



# Define two models
class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(input_dim, 180)
        self.relu = nn.ReLU()
        self.output = nn.Linear(180, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x

class Deep(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

# Compare model sizes
model1 = Wide()
model2 = Deep()
print(sum([x.reshape(-1).shape[0] for x in model1.parameters()]))  # 11161
print(sum([x.reshape(-1).shape[0] for x in model2.parameters()]))  # 11041

# Helper function to train one model
def model_train(model, X_train, y_train, X_val, y_val):
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 300   # number of epochs to run
    batch_size = 512  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()

                # print(loss.item())
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc

# train-test split: Hold out the test set for final model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# define 5-fold cross validation test harness
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cv_scores_wide = []
for train, test in kfold.split(X_train, y_train):
    # create model, train, and get accuracy
    model = Wide()
    acc = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test])
    print("Accuracy (wide): %.2f" % acc)
    cv_scores_wide.append(acc)
cv_scores_deep = []
for train, test in kfold.split(X_train, y_train):
    # create model, train, and get accuracy
    model = Deep()
    acc = model_train(model, X_train[train], y_train[train], X_train[test], y_train[test])
    print("Accuracy (deep): %.2f" % acc)
    cv_scores_deep.append(acc)

# evaluate the model
wide_acc = np.mean(cv_scores_wide)
wide_std = np.std(cv_scores_wide)
deep_acc = np.mean(cv_scores_deep)
deep_std = np.std(cv_scores_deep)
print("Wide: %.2f%% (+/- %.2f%%)" % (wide_acc*100, wide_std*100))
print("Deep: %.2f%% (+/- %.2f%%)" % (deep_acc*100, deep_std*100))

# rebuild model with full set of training data
if wide_acc > deep_acc:
    print("Retrain a wide model")
    model = Wide()
else:
    print("Retrain a deep model")
    model = Deep()
acc = model_train(model, X_train, y_train, X_test, y_test)
print(f"Final model accuracy: {acc*100:.2f}%")

model.eval()
with torch.no_grad():
    # Test out inference with 5 samples
    for i in range(5):
        y_pred = model(X_test[i:i+1])
        print(f"{X_test[i].numpy()} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")

    # Plot the ROC curve
    y_pred = model(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr) # ROC curve = TPR vs FPR
    plt.title("Receiver Operating Characteristics")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()