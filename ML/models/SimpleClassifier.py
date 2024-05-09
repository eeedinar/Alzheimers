import torch
import torch.nn as nn
import torch.nn.functional as F


class MyClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MyClassifier, self).__init__()

        self.layer1 = nn.Linear(input_dim, 16)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(16, 32)
        self.act2 = nn.LeakyReLU(0.2)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.dropout(self.act2(self.layer2(x)))
        x = self.sigmoid(self.output(x))

        return x

if __name__ == '__main__' :
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyClassifier(10)
    model.to(device)
    inputs = torch.randn(5, 10).to(device)
    out = model(inputs)
    print(out)
    print(out.is_cuda, inputs.is_cuda, next(model.parameters()).is_cuda)