class OutputFeedForward(nn.Module):

    def __init__(self, input_dim, dropout, n_classes):
        super().__init__()
        in_channels = 1
        # self.layer1 = nn.Conv1d(in_channels, 16, kernel_size=1, padding=0, dilation=1, stride=1, bias=True)
        # self.act1   = nn.LeakyReLU(0.01)
        kernel_size = 3
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

        self.layer3  = nn.Linear(self.layer2_dim, 4) # self.layer2_dim   input_dim
        self.act3    = nn.LeakyReLU(0.01) # nn.ReLU() # nn.LeakyReLU(0.01) 
        self.dropout = nn.Dropout(dropout)
        self.layer4  = nn.Linear(4,n_classes)
        # self.layer4  = nn.Linear(self.layer2_dim,n_classes)

    def forward(self,x):
        # x      = self.act1(self.layer1(x))
        x      = self.act2(self.layer2(x))
        
        # self.feature.weight.data.mul_(self.const)
        # x      = self.feature(x)

        x      = x.view(-1,x.size()[-1])
        x      = self.dropout(self.act3(self.layer3(x)))
        x      = self.layer4(x)
        return x