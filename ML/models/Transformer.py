import torch
import torch.nn as nn
import math
import onnx

class LayerNormalization(nn.Module):
    """
    x= torch.tensor([[[1,2,3],
                       [4,5,6]],
                     [[1,2,3],
                      [4,5,6]]
               ], dtype=torch.float32)
    m = LayerNormalization_me(3)
    m(x) =     [[[-1.,  0.,  1.],
                 [-1.,  0.,  1.]],

                [[-1.,  0.,  1.],
                 [-1.,  0.,  1.]]]
    """
    def __init__(self, norm_dim, eps = 1e-5):
        super().__init__()
        self.eps   = eps                                   # value given for numerical stability
        self.gamma = nn.Parameter(torch.ones (norm_dim) )  # vector of size norm_dim as is going to be squeezed for mean and var
        self.beta  = nn.Parameter(torch.zeros(norm_dim) )  # vector of size norm_dim as is going to be squeezed for mean and var

    def forward(self, x):
        mean  = torch.mean(x,  dim = -1, keepdim = True)   # mean over normalization dimension
        var   = torch.var (x,  dim = -1, keepdim = True)   # variance over normalization dimension

        return self.gamma*(x - mean)/(torch.sqrt(var + self.eps)) + self.beta


class InputEmbeddings(nn.Module):
    """
        x = torch.tensor([
                            [0,1],
                            [2,0]
                        ])
        m = InputEmbeddings(5, 2)
        print(m.embedding.weight* math.sqrt(m.embedding_dim))
            tensor([[ 2.4452,  0.8923],
                    [ 1.5760,  1.2485],
                    [-0.0620, -0.6966],
                    [-0.1592,  1.4780],
                    [ 0.7227,  2.2764]], grad_fn=<MulBackward0>)
        print(m(x))
            tensor([[[ 2.4452,  0.8923],
                     [ 1.5760,  1.2485]],

                    [[-0.0620, -0.6966],
                     [ 2.4452,  0.8923]]], grad_fn=<MulBackward0>)

        print(m(x).size())              # torch.Size([2, 2, 2])
    """

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings                                         # vocabulary of a dictionary
        self.embedding_dim  = embedding_dim                                          # embedding dimension
        self.embedding      = nn.Embedding(num_embeddings, embedding_dim)            # (num_embeddings, embedding_dim)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding_dim)                    # (batch, seq_len_of_indices_for_words) -> (batch, seq_len, embedding_dim) scaling per paper

                                                       # (batch, seq_len, embedding_dim) -> (batch, seq_len, embedding_dim)

class PositionalEncodings(nn.Module):
    """
    x.size() = seq_len, embedding_dim
    x= torch.tensor([[[1,2,3,4,5,6],
                      [2,1,3,4,5,6],
                      [6,5,4,3,2,1],
                      [1,2,3,4,5,6]],
               ], dtype=torch.float32)
    m = PositionalEncodings(4,6)
    print(m.pe)
                tensor([[[ 0.0000,  1.0000,  0.0000,  1.0000,  0.0000,  1.0000],
                         [ 0.8415,  0.5403,  0.0464,  0.9989,  0.0022,  1.0000],
                         [ 0.9093, -0.4161,  0.0927,  0.9957,  0.0043,  1.0000],
                         [ 0.1411, -0.9900,  0.1388,  0.9903,  0.0065,  1.0000]]])
    print(m(x))
                tensor([[[1.0000, 3.0000, 3.0000, 5.0000, 5.0000, 7.0000],
                         [2.8415, 1.5403, 3.0464, 4.9989, 5.0022, 7.0000],
                         [6.9093, 4.5839, 4.0927, 3.9957, 2.0043, 2.0000],
                         [1.1411, 1.0100, 3.1388, 4.9903, 5.0065, 7.0000]],

    """
    def __init__(self, seq_len, embedding_dim):
        super().__init__()

        half_len = embedding_dim //2
        print('embedding dim is ',embedding_dim)
        assert embedding_dim %2 ==0 , "embedding dimension must be divisible by 2"
        pe  = torch.zeros(seq_len, embedding_dim)

        pos = torch.arange(0,seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = 10000**-(2*torch.arange(0,half_len)/embedding_dim).unsqueeze(0)

        pe[:,0::2] = torch.sin(pos @ div_term)
        pe[:,1::2] = torch.cos(pos @ div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self, x):
        return x + self.pe

class MaskedMultiHeadAttention(nn.Module):
    """
    x= torch.tensor([[[1,2,3,4],
                     [4,5,6,7]],
                     [[1,2,3,4],
                     [4,5,6,7]]
                ], dtype=torch.float32)

    m = MaskedMultiHeadAttention(n_heads=2, embedding_dim=4)
    m(x,x,x,None)
    """

    def __init__(self, n_heads, embedding_dim, dropout=None):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_heads        = n_heads

        assert self.embedding_dim % self.n_heads == 0 , "embedding_dim is not divisible by total no. of heads"
        self.d_k           = self.embedding_dim // n_heads                                 # each head gets partitioned embedding_dim

        self.layer_query   = nn.Linear(embedding_dim, embedding_dim, bias = False)        # query  layer weight
        self.layer_key     = nn.Linear(embedding_dim, embedding_dim, bias = False)        # key    layer weight
        self.layer_value   = nn.Linear(embedding_dim, embedding_dim, bias = False)        # value  layer weight
        self.layer_out     = nn.Linear(embedding_dim, embedding_dim, bias = False)        # output layer weight
        self.softmax       = nn.Softmax(dim = -1)


        self.dropout       = dropout
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        query = self.layer_query(q)                   # (batch, seq_len, embedding_dim) -> (batch, seq_len, embedding_dim)
        key   = self.layer_key(k)                     # (batch, seq_len, embedding_dim) -> (batch, seq_len, embedding_dim)
        value = self.layer_value(v)                   # (batch, seq_len, embedding_dim) -> (batch, seq_len, embedding_dim)

        # split q,k,v embedding_dim into d_k dim
        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(2, 1)  # (batch, seq_len, embedding_dim) -> (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(2, 1)  # (batch, seq_len, embedding_dim) -> (batch, seq_len, n_heads, d_k) -> (batch, n_heads, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).permute((0, 2, 3, 1))    # (batch, seq_len, embedding_dim) -> (batch, seq_len, n_heads, d_k) -> (batch, n_heads, d_k, seq_len)
        
        self.attention_scores = query @ key / math.sqrt(self.d_k)   # (batch, n_heads, seq_len, d_k)*(batch, n_heads, d_k, seq_len) -> (batch, n_heads, seq_len, seq_len)
        if mask is not None:
            self.attention_scores.masked_fill_(mask==0, -1e-5)                 # (batch, n_heads, seq_len, seq_len)
        
        self.attention_scores = self.softmax(self.attention_scores)            # (batch, n_heads, seq_len, seq_len)
        
        if self.dropout is not None:
            self.attention_scores = self.dropout(self.attention_scores)        # (batch, n_heads, seq_len, seq_len)

        self.out = self.attention_scores @ value                               # (batch, n_heads, seq_len, seq_len)*(batch, n_heads, seq_len, d_k) -> (batch, n_heads, seq_len, d_k)

        # concatenate heads back into embedding_dim
        self.out = self.out.transpose(2,1).contiguous().view(value.shape[0], -1, self.embedding_dim)   # -1 for seq_len --> (batch, n_heads, seq_len, d_k)-> (batch, seq_len, n_heads, d_k) -> (batch, seq_len, embedding_dim)
        self.out = self.layer_out(self.out)      # (batch, seq_len, embedding_dim) -> (batch, seq_len, embedding_dim)

        return self.out

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.linear   = nn.Linear(input_dim,  hidden_dim)  # weight and bias
        self.out      = nn.Linear(hidden_dim, output_dim)  # weight and bias
        self.dropout  = nn.Dropout(dropout)
        self.relu     = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.linear(x)))                       # (batch, seq_len, embedding_dim) -> (batch, seq_len, hidden_dim)
        x = self.out(x)                                                   # (batch, seq_len, hidden_dim)    -> (batch, seq_len, embedding_dim)
        return x   

class ResidualConnection(nn.Module):
    def __init__(self , norm_dim, dropout):
        super().__init__()
        self.layernorm = LayerNormalization(norm_dim)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.layernorm( x + sublayer( self.dropout(x) ) )         # (batch, seq_len, embedding_dim)


class EncoderBlock(nn.Module):
    def __init__(self, norm_dim: int, self_attention : MaskedMultiHeadAttention, feed_forward: FeedForward, dropout:float):
        super().__init__()
        self.self_attention  = self_attention
        self.feed_forward    = feed_forward
        self.residual_connection = nn.ModuleList([ResidualConnection(norm_dim, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention(x, x, x, src_mask) )
        x = self.residual_connection[1](x, self.feed_forward      )
        return x                                                         # (batch, seq_len, embedding_dim)

class Encoder(nn.Module):
    """
    Nx blocks are connected -> the output of one encoder output is feed to the next block
    """
    def __init__(self, norm_dim: int, modules_list: nn.ModuleList ):
        super().__init__()
        self.layernorm = LayerNormalization(norm_dim)
        self.modules_list = modules_list

    def forward(self, x, src_mask):
        for module in self.modules_list:
            x = module(x, src_mask)              # encoder blocks are serially connected
        return self.layernorm(x)                 # (batch, seq_len, embedding_dim)

class InputFeeding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, seq_len):
        super().__init__()
        # self.src_emb = InputEmbeddings(num_embeddings, embedding_dim)
        self.src_enc = PositionalEncodings(seq_len, embedding_dim)

    def forward(self, x):
        return x + self.src_enc(x)

# class OutputFeedForward(nn.Module):
#     def __init__(self, input_dim, dropout, n_classes):
#         super().__init__()
#         self.input_dim = input_dim
#         self.m = nn.Sequential(nn.Linear(input_dim,32), 
#                                         nn.LeakyReLU(), 
#                                         nn.Dropout(dropout), 
#                                         nn.Linear(32,16),
#                                         nn.LeakyReLU(), 
#                                         nn.Dropout(dropout), 
#                                         nn.Linear(16,n_classes))

#     def forward(self, x):
#         x = x.view(-1, self.input_dim)
#         return self.m(x)

class OutputFeedForward(nn.Module):

    def __init__(self, input_dim, dropout, n_classes):
        super().__init__()
        in_channels = 1
        # self.layer1 = nn.Conv1d(in_channels, 16, kernel_size=1, padding=0, dilation=1, stride=1, bias=True)
        # self.act1   = nn.LeakyReLU(0.01)
        kernel_size = 5
        padding     = 0
        dilation    = 2
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

    def forward(self,x):
        # x      = self.act1(self.layer1(x))
        x      = self.act2(self.layer2(x))
        
        # self.feature.weight.data.mul_(self.const)
        # x      = self.feature(x)

        x      = x.view(-1,x.size()[-1])
        x      = self.dropout(self.act3(self.layer3(x)))
        x      = self.layer4(x)
        return x

class Transformer(nn.Module):
    def __init__(self, n_heads, seq_len, embedding_dim, hidden_dim, N, dropout, n_classes):
        super().__init__()
        n_heads                     = n_heads              # heads for one encoder blcok
        embedding_dim               = embedding_dim        # embedding dim
        input_dim                   = embedding_dim
        norm_dim                    = embedding_dim
        output_dim                  = input_dim
        hidden_dim                  = hidden_dim
        attn_dropout                = dropout
        feed_forward_dropout        = dropout
        residual_dropout            = dropout
        N                           = N                    # total no. of encodoer blocks
        self.seq_len                = seq_len
        self.embedding_dim          = embedding_dim
        self.n_classes              = n_classes            # output layer number of neurons
        encoder_blocks = []

        for _ in range(N):
            self_attention = MaskedMultiHeadAttention(n_heads, embedding_dim, attn_dropout)
            feed_forward   = FeedForward(input_dim, hidden_dim, output_dim, feed_forward_dropout)
            encoder_block  = EncoderBlock(norm_dim, self_attention, feed_forward, residual_dropout) # norm_dim is for layernormalization
            encoder_blocks.append(encoder_block)

        self.encoder = Encoder(norm_dim, nn.ModuleList(encoder_blocks))  # norm_dim is for layernormalization

        ### dummy
        num_embeddings = 1   # has no impact on computation
        self.src_enc = InputFeeding(num_embeddings, embedding_dim, seq_len)
        self.model   = OutputFeedForward(self.seq_len*self.embedding_dim, dropout, self.n_classes)
    
    def forward(self, x, src_mask):
        x = x.view(x.shape[0], self.seq_len, self.embedding_dim)
        # x = self.src_enc(x)
        # x = self.encoder(x, src_mask)
        x = self.model(x)
        return x

if __name__ == '__main__' :

    n_heads        = 1
    hidden_dim     = 3
    N              = 1
    dropout        = 0

    seq_len        = 1
    embedding_dim  = 4

    n_classes      = 3

    # x = torch.tensor([[[1,2,3,4],
    #                   [4,5,6,7]],
    #                  [[1,2,3,4],
    #                   [4,5,6,7]]
    #         ], dtype=torch.float32)

    x = torch.tensor([[1,2,3,4, 4,5,6,7],
                      [1,2,3,4, 4,5,6,7]], dtype=torch.float32)

    transformer = Transformer(n_heads, seq_len, embedding_dim, hidden_dim, N, dropout, n_classes)
    out = transformer(x[:,:4], None)
    print(out)
    # print(transformer.encoder.modules_list[0].self_attention.attention_scores[0,0,:,:])

    # torch.onnx.export(transformer, (x,None), 'bnl.onnx', input_names=["features"], output_names=["logits"])

