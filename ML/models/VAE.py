import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VariationalAutoEncoder, self).__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # encoder
        self.input_2hidden  = nn.Linear(self.input_dim,  self.hidden_dim )
        self.hidden_2mu     = nn.Linear(self.hidden_dim, self.latent_dim )
        self.hidden_2sigma  = nn.Linear(self.hidden_dim, self.latent_dim )        

        # decoder
        self.latent_2hidden = nn.Linear(self.latent_dim, self.hidden_dim )
        self.hidden_2input  = nn.Linear(self.hidden_dim, self.input_dim  )


        self.eps = torch.randn(self.latent_dim, requires_grad=False,)

    def encoder(self, x):
        # q_phi(z|x) --> mu and sigma
        hidden = F.relu(self.input_2hidden(x))
        mu     = self.hidden_2mu(hidden)                           # mu and sigma don't require activation function
        logvar  = self.hidden_2sigma(hidden)
        return mu, logvar    # mu, log(var) = log(sigma^2)

    def reparameterize(self,  mu, logvar):
        std = torch.exp(0.5*logvar)
        # eps = torch.randn_like(std)
        latent = mu + std*self.eps
        return latent

    def decoder(self, latent):
        # p(x|z)
        hidden = F.relu(self.latent_2hidden(latent))
        x_reconstructed = F.sigmoid(self.hidden_2input(hidden)) # reconstructed value is [0,1]
        return x_reconstructed

    def forward(self, x):
        # p(x_reconstruc|x)
        # encoder
        mu, logvar = self.encoder(x)             # encoder --> mu, logvar for KL divergence
        # parameterization
        self.latent = self.reparameterize(mu, logvar)
        # decoder
        x_reconstructed = self.decoder(self.latent)  # decoder - x_reconstructed --> for reconstruction loss

        return x_reconstructed, mu, logvar

if __name__ == '__main__':
    x = torch.randn(100,50)

    model = VariationalAutoEncoder(input_dim=50, hidden_dim=20, latent_dim=3)
    x_reconstructed, mu, logvar = model(x)

    print(x_reconstructed.shape, mu.shape, logvar.shape)
