import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            # nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(8, latent_dim)
            )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            # nn.Linear(16, 64),
            # nn.ReLU(),
            nn.Linear(8, input_dim),

            )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

if __name__ == '__main__':
    x = torch.randn(100,50)

    model = AutoEncoder(input_dim=50, latent_dim=3)
    decoded = model(x)

    print(decoded.shape)
