import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], 32, 4, 4)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # b 3 32 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            # b 32 16 16
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            # b 32 8 8
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            # b 32 4 4
            Flatten()
            # b 512
        )
        
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)

        self.decoder = nn.Sequential(
            # b 512
            UnFlatten(),
            # b 32 4 4
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            # b 16 8 8
            nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            # b 16 16 16
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid(),
            # b 3 32 32
        )
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).cuda()
        z = mu + std * eps
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return self.decoder(z), mu, logvar

