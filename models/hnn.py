import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], 16, 4, 4)


class HNN(nn.Module):
    def __init__(self, dt=0.125):
        super().__init__()
        self.dt = dt

        self.encoder = nn.Sequential(
            # b 90 32 32
            nn.Conv2d(90, 32, kernel_size=3, padding=1),
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

        self.hamiltonian = nn.Sequential(
            nn.Linear(512, 1)
        )

        self.decoder = nn.Sequential(
            # b 256
            UnFlatten(),
            # b 16 4 4
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            # b 16 8 8
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            # b 16 16 16
            nn.ConvTranspose2d(16, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
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

    def ham_grad(self, z):
        batch_size = z.shape[0]
        return self.hamiltonian[0].weight.repeat(batch_size, 1)

    def integrate(self, z):
        z[:, 256:] -= 0.5 * self.ham_grad(z)[:, :256] * self.dt
        z[:, :256] += self.ham_grad(z)[:, 256:] * self.dt
        z[:, 256:] -= 0.5 * self.ham_grad(z)[:, :256] * self.dt
        return z

    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        result = torch.empty_like(x)
        result[:, :3] = self.decoder(z[:, :256])
        for i in range(3, 90, 3):
            z = self.integrate(z)
            result[:, i: i + 3] = self.decoder(z[:, :256])
        return result, mu, logvar

