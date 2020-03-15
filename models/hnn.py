import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], 16, 4, 4)


class HNN(nn.Module):
    def __init__(self, input_length=30, output_length=45, dt=0.125):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.dt = dt

        self.encoder = nn.Sequential(
            # b input_length 32 32
            nn.Conv2d(input_length, 32, kernel_size=3, padding=1),
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
            nn.Linear(512, 512),
            nn.Softplus(),
            nn.Linear(512, 512),
            nn.Softplus(),
            nn.Linear(512, 1)
        )

        self.h1 = lambda z: self.hamiltonian[0](z)
        self.h2 = lambda z: self.hamiltonian[2](self.hamiltonian[1](self.h1(z)))

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
        std = logvar.mul(0.5).exp()
        eps = torch.randn(*mu.size()).cuda()
        z = mu + std * eps
        return z
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def ham_grad(self, z):
        return ((self.hamiltonian[0].weight.t()[None] * torch.sigmoid(self.h1(z))[:, None])
                @ (self.hamiltonian[2].weight.t()[None] * torch.sigmoid(self.h2(z))[:, None])
                @ self.hamiltonian[4].weight.t()[None])[:, :, 0]

    def integrate(self, z):
        grad = self.ham_grad(z)[:, :256]
        z1 = z - 0.5 * torch.cat([torch.zeros_like(grad), grad], axis=1) * self.dt
        grad = self.ham_grad(z1)[:, 256:]
        z2 = z1 + torch.cat([grad, torch.zeros_like(grad)], axis=1) * self.dt
        grad = self.ham_grad(z2)[:, :256]
        z3 = z2 - 0.5 * torch.cat([torch.zeros_like(grad), grad], axis=1) * self.dt
        return z3

    def forward(self, x):
        batch_size = x.shape[0]
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        result = torch.empty(batch_size, self.output_length, 32, 32).cuda()
        result[:, :3] = self.decoder(z[:, :256])
        for i in range(3, self.output_length, 3):
            z = self.integrate(z)
            result[:, i: i + 3] = self.decoder(z[:, :256])
        return result, mu, logvar

