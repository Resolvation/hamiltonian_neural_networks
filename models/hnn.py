import torch
import torch.nn as nn


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.view(input.shape[0], -1)


class UnFlatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.view(input.shape[0], 16, 4, 4)


class Merge(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, qp):
        return torch.cat(qp, dim=1)


class HNN(nn.Module):
    def __init__(self, input_length=90, output_length=90, dt=0.125):
        super().__init__()
        self.input_length = input_length
        self.output_length = output_length
        self.dt = dt

        self.encoder = nn.Sequential(
            # b input_length 32 32
            nn.Conv2d(input_length, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            # b 64 16 16
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            # b 64 8 8
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            # b 32 4 4
            Flatten()
            # b 512
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
        )

        self.hamiltonian = nn.Sequential(
            Merge(),
            nn.Linear(512, 512),
            nn.Softplus(),
            nn.Linear(512, 256),
            nn.Softplus(),
            nn.Linear(256, 128),
            nn.Softplus(),
            nn.Linear(128, 128),
            nn.Softplus(),
            nn.Linear(128, 1)
        )

        self.h1 = lambda z: self.hamiltonian[1](self.hamiltonian[0](z))
        self.h2 = lambda z: self.hamiltonian[3](self.hamiltonian[2](self.h1(z)))
        self.h3 = lambda z: self.hamiltonian[5](self.hamiltonian[4](self.h2(z)))
        self.h4 = lambda z: self.hamiltonian[7](self.hamiltonian[6](self.h3(z)))

        self.decoder = nn.Sequential(
            # b 256
            UnFlatten(),
            # b 16 4 4
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            # b 16 8 8
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            # b 16 16 16
            nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            # b 16 16 16
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
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
        return z[:, :256], z[:, 256:], mu, logvar

    def ham_grad(self, z):
        return ((self.hamiltonian[1].weight.t()[None] * torch.sigmoid(self.h1(z))[:, None])
                @ (self.hamiltonian[3].weight.t()[None] * torch.sigmoid(self.h2(z))[:, None])
                @ (self.hamiltonian[5].weight.t()[None] * torch.sigmoid(self.h3(z))[:, None])
                @ (self.hamiltonian[7].weight.t()[None] * torch.sigmoid(self.h4(z))[:, None])
                @ self.hamiltonian[9].weight.t()[None])[:, :, 0]

    def integrate(self, q, p):
        grad = self.ham_grad((q, p))[:, 256:]
        p1 = p - 0.5 * grad * self.dt
        grad = self.ham_grad((q, p1))[:, :256]
        q1 = q + grad * self.dt
        grad = self.ham_grad((q1, p1))[:, 256:]
        p2 = p1 - 0.5 * grad * self.dt
        return q1, p2

    def forward(self, x):
        batch_size = x.shape[0]
        h = self.encoder(x)
        q, p, mu, logvar = self.bottleneck(h)
        result = torch.empty(batch_size, self.output_length, 32, 32).cuda()
        result[:, :3] = self.decoder(q)
        for i in range(3, self.output_length, 3):
            q, p = self.integrate(q, p)
            result[:, i: i + 3] = self.decoder(q)
        return result, mu, logvar

