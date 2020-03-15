from math import *
import os

import cv2
import numpy as np
from scipy.integrate import solve_ivp

import torch
from torch.utils.data import Dataset


class Pendulum(Dataset):
    def __init__(self, mode='vae', root='data', n_samples=1200, verbose=False):
        if mode not in {'vae', 'hnn'}:
            raise ValueError('Wrong mode.')
        self.mode = mode
        self.root = root
        self.n_samples = n_samples
        self.verbose = verbose

        self.path = os.path.join(self.root, f'pendulum_{n_samples}.tar')
        if not os.path.exists(self.path):
            self.generate()
        self.data = torch.load(self.path)

    def __getitem__(self, index):
        if self.mode == 'vae':
            return self.data[index // 30, index % 30]
        elif self.mode == 'hnn':
            return self.data[index].view(-1, 32, 32)

    def __len__(self):
        if self.mode == 'vae':
            return 30 * self.n_samples
        elif self.mode == 'hnn':
            return self.n_samples

    def generate(self):
        if self.verbose:
            print('Generating data.')

        def f(t, y):
            return 2 * y[1], -3 * sin(y[0])

        results = torch.empty(self.n_samples, 30, 3, 32, 32)

        for i in range(self.n_samples):
            r = np.random.uniform(1.3, 2.3)
            q = np.random.uniform(-acos(1 - r / 3), acos(1 - r / 3))
            p = np.random.choice([-1, 1]) * sqrt(r - 3 * (1 - cos(q)))

            q += np.random.normal(scale=sqrt(0.1))
            p += np.random.normal(scale=sqrt(0.1))

            sol = solve_ivp(f, (0, 14.5), (q, p), t_eval=np.arange(0, 15, 0.5))

            for j, (q, p) in enumerate(zip(sol.y[0], sol.y[1])):
                img = np.full((32, 32, 3), 80, 'uint8')
                cv2.circle(img, (15 + int(8 * sin(q)), 15 + int(8 * cos(q))), 3, (255, 255, 0), -1)
                img = cv2.blur(img, (3, 3))
                results[i, j] = torch.tensor(img, dtype=float).transpose(0, 2)

        results /= 255

        torch.save(results, self.path)

        if self.verbose:
            print('Data generated.')
