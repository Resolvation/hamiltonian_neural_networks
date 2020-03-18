from math import *
import os

import cv2
import numpy as np
from scipy.integrate import solve_ivp

import torch
from torch.utils.data import Dataset


class TwoBody(Dataset):
    def __init__(self, mode='vae', root='data', n_samples=1200, verbose=False):
        if mode not in {'vae', 'hnn'}:
            raise ValueError('Wrong mode.')
        self.mode = mode
        self.root = root
        self.n_samples = n_samples
        self.verbose = verbose

        self.path = os.path.join(self.root, f'two_body_{n_samples}.tar')
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
            return np.hstack((y[4:], np.hstack((y[2: 4] - y[: 2], y[: 2] - y[2: 4]))))

        results = torch.empty(self.n_samples, 30, 3, 32, 32)

        for i in range(self.n_samples):
            r = np.random.uniform(0.5, 1.5)
            q_diff = np.random.uniform(0.6, 1.4)
            alpha = np.random.uniform(0., 2 * pi)
            ln = sqrt(r + 1 / q_diff)

            q = np.array([sin(alpha), cos(alpha), sin(alpha + pi), cos(alpha + pi)]) * q_diff / 2
            p = np.array([cos(alpha), sin(alpha), -cos(alpha), -sin(alpha)]) * ln

            sol = solve_ivp(f, (0, 14.5), np.hstack((q, p)), t_eval=np.arange(0, 15, 0.5))

            sol.y += np.random.normal(scale=0.05, size=sol.y.shape)

            for j, q in enumerate(sol.y[:4].transpose()):
                img = np.full((32, 32, 3), 80, 'uint8')
                cv2.circle(img, (15 + int(q[0] * 16), 15 + int(q[1] * 16)), 2, (255, 255, 0), -1)
                cv2.circle(img, (15 + int(q[2] * 16), 15 + int(q[3] * 16)), 2, (255, 0, 0), -1)
                img = cv2.blur(img, (3, 3))
                results[i, j] = torch.tensor(img, dtype=float).transpose(0, 2)

        results /= 255

        torch.save(results, self.path)

        if self.verbose:
            print('Data generated.')
