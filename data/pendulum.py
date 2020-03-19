from math import *
import os

import cv2
import numpy as np
from scipy.integrate import solve_ivp

import torch

from .hamiltonain_dataset import HamiltonianDataset


class Pendulum(HamiltonianDataset):
    def __init__(self, n_samples, root='data'):
        super().__init__(n_samples, root)

        self.data_path = os.path.join(self.root, f'pendulum_{n_samples}.tar')
        if not os.path.exists(self.data_path):
            self.generate()
        self.data = torch.load(self.data_path)

    def generate(self):
        def f(t, y):
            return 2 * y[1], -3 * sin(y[0])

        results = torch.empty(self.n_samples, 90, 32, 32)

        for i in range(self.n_samples):
            r = np.random.uniform(1.3, 2.3)
            q = np.random.uniform(-acos(1 - r / 3), acos(1 - r / 3))
            p = np.random.choice([-1, 1]) * sqrt(r - 3 * (1 - cos(q)))

            sol = solve_ivp(f, (0, 14.5), (q, p), t_eval=np.arange(0, 15, 0.5)).y

            sol += np.random.normal(scale=0.1, size=sol.shape)

            for j, (q, p) in enumerate(zip(sol[0], sol[1])):
                img = np.full((32, 32, 3), 80, 'uint8')
                cv2.circle(img, (15 + int(8 * cos(q)), 15 + int(8 * sin(q))), 3, (255, 255, 0), -1)
                img = cv2.blur(img, (3, 3))
                results[i, 3 * j: 3 * j + 3] = torch.tensor(img, dtype=float).transpose(0, 2)

        results /= 255

        torch.save(results, self.data_path)
