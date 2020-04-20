from math import *
import os

import cv2
import numpy as np
from scipy.integrate import solve_ivp

import torch

from .hamiltonain_dataset import HamiltonianDataset


class MassSpring(HamiltonianDataset):
    def __init__(self, n_samples, root='data'):
        super().__init__(n_samples, root)

        self.data_path = os.path.join(self.root, f'mass_spring_{n_samples}.tar')
        if not os.path.exists(self.data_path):
            self.generate()
        self.data = torch.load(self.data_path)

    def generate(self):
        def f(t, y):
            return 2 * y[1], -2 * y[0]

        results = torch.empty(self.n_samples, 90, 32, 32)

        for i in range(self.n_samples):
            r = np.random.uniform(0.1, 1)
            q = np.random.uniform(-sqrt(r), sqrt(r))
            p = np.random.choice([-1, 1]) * sqrt(r - q * q)

            sol = solve_ivp(f, (0, 7.25), (q, p), t_eval=np.arange(0, 7.5, 0.25)).y

            sol += np.random.normal(scale=0.1, size=sol.shape)

            for j, q in enumerate(sol[0]):
                img = np.full((32, 32, 3), 80, 'uint8')
                cv2.circle(img, (15 + int(q * 8), 15), 3, (255, 255, 0), -1)
                img = cv2.blur(img, (3, 3))
                results[i, 3 * j: 3 * j + 3] = torch.tensor(img, dtype=float).transpose(0, 2)

        results /= 255

        torch.save(results, self.data_path)
