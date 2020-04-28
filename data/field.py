from math import *
import os

import cv2
import numpy as np
from scipy.integrate import solve_ivp

import torch

from .hamiltonain_dataset import HamiltonianDataset


class Field(HamiltonianDataset):
    def __init__(self, n_samples, root='data'):
        super().__init__(n_samples, root)

        self.data_path = os.path.join(self.root, f'field_{n_samples}.tar')
        if not os.path.exists(self.data_path):
            self.generate()
        self.data = torch.load(self.data_path)

    def generate(self):
        def f(t, y):
            return np.hstack((y[2:], 0, -y[1]))

        results = torch.empty(self.n_samples, 90, 32, 32)

        for i in range(self.n_samples):
            r = np.random.uniform(0.1, 0.2)
            q = np.array([0, 0])
            alpha = np.random.uniform(0.2 * pi, 0.4 * pi)
            p = np.array([cos(alpha), sin(alpha)]) * sqrt(2 * r)

            sol = solve_ivp(f, (0, 3.625), np.hstack((q, p)), t_eval=np.arange(0, 3.75, 0.125)).y

            sol += np.random.normal(scale=0.005, size=sol.shape)

            for j, q in enumerate(sol[:2].transpose()):
                print(q)
                img = np.full((32, 32, 3), 80, 'uint8')
                cv2.circle(img, (int(32 * q[0]), int(32 * (1 - q[1]))), 2, (255, 255, 0), -1)
                img = cv2.blur(img, (3, 3))
                cv2.imshow('', img)
                cv2.waitKey(300)
                results[i, 3 * j: 3 * j + 3] = torch.tensor(img, dtype=float).transpose(0, 2)

        results /= 255

        torch.save(results, self.data_path)
