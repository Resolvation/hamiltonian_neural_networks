from math import *
import os

import cv2
import numpy as np
from scipy.integrate import solve_ivp

import torch

from .hamiltonain_dataset import HamiltonianDataset


class TwoBody(HamiltonianDataset):
    def __init__(self, n_samples, root='data'):
        super().__init__(n_samples, root)

        self.data_path = os.path.join(self.root, f'two_body_{n_samples}.tar')
        if not os.path.exists(self.data_path):
            self.generate()
        self.data = torch.load(self.data_path)

    def generate(self):
        def f(t, y):
            return np.hstack((y[4:], np.hstack((y[2: 4] - y[: 2], y[: 2] - y[2: 4]))))

        results = torch.empty(self.n_samples, 90, 32, 32)

        for i in range(self.n_samples):
            r = np.random.uniform(0.5, 1.5)
            q_diff = np.random.uniform(0.6, 1.4)
            alpha = np.random.uniform(0., 2 * pi)
            ln = sqrt(r + 1 / q_diff)
            q = np.array([sin(alpha), cos(alpha), sin(alpha + pi), cos(alpha + pi)]) * q_diff / 2
            p = np.array([cos(alpha), sin(alpha), -cos(alpha), -sin(alpha)]) * ln

            sol = solve_ivp(f, (0, 7.25), np.hstack((q, p)), t_eval=np.arange(0, 7.5, 0.25)).y

            sol += np.random.normal(scale=0.05, size=sol.shape)

            for j, q in enumerate(sol[:4].transpose()):
                img = np.full((32, 32, 3), 80, 'uint8')
                cv2.circle(img, (15 + int(q[0] * 16), 15 + int(q[1] * 16)), 2, (255, 255, 0), -1)
                cv2.circle(img, (15 + int(q[2] * 16), 15 + int(q[3] * 16)), 2, (255, 0, 0), -1)
                img = cv2.blur(img, (3, 3))
                results[i, 3 * j: 3 * j + 3] = torch.tensor(img, dtype=float).transpose(0, 2)

        results /= 255

        torch.save(results, self.data_path)
