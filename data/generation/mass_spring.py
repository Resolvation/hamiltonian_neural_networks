from math import *
from tqdm import tqdm
import numpy as np
import torch
from scipy.integrate import solve_ivp
import cv2


N_samples = 1200


def f(t, y):
    return 2 * y[1], -2 * y[0]


results = torch.empty(N_samples, 30, 3, 32, 32)

for i in tqdm(range(N_samples)):
    r = np.random.uniform(0.1, 1)
    q = np.random.uniform(-sqrt(r), sqrt(r))
    p = np.random.choice([-1, 1]) * sqrt(r - q * q)
    sol = solve_ivp(f, (0, 14.5), (q, p), t_eval=np.arange(0, 15, 0.5))

    for j, q in enumerate(sol.y[0]):
        img = np.full((32, 32, 3), 80, 'uint8')
        cv2.circle(img, (15 + int(q * 8), 15), 3, (255, 255, 0), -1)
        img = cv2.blur(img, (3, 3))
        results[i, j] = torch.tensor(img, dtype=float).transpose(0, 2)

results /= 255

torch.save(results, '../mass_spring.tar')
