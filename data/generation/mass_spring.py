from math import *
from tqdm import tqdm
import numpy as np
from scipy.integrate import solve_ivp
import cv2


def f(t, y):
    return 2 * y[1], -2 * y[0]


for i in tqdm(range(12000)):
    r = np.random.uniform(0.1, 1)
    q = np.random.uniform(-sqrt(r), sqrt(r))
    p = np.random.choice([-1, 1]) * sqrt(r - q * q)
    sol = solve_ivp(f, (0, 14.5), (q, p), t_eval=np.arange(0, 15, 0.5))

    for j, q in enumerate(sol.y[0]):
        img = np.full((32, 32, 3), 80, 'uint8')
        cv2.circle(img, (15, 15 + int(q * 8)), 3, (0, 255, 255), -1)
        img = cv2.blur(img, (3, 3))
        cv2.imwrite(f'../mass_spring/{i}_{j}.jpg', img)
