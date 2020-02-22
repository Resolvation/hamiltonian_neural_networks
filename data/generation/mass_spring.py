from math import *
import numpy as np
from scipy.integrate import solve_ivp
import cv2


def f(t, y):
    return np.array([2 * y[1], -2 * y[0]])


for i in range(12000):
    r = np.random.uniform(0.1, 1)
    q = np.random.uniform(-sqrt(r), sqrt(r))
    p = sqrt(r - q * q)
    sol = solve_ivp(f, (0, 14.5), np.array([q, p]), t_eval=np.arange(0, 15, 0.5))

    for j, q in enumerate(sol.y[0]):
        img = np.full((64, 64, 3), 80, 'uint8')
        cv2.circle(img, (31, 31 + int(q * 16)), 8, (0, 255, 255), -1)
        img = cv2.blur(img, (4, 4))
        cv2.imwrite(f'../mass_spring/{i}_{j}.jpg', img)
