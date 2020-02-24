from math import *
from tqdm import tqdm
import numpy as np
from scipy.integrate import solve_ivp
import cv2


def f(t, y):
    return 2 * y[1], -3 * sin(y[0])


for i in tqdm(range(12000)):
    r = np.random.uniform(1.3, 2.3)
    q = np.random.uniform(-acos(1 - r / 3), acos(1 - r / 3))
    p = np.random.choice([-1, 1]) * sqrt(r - 3 * (1 - cos(q)))
    sol = solve_ivp(f, (0, 14.5), (q, p), t_eval=np.arange(0, 15, 0.5))

    for j, (q, p) in enumerate(zip(sol.y[0], sol.y[1])):
        img = np.full((64, 64, 3), 80, 'uint8')
        cv2.circle(img, (31 + int(16 * sin(q)), 31 + int(16 * cos(q))), 8, (0, 255, 255), -1)
        img = cv2.blur(img, (4, 4))
        cv2.imwrite(f'../pendulum/{i}_{j}.jpg', img)
