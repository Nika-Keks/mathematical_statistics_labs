import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
from scipy import stats as stats
from matplotlib import pyplot as plt
from scipy.stats.mstats_basic import siegelslopes
import os


def LSM(x: np.array, y: np.array) -> tuple:
    b1 = ((x*y).mean() - x.mean() * y.mean()) / ((x**2).mean() - x.mean()**2)
    b0 = y.mean() - x.mean() * b1
    Y_LSM = lambda x_: b0 + b1 * x_
    return Y_LSM, b0, b1

def LMM(x: np.array, y: np.array) -> tuple:
    n = len(x)
    rQ = 1 / n * sum((np.sign(x - np.median(x)) * np.sign(y - np.median(y))))
    l = n // 4 + (0 if n % 4 == 0 else 1)
    j = n - l + 1
    b1 = rQ * (y[j] - y[l]) / (x[j] - x[l])
    b0 = np.median(y) - b1 * np.median(x)
    Y_LMM = lambda x_: b0 + b1 * x_
    return Y_LMM, b0, b1

def plot(x: np.array, y_noise: np.array, y_model: np.array, y_lsm: np.array, y_lmm: np.array, title: str = "", name: str = "plot") -> None:
    plt.grid()
    plt.title(title)
    plt.plot(x, y_noise, "ko", label="Элементы выборки")
    plt.plot(x, y_model, "b", label="Эталонная зависимость")
    plt.plot(x, y_lsm, "r", label="МНК")
    plt.plot(x, y_lmm, "g", label="МНМ")
    plt.legend()
    plt.savefig(os.path.join("plots", name))
    plt.close()

sample_size = 20
segment = np.linspace(-1.8, 2., sample_size)
noise = stats.norm.rvs(size=sample_size)
disturbance = np.array([10.] + [0. for _ in range(sample_size - 2)] + [-10.])
a = 2.
b = 2.

y_model = lambda x: a + b * x
y_noise = lambda x: a + b * x + noise
y_disturbance = lambda x: a + b * x + noise + disturbance


if __name__ == "__main__":
    x = np.array(segment)
    
    lsm = LSM(x, y_noise(x))
    lmm = LMM(x, y_noise(x))
    plot(x, y_noise(x), y_model(x), lsm[0](x), lmm[0](x), "Выборка без возмущений", "default")

    lsmd = LSM(x, y_disturbance(x))
    lmmd = LMM(x, y_disturbance(x))
    plot(x, y_disturbance(x), y_model(x), lsmd[0](x), lmmd[0](x), "Выборка с возмущениями", "altered")

    print(lsm[1:])
    print(lmm[1:])
    print("")
    print(lsmd[1:])
    print(lmmd[1:])
