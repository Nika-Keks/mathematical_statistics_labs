import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

data  = {
    "normal": {
        "args": (stats.norm.rvs, np.arange(-3., 3., 0.1), stats.norm.pdf, 0., 1.),
        "kwargs": {}},
    "cauchy": {
        "args": (stats.cauchy.rvs, np.arange(-10., 10., 0.1), stats.cauchy.pdf, 0., 1.),
        "kwargs": {}},
    "laplace": {
        "args": (stats.laplace.rvs, np.arange(-10., 10., 0.1), stats.laplace.pdf, 0, 1. / np.sqrt(2)),
        "kwargs": {}},
    "poisson": {
        "args": (stats.poisson.rvs, np.arange(0., 20.), stats.poisson.pmf),
        "kwargs": {"mu": 10}},
    "uniform": {
        "args": (stats.uniform.rvs, np.arange(-2., 2., 0.1), stats.uniform.pdf),
        "kwargs": {"scale": 2 * np.sqrt(3), "loc": -np.sqrt(3)}
    }
}


def plot(name: str, size: int, dist, x: np.array, f, *args, **kwargs):
    y = f(x, *args, **kwargs)
    plt.hist(dist(size=size, **kwargs), density=True)
    plt.plot(x, y)
    plt.grid()
    #plt.show()
    plt.savefig("plots/" + name + str(size))
    plt.close()


if __name__ == '__main__':
    for size in [10, 50, 1000]:
        for name, params in data.items():
            plot(name, size, *(params["args"]), **(params["kwargs"]))
