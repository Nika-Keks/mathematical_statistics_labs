import numpy as np
from numpy.core.fromnumeric import partition
import scipy.stats as stats

data  = {
    "normal": {
        "args": (stats.norm.rvs, np.arange(-3., 3., 0.1), 0., 1.),
        "kwargs": {}},
    "cauchy": {
        "args": (stats.cauchy.rvs, np.arange(-10., 10., 0.1), 0., 1.),
        "kwargs": {}},
    "laplace": {
        "args": (stats.laplace.rvs, np.arange(-10., 10., 0.1), 0, 1. / np.sqrt(2)),
        "kwargs": {}},
    "poisson": {
        "args": (stats.poisson.rvs, np.arange(0., 20.)),
        "kwargs": {"mu": 10}},
    "uniform": {
        "args": (stats.uniform.rvs, np.arange(-2., 2., 0.1)),
        "kwargs": {"scale": 2 * np.sqrt(3), "loc": -np.sqrt(3)}
    }
}

def zR(x: np.ndarray) -> float:
    return (max(x) + min(x)) / 2.

def zQ(x: np.ndarray) -> float:
    return (np.percentile(x, 25) + np.percentile(x, 75)) / 2.

def ztr(x: np.ndarray) -> float:
    return stats.tmean(x, limits=(np.percentile(x, 25), np.percentile(x, 75)))

metrics = [np.mean, np.median, zR, zQ, ztr]

def calc_value(name:str, size: int, dist, x: np.array, *args, **kwargs):
    print(name + " " + str(size))
    values = [[] for i in range(5)]

    for i in range(1000):
        y = dist(size=size, **kwargs)
        for i, metric in enumerate(metrics):
            values[i].append(metric(y))
    
    for val in values:
        npval = np.array(val)
        print(f"{npval.mean():.4}", end=" ")
    print("")

    for val in values:
        npval = np.array(val)
        print(f"{npval.var():.4}", end=" ")
    print("")



if __name__ == "__main__":
    for name, params in data.items():
        for size in [10, 100, 1000]:
            calc_value(name, size, *(params["args"]), **(params["kwargs"]))