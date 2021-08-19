from numpy.core.fromnumeric import size
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from seaborn.distributions import distplot

data  = {
    "normal": {
        "args": (stats.norm.rvs, stats.norm.cdf, stats.norm.pdf, np.arange(-4., 4., 0.1), 0., 1.),
        "kwargs": {}},
    "cauchy": {
        "args": (stats.cauchy.rvs, stats.cauchy.cdf, stats.cauchy.pdf, np.arange(-4., 4., 0.1), 0., 1.),
        "kwargs": {}},
    "laplace": {
        "args": (stats.laplace.rvs, stats.laplace.cdf, stats.laplace.pdf, np.arange(-4., 4., 0.1), 0, 1. / np.sqrt(2)),
        "kwargs": {}},
    "poisson": {
        "args": (stats.poisson.rvs, stats.poisson.cdf, stats.poisson.pmf, np.arange(6., 14.)),
        "kwargs": {"mu": 10}},
    "uniform": {
        "args": (stats.uniform.rvs, stats.uniform.cdf, stats.uniform.pdf, np.arange(-4., 4., 0.1)),
        "kwargs": {"scale": 2 * np.sqrt(3), "loc": -np.sqrt(3)}
    }
}

def naplot(name: str, size: int, bwmarr: list, rvs, pdf, x: np.array, *args, **kwargs) -> None:
    """
    """
    fig, ax = plt.subplots(1, len(bwmarr))
    fig.suptitle(f"{name}, size={size}")

    for i, bwm in enumerate(bwmarr):
        sample = rvs(size=size, **kwargs)
        na = stats.gaussian_kde(sample, bw_method=bwm)
        ax[i].plot(x, na(x))
        ax[i].plot(x, pdf(x, *args, **kwargs))
        ax[i].set_title(f"h={bwm}")
    
    plt.savefig(os.path.join("plots", f"{name}_{size}"))
    plt.close()


def kdeplot(name: str, size: int, rvs, cdf, *args, **kwargs)-> None:
    """
    """
    sample = sorted(rvs(size=size, **kwargs))
    x = np.linspace(1 / len(sample), 1, len(sample))
    plt.step(sample, x, label="Empirical cumulative function")
    plt.plot(sample, cdf(sample, *args, **kwargs), label="Actual cumulative fanction")
    plt.legend()
    plt.savefig(os.path.join("plots", f"{name}_cum_{size}"))
    plt.close()


def plot(name: str, size: int, rvs, cdf, pdf, x, *args, **kwargs) -> None:
    """
    """
    naplot(name, size, [0.5, 1, 2], rvs, pdf, x, *args, **kwargs)
    kdeplot(name, size, rvs, cdf, *args, **kwargs)


if __name__ == "__main__":
    
    for name, params in data.items():
        for size in [20, 60, 100]:
            plot(name, size, *(params["args"]), **(params["kwargs"]))