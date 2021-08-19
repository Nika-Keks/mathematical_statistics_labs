import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os


data  = {
    "normal": {
        "args": (stats.norm.rvs, ),
        "kwargs": {}},
    "cauchy": {
        "args": (stats.cauchy.rvs, ),
        "kwargs": {}},
    "laplace": {
        "args": (stats.laplace.rvs, ),
        "kwargs": {}},
    "poisson": {
        "args": (stats.poisson.rvs, ),
        "kwargs": {"mu": 10}},
    "uniform": {
        "args": (stats.uniform.rvs, ),
        "kwargs": {"scale": 2 * np.sqrt(3), "loc": -np.sqrt(3)}
    }
}


def plot(name: str, sizes: list, dist, *args, **kwargs) -> None:
    """
    """
    fig, axs = plt.subplots(len(sizes))
    fig.suptitle(name)

    for i, size in enumerate(sizes):
        sns.boxplot(ax=axs[i], orient="h", data=pd.DataFrame({str(size): dist(size=size, **kwargs)}))
    
    plt.savefig(os.path.join("plots", name))
    plt.close()


def shareofemis(name, sizes:list, dist, *args, **kwargs) -> list:
    """
    """
    meanemisarr = {}
    for size in sizes:
        emisarr = []
        for _ in range(1000):
            y = dist(size=size, **kwargs)
            y25 = np.percentile(y, 25)
            y75 = np.percentile(y, 75)
            nonemis = (y25 - 1.5 * (y75 - y25), y75 + 1.5 * (y75 - y25))
            share = 0
            for yi in y:
                if not (nonemis[0] <= yi <= nonemis[1]):
                    share += 1
            emisarr.append(share / size)
        
        meanemisarr[size] = np.mean(emisarr)
    
    return meanemisarr


if __name__ == "__main__":

    for name, params in data.items():
        plot(name, [20, 100], *(params["args"]), **(params["kwargs"]))

        print(name)
        print(shareofemis(name, [20, 100], *(params["args"]), **(params["kwargs"])))