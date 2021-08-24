from numpy.core.fromnumeric import var
from scipy import stats as stats
import numpy as np
from matplotlib import pyplot as plt
import os
from ellipse import Ellipse

def mkcov(sig: tuple, ro: float) -> list:
    cov = [
        [sig[0]**2, ro * sig[0] * sig[1]], 
        [ro * sig[0] * sig[1], sig[1]**2]
    ]
    return cov

def plot(size: int, ro: float, mean: tuple, sig: tuple):
    cov = mkcov(sig=sig, ro=ro)
    sample = stats.multivariate_normal.rvs(size=size, cov=cov, mean=mean)
    const = max([(-(x - mean[0])**2 / sig[0]**2 * (ro**2 - 1)) for x in sample[:, 0]])
    ellipse = Ellipse(mean=mean, disp=sig, cov=ro, C=const)
    g = ellipse.gragh()
    plt.plot(sample[:, 0], sample[:, 1], "bo")
    plt.plot(g[0, :], g[1, :], "r")
    plt.grid()
    plt.savefig(os.path.join("plots", f"{size}_{ro}".replace(".", "")))
    plt.close()


coefs = [
    lambda x, y: stats.stats.pearsonr(x, y)[0], 
    lambda x, y: stats.stats.spearmanr(x, y)[0], 
    lambda x, y: stats.linregress(x, y)[2]**2
]

variables = [
        np.mean,
        lambda x: np.mean(x**2),
        np.var
]

def printcoefs(size: int, rvs):
    
    vararr = [[] for _ in range(len(coefs))]

    for _ in range(1000):
        sample =rvs(size)
        for i, ceaf in enumerate(coefs):
            vararr[i].append(ceaf(sample[:, 0], sample[:, 1]))
    
    vararr = np.array(vararr)
    for i in range(len(coefs)):
        for variable in variables:
            print(variable(vararr[i]), end="  ")
        print("")
    print("")
    print("")  



sig1 = (1., 1.)
sig2 = (10., 10.)
ro1 = 0.9
ro2 = -0.9
mean0 = (0., 0.)

cov1 = mkcov(sig=sig1, ro=ro1)
cov2 = mkcov(sig=sig2, ro=ro2)

dists = [
    lambda size: stats.multivariate_normal.rvs(size=size, cov=cov1, mean=mean0),
    lambda size: 0.9*stats.multivariate_normal.rvs(size=size, cov=cov1, mean=mean0) + 0.1*stats.multivariate_normal.rvs(size=size, cov=cov2, mean=mean0)
]

if __name__ == "__main__":  
    sig = (1., 1.)
    mean = (0., 0.)
    
    for size in [20, 60, 100]:
        for ro in [0., 0.5, 0.9]:
            plot(size=size, ro=ro, mean=mean, sig=sig)
            for dist in dists:
                print(f"{size}_{ro}")
                printcoefs(size, dist)

    