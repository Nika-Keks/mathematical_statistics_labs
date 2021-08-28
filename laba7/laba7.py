from typing_extensions import TypeAlias
import numpy as np
from numpy.lib.nanfunctions import _nanprod_dispatcher
import pandas as pd
from scipy import stats as stats 
from matplotlib import pyplot as plt, scale
import os

def mk_param_estim(x: np.array) -> tuple:
    xmean = x.mean()
    xdisp = ((x - xmean)**2).mean()
    return xmean, xdisp

def mk_x2_table(x: np.array, segments: list, cdf, quantile: float):
    print(f"X^2 quantile:{quantile}")
    ndelta = len(segments)
    n = len(x)
    niarr = np.array([0 for _ in range(ndelta)])
    piarr = np.array([0. for _ in range(ndelta)])

    for i, (lseg, rseg) in enumerate(segments):
        piarr[i] = cdf(rseg) - cdf(lseg)

    for xi in x:
        for i, (lseg, rseg) in enumerate(segments):
            if lseg < xi <= rseg:
                niarr[i] += 1

    for i in range(ndelta):
        print(f"[{segments[i][0]:.1f}, {segments[i][1]:.1f}] | {niarr[i]:.4f} | {piarr[i]:.4f} | {(n * piarr[i]):.4f} | {(niarr[i] - n * piarr[i]):.4f} | {((niarr[i] - n * piarr[i])**2 / (n * piarr[i])):.4f}")
    print(f"[{segments[0][0]:.1f}, {segments[-1][1]:.1f}] | {sum(niarr):.4f} | {sum(piarr):.4f} | {sum(n * piarr):.4f} | {sum(niarr - n * piarr):.4f} | {sum((niarr - n * piarr)**2 / (n * piarr)):.4f}")

def points2segs(x: np.array) -> list:
    segs = [(float("-inf"), x[0])]
    for i in range(len(x) - 1):
        segs.append((x[i], x[i + 1]))
    segs.append((x[-1], float("inf")))

    return segs

def mk_table(x:np.array, points: np.array, paramcdf, quantile: float, *args, **kwargs):
    segments = points2segs(points)
    cdf = lambda x: paramcdf(x, *args, **kwargs)
    mk_x2_table(x, segments, cdf, quantile)

def mk_table(x: np.array, points: np.array, quantile: float):
    segments = points2segs(points)
    xmean, xdisp = mk_param_estim(x)
    cdf = lambda x_: stats.norm.cdf(x_, loc=xmean, scale=xdisp)
    mk_x2_table(x, segments, cdf, quantile)


datadict = {
    "norm": {
        "x": stats.norm.rvs(size=100),
        "points": np.linspace(-1., 1., 7),
        "quantile": 12.59
    },
    "uniform": {
        "x": stats.uniform.rvs(size=20, scale=3., loc=1.5),
        "points": np.linspace(-1., 1, 5),
        "quantile": 9.49
    },
    "laplace":{
        "x": stats.laplace.rvs(size=20),
        "points": np.linspace(-1., 1., 5),
        "quantile": 9.49
    }
}

if __name__ == "__main__":

    for name, params in datadict.items():
        print(name)
        mk_table(**params)
        print("_"*40)
