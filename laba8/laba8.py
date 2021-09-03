from genericpath import samestat
from re import sub
import numpy as np
from numpy.core.records import array
from scipy import stats
from scipy import signal
from matplotlib import pyplot as plt
import os

start = 1024 * 21
stop = start + 1024
file = open(os.path.join("data", "wave_ampl.txt"), "r")
data = eval(file.read())
data = np.array(data[start:stop])
data = signal.medfilt(data, kernel_size=3)

col_hight, col_border, _ = plt.hist(data)
list_hight = list(col_hight)
plt.grid()
plt.savefig(os.path.join("plots", "hist"))
plt.close()

fon_i = list_hight.index(max(list_hight))
list_hight[fon_i] = 0
signal_i = list_hight.index(max(list_hight))
print(f"{fon_i}\n{signal_i}")

in_fon = lambda x: col_border[fon_i] < x <= col_border[fon_i + 1]
in_signal = lambda x: col_border[signal_i] < x <= col_border[signal_i + 1]

class MiniPlot():

    x: list = None
    y: list = None
    color: str = None
    segment: str = None

    def __init__(self, color: str, segment: str) -> None:
        self.color = color
        self.segment = segment
        self.x = []
        self.y = []

    def append(self, x: int, y: float):
        self.x.append(x)
        self.y.append(y)

    def toplot(self) -> None:
        plt.plot(self.x, self.y, self.color, label=self.segment)

    def Is(self, segment: str) -> bool:
        return self.segment == segment 


in_seg = lambda x, i: col_border[i] < x <= col_border[i + 1]
in_fon = lambda x: in_seg(x, fon_i)
in_signal = lambda x: in_seg(x, signal_i)


segment_data = [('g', "фон"), ('b' , "переход"), ('r', "сигнал"), ('b', "переход"), ('g', "фон")]
pdata = [MiniPlot(*segment_data[0])]


for x, y in enumerate(data):
    if (in_fon(y) and not pdata[-1].Is("фон")) or \
        (in_signal(y) and not pdata[-1].Is("сигнал")) or \
        (not pdata[-1].Is("переход") and (not in_fon(y)) and (not in_signal(y))):
            pdata.append(MiniPlot(*segment_data[len(pdata) % 5]))
    
    pdata[-1].append(x, y)
    
for miniplot in pdata:
    miniplot.toplot()

plt.grid()
plt.savefig(os.path.join("plots", "plot")) 


def intra_group_dispersion(sample: np.array, group_count: int):
    dispersions = np.zeros(group_count)
    for i in range(group_count):
        for j in range(int(len(sample) / group_count)):
            dispersions[i] += (sample[i * group_count + j] - np.mean(
                sample[i * group_count:int(i * group_count + len(sample) / group_count)])) ** 2 / (group_count - 1)
    #print(np.mean(dispersions))
    return np.mean(dispersions)


def inter_group_dispersion(sample: np.array, group_count: int):
    group_means = np.zeros(group_count)
    for i in range(group_count):
        group_means[i] += np.mean(sample[i * group_count:int(i * group_count + len(sample) / group_count)])
    general_mean = np.mean(group_means)
    dispersion = 0
    for i in range(group_count):
        dispersion += (group_means[i] - general_mean) ** 2
    #print(group_count * dispersion / (group_count - 1))
    return group_count * dispersion / (group_count - 1)


def fisher_criteria(sample: np.array, group_count: int):
    return inter_group_dispersion(sample, group_count) / intra_group_dispersion(sample, group_count)


for miniplot in pdata:
    y = miniplot.y
    if len(y) < 10:
        continue
    print(miniplot.segment, end="   ")
    print(fisher_criteria(np.array(y), 6))