import numpy as np
from math import ceil
from scipy.signal import filtfilt
from scipy import signal


def get_mean(data: np.array) -> float:
    return sum(data) / len(data)


def getROI(data: np.array) -> np.array:
    mean = get_mean(data)
    rois = dict()
    begin, end = 0, 0
    for i in range(len(data)):
        if data[i] > mean:
            if begin == 0:
                begin = i
        elif data[i] < mean and begin != 0:
            end = i
            rois.update({end - begin: (begin, end)})
            begin, end = 0, 0
    longest_roi = max(rois.keys())
    return rois[longest_roi]


def rc_low_pass(x_new: float, y_old: float, sample_rate_hz: float, low_pass_cutoff_hz: int) -> float:
    dt = 1 / sample_rate_hz
    rc = 1 / (2 * np.pi * low_pass_cutoff_hz)
    alpha = dt / (rc + dt)
    y_new = x_new * alpha + (1 - alpha) * y_old
    return y_new


def rc_high_pass(x_new: float, x_old: float, y_old: float, sample_rate_hz: float, high_pass_cutoff_hz: int) -> float:
    dt = 1 / sample_rate_hz
    rc = 1 / (2 * np.pi * high_pass_cutoff_hz)
    alpha = rc / (rc + dt)
    y_new = alpha * (y_old + x_new - x_old)
    return y_new


def signal_filter(xs: np.array, sample_rate_hz: int, cutoff_hz: int, mode: str) -> np.array:
    x_prev = 0
    y_prev = 0
    filtered = np.zeros(len(xs))
    for i in range(len(xs)):
        if mode == "high":
            y_prev = rc_high_pass(xs[i], x_prev, y_prev, sample_rate_hz, cutoff_hz)
        elif mode == "low":
            y_prev = rc_low_pass(xs[i], y_prev, sample_rate_hz, cutoff_hz)
        else:
            print("ERROR: Wrong filter mode!")
            exit(1)
        x_prev = xs[i]
        filtered[i] = y_prev
    return np.array(filtered)


def DDF(data: np.array, rank: int) -> np.array:
    filtered = np.zeros(len(data))
    for i in range(len(data)):
        filtered[i] = 1 / (rank * (rank + 1))
        tmp_sum = 0
        for j in range(rank):
            if i + j < len(data) and i - j >= 0:
                tmp_sum += data[i + j] - data[i - j]
        filtered[i] *= tmp_sum
    return filtered


def get_local_max_sequence(data: np.array, window_size: int) -> np.array:
    max_seq = np.zeros(ceil(len(data) / window_size))
    for i in range(len(max_seq)):
        right_border = (i + 1) * window_size if (i + 1) * window_size <= len(data) else len(data) - 1
        max_seq[i] = max(data[i * window_size:right_border])
    return max_seq


def get_roots(data: np.array, left_border: int, right_border: int, sample_rate: int,
              threshold: float = 0.1) -> np.array:
    roots = list()
    for x, i in zip(data, range(left_border, right_border)):
        if abs(x) < threshold:
            roots.append(i)

    tmp_roots = []
    min_roots = []

    empiric_threshold = 1e-6 + 1.72e-7
    for root in roots:
        tmp_root = root / sample_rate
        if len(tmp_roots) == 0:
            tmp_roots.append(tmp_root)
        if abs(tmp_roots[-1] - tmp_root) < empiric_threshold:

            tmp_roots.append(tmp_root)
        else:
            min_roots.append(int(min(tmp_roots) * sample_rate))
            tmp_roots = []
    if tmp_roots:
        min_roots.append(int(min(tmp_roots) * sample_rate))
    return np.array(min_roots)


def get_instant_frequencies(roots: np.array, sample_rate: int, filtered: bool = False) -> np.array:
    frequencies = np.zeros(len(roots) - 1)
    for i in range(len(frequencies)):
        frequencies[i] = 1 / ((roots[i + 1] - roots[i]) / sample_rate)
    # b, a = signal.butter(2, 0.250)
    return filtfilt(*signal.butter(2, 0.250), frequencies) if filtered else frequencies
