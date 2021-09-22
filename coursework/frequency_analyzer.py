import shtRipper
import signal_utils as sgu
import numpy as np
import matplotlib.pyplot as plt


def analyze_frequency(tmp_signal: dict, left_border: float, right_border: float, sample_rate: int):
    # Getting region of interest (ROI) with detected sawtooth
    tmp_signal["tMax"] = right_border
    tmp_signal["tMin"] = left_border
    tmp_signal["#ch"] = int((right_border - left_border) * sample_rate)
    tmp_signal["data"] = tmp_signal["data"][int(left_border * sample_rate):int(right_border * sample_rate)]
    shtRipper.plot_hist(tmp_signal)

    # Processing high-pass filtering
    print("Processing high-pass filtering")
    tmp_signal["data"] = sgu.signal_filter(tmp_signal["data"], sample_rate, 250, "high")
    print("Done")
    shtRipper.plot_hist(tmp_signal)

    # Processing low-pass filtering
    print("Processing low-pass filtering")
    tmp_signal["data"] = sgu.signal_filter(tmp_signal["data"], sample_rate, 200, "low")
    print("Done")
    roots = sgu.get_roots(tmp_signal["data"], int(tmp_signal["tMin"] * sample_rate),
                          int(tmp_signal["tMax"] * sample_rate), sample_rate)
    shtRipper.plot_hist(tmp_signal, show=False)
    plt.scatter([root / sample_rate for root in roots], [0 * i for i in range(len(roots))], c="red")
    plt.show()

    # Getting frequency function
    print("Getting frequency function")
    fs = sgu.get_instant_frequencies(roots, sample_rate, filtered=True)
    xs = np.linspace(tmp_signal["tMin"], tmp_signal["tMax"], len(fs))
    print("Done")
    plt.plot(xs, fs)
    plt.xlabel("time (s)")
    plt.ylabel("frequency (Hz)")
    plt.scatter(xs, fs)
    plt.grid()
    plt.show()
