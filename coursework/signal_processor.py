import shtRipper
import signal_utils as sgu
import numpy as np
import matplotlib.pyplot as plt


def process_signal(tmp_signal: dict, ddf_order: int, up_border: float, sample_rate: int):

    # Drawing raw signal plot
    shtRipper.plot_hist(tmp_signal)

    # Getting region of interest (ROI)
    print("Getting region of interest (ROI)")
    borders = sgu.getROI(tmp_signal["data"])
    tmp_signal["tMax"] = borders[1] / sample_rate
    tmp_signal["tMin"] = borders[0] / sample_rate
    tmp_signal["#ch"] = borders[1] - borders[0]
    tmp_signal["data"] = tmp_signal["data"][borders[0]:borders[-1]]
    print("Done")
    shtRipper.plot_hist(tmp_signal, show=False)
    plt.plot((0.167, 0.167), (0, 0.5), c="red", linewidth=2)
    plt.plot((0.2, 0.2), (0, 0.5), c="red", linewidth=2)
    plt.show()

    # Processing high-pass filtering
    print("Processing high-pass filtering")
    tmp_signal["data"] = sgu.signal_filter(tmp_signal["data"], sample_rate, 625, "high")
    print("Done")
    shtRipper.plot_hist(tmp_signal)

    # Counting 1'st derivative (with digital derivative filter)
    print("Processing digital derivative filtering")
    tmp_signal["data"] = sgu.DDF(tmp_signal["data"], ddf_order)
    print("Done")
    shtRipper.plot_hist(tmp_signal)

    # Getting absolute values of 1'st derivative
    print("Getting absolute values of 1'st derivative")
    tmp_signal["data"] = np.array([abs(x) for x in tmp_signal["data"]])
    print("Done")
    shtRipper.plot_hist(tmp_signal)

    # Processing low-pass filtering
    print("Processing low-pass filtering")
    tmp_signal["data"] = sgu.signal_filter(tmp_signal["data"], sample_rate, 5000, "low")
    print("Done")
    shtRipper.plot_hist(tmp_signal, show=False)
    plt.plot((0.167, 0.167), (0, 0.0025), c="red", linewidth=2)
    plt.plot((0.2, 0.2), (0, 0.0025), c="red", linewidth=2)
    plt.plot((tmp_signal["tMin"], tmp_signal["tMax"]), (up_border, up_border))
    plt.show()

    # Optional, getting sequence of local max values
    print("Getting sequence of local max values")
    window_size = 500
    tmp_signal["data"] = sgu.get_local_max_sequence(tmp_signal["data"], window_size)
    tmp_signal["#ch"] /= window_size
    print("Done")
    shtRipper.plot_hist(tmp_signal, scatter=True, show=False)
    plt.plot((0.167, 0.167), (0, 0.0025), c="red", linewidth=2)
    plt.plot((0.2, 0.2), (0, 0.0025), c="red", linewidth=2)
    plt.show()
