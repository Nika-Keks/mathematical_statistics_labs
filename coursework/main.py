import shtRipper
import signal_processor as sp
import frequency_analyzer as fan

signal_num = 36
signal_name = 38515
process_signal_flag = "p"
analyze_frequency_flag = "a"


def run(*modes: str):
    # Extracting signal data
    signal, table = shtRipper.extract_sht("data", signal_name, [signal_num])
    sample_rate = 1000000
    if process_signal_flag in modes:
        sp.process_signal(signal[signal_num].copy(), 20, 0.0004, sample_rate)
    if analyze_frequency_flag in modes:
        fan.analyze_frequency(signal[signal_num].copy(), 0.167, 0.2, sample_rate)


run("a", "p")
