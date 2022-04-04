import numpy as np


def make_analog_trigger(fs, n, duration=0.1):
    n_samp = int(round(duration * fs))
    trigger = np.zeros(n)
    trigger[:n_samp] = 1
    return trigger


def make_analog_trigger_cos(fs, n, pulse_frequency=128, pulse_cycles=1):
    n_samp = int(round((1 / pulse_frequency) * fs)) * pulse_cycles
    t = np.arange(n_samp) / fs
    pulse = np.sin(2 * np.pi * pulse_frequency * t)

    trigger = np.zeros(n, dtype='double')
    trigger[:n_samp] = pulse
    return trigger
