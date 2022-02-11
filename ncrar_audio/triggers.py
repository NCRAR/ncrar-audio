import numpy as np


def make_analog_trigger(n):
    trigger = np.zeros(n, dtype=stim.dtype)
    trigger[:100] = 1
    return trigger


def make_analog_trigger_cos(fs, n, pulse_frequency=128):
    n_samp = int(round((1 / pulse_frequency) * fs))
    t = np.arange(n_samp) / fs
    pulse = np.sin(2 * np.pi * pulse_frequency * t)

    trigger = np.zeros(n, dtype='double')
    trigger[:n_samp] = pulse
    return trigger
