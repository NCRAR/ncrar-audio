import numpy as np


def _square_waveform(fs, duration=0.01):
    return np.ones(int(round(fs * duration)))


def _cos_waveform(fs, pulse_frequency=128, pulse_cycles=1, repeat=1):
    n_samp = int(round((1 / pulse_frequency) * fs)) * pulse_cycles
    t = np.arange(n_samp) / fs
    y = np.sin(2 * np.pi * pulse_frequency * t)
    y = np.tile(y, (repeat, 1))
    y[1:] *= 0.5
    return y.ravel()


def make_trigger(fs, n, repeat=1, iti=0.02, shape='square',
                 shape_settings=None):

    n_trig = int(round(fs * iti))
    trig = np.zeros(n_trig)
    if shape_settings is None:
        shape_settings = {}
    fn = globals()[f'_{shape}_waveform']
    w = fn(fs, **shape_settings)
    if len(w) > n_trig:
        raise ValueError('ITI too short')
    trig[:len(w)] = w

    trig = np.tile(trig, (repeat, 1))
    trig[1:] *= 0.5
    trig = trig.ravel()

    if len(trig) > n:
        raise ValueError('Trigger sequence is too long')

    waveform = np.zeros(n)
    waveform[:len(trig)] = trig
    return waveform


