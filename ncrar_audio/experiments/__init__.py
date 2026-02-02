#import time
#
#import matplotlib.pyplot as plt
#import numpy as np
#from scipy import signal
#from tqdm import tqdm
#
#from pathlib import Path
#from psiaudio.calibration import FlatCalibration
#from psiaudio import stim
#from psiaudio.queue import BlockedFIFOSignalQueue
#
##from ncrar_audio.babyface import Babyface
##from ncrar_audio.cpod import CPod
#from ncrar_audio import triggers
#
#
#def make_click(fs, amplitude, duration):
#    samples = int(round(fs * duration))
#    return amplitude * np.ones(samples)
#
#
#def make_tone_pip(fs, amplitude, frequency, duration):
#    samples = int(round(fs * duration))
#    time = np.arange(samples) / fs
#    tone = amplitude * np.cos(2 * np.pi * frequency * time)
#    envelope = signal.windows.blackmanharris(samples)
#    return tone * envelope
#
#
#def load_wav(fs, amplitude, wavfile):
#    waveform = amplitude * stim.load_wav(fs, wavfile)
#    if waveform.ndim != 1:
#        raise ValueError('Not designed to handle N-dimensional waveforms')
#    return waveform
#
