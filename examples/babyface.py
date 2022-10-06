import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from psiaudio.calibration import FlatCalibration
from psiaudio.stim import ramped_tone
from psiaudio import util

from ncrar_audio.babyface import Babyface


def play_tones():
    calibration = FlatCalibration.unity()

    device = Babyface()
    frequencies = np.array([1e3, 2e3])[:, np.newaxis]
    tones = ramped_tone(device.fs, frequency=frequencies, duration=5,
                       rise_time=0.1, level=-18.94, calibration=calibration)
    recording = device.acquire(tones, input_channels=2)
    recording = signal.detrend(recording)
    psd = util.db(util.psd_df(recording, device.fs))

    # The 1: slice along the columns axis of the dataframe throws out the DC
    # component of the FFT which tends to throw off the Y-axis limits since we
    # detrend the signal.
    psd = psd.iloc[:, 1:]
    plt.plot(psd.columns, psd.iloc[0].values, label='In 1')
    plt.plot(psd.columns, psd.iloc[1].values, label='In 2')

    # The octave scaling is a special one provided by psiaudio. It is only
    # enabled when you import psiaudio in your own code.
    plt.xscale('octave')
    plt.axis(xmin=500, xmax=4000)
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Signal (dBVrms)')
    plt.show()


if __name__ == '__main__':
    play_tones()
