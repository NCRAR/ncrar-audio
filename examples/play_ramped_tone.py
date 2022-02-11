import logging
logging.basicConfig(level='INFO')

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from psiaudio.calibration import FlatCalibration
from psiaudio.stim import ramped_tone
from psiaudio import util

from ncrar_audio.babyface import Babyface


def main():
    calibration = FlatCalibration.unity()
    device = Babyface('earphones', 'XLR1')
    frequencies = np.array([1e3, 2e3])[:, np.newaxis]
    tone = ramped_tone(device.fs, frequency=frequencies, duration=0.5,
                       rise_time=2.5e-3, level=-16.7, calibration=calibration)

    #device.set_output('XLR')
    #device.play(tone)
    for i in range(10):
        device.play(tone)

    return
    recording = device.play(tone, 2, [0, 3])
    psd = util.db(util.psd_df(recording, device.fs)).iloc[:, 1:]

    plt.plot(psd.columns, psd.iloc[0].values, label='In 1')
    plt.plot(psd.columns, psd.iloc[1].values, label='In 2')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Signal (dBVrms)')
    plt.show()


if __name__ == '__main__':
    main()
