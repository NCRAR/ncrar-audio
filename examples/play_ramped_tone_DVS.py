import time
import logging
logging.basicConfig(level='INFO')

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from psiaudio.calibration import FlatCalibration
from psiaudio.stim import ramped_tone
from psiaudio import util

from ncrar_audio.dante import Dante


def main():
    calibration = FlatCalibration.unity()
    device = Dante()
    #device = SoundDevice('ASIO Fireface USB',
    #                     'ASIO Fireface USB', fs=48e3)
    tone = ramped_tone(device.fs, frequency=1e3, duration=2,
                       rise_time=2.5e-3, level=-40, calibration=calibration)

    time.sleep(10)
    for i in range(32):
        signal = np.zeros(shape=(32, len(tone)))
        signal[i] = tone
        with device.play(signal):
            device.join()
        time.sleep(2)


if __name__ == '__main__':
    main()
