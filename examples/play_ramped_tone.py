from pathlib import Path
from psiaudio.stim import load_wav

import logging
logging.basicConfig(level='INFO')

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

from psiaudio.calibration import FlatCalibration
from psiaudio.stim import ramped_tone
from psiaudio import util

from ncrar_audio.babyface import Babyface

base_path = Path(r'C:\Users\biosemi\Desktop\Koerner_CDA2')
wav_path = base_path / 'tone1k_150ms.wav'


def main():
    calibration = FlatCalibration.unity()
    device = Babyface(output_channels='earphones', trigger_channels='XLR1',
                      use_osc=False)

    wav = load_wav(device.fs, wav_path)
    wav = wav[:, :2].T

    for i in range(10):
        with device.play(wav):
            device.join()


if __name__ == '__main__':
    main()
