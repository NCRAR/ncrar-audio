'''
Record the output of XLR1 via the input of XLR1 on the Babyface. Connect the
input to the output with a XLR cable before running this demo.
'''

import matplotlib.pyplot as plt
import numpy as np

from psiaudio.calibration import FlatCalibration
from psiaudio.stim import tone
from ncrar_audio.babyface import Babyface


device = Babyface('XLR1', 'XLR2', use_osc=False)
cal = FlatCalibration.unity()
waveform = tone(fs=device.fs, duration=1, frequency=1e3, level=-20, calibration=cal)
recording = device.acquire(waveform, 2)
t = np.arange(5000) / device.fs
plt.plot(t, recording[..., :5000].T)

plt.axvline(0.046439909297052155, c='k', label='Sound driver delay')
plt.axvline(0.046439909297052155 + 12 / device.fs, ls=':', c='k',
            label='Sound driver delay plus DA/AD delay')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude (Volts)')
plt.show()
