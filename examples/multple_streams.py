import matplotlib.pyplot as plt
import numpy as np

from psiaudio.calibration import FlatCalibration
from psiaudio.stim import tone
from ncrar_audio.babyface import Babyface


device = Babyface('XLR1', None, use_osc=False)
cal = FlatCalibration.unity()
waveform = tone(fs=device.fs, duration=1, frequency=1e3, level=-20, calibration=cal)

# This allows you to start playing audio and then do other operations in a
# non-blocking fashion.
with device.play_async(waveform) as stream:
    print('Async stream started')
    while not device.wait(timeout=0.1):
        print('... still waiting')
    print('Async stream done!')


# This just plays the sound and then exits once it's done.
print('Stream started')
device.play(waveform)
print('Stream done!')
