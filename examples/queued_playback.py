import matplotlib.pyplot as plt
from psiaudio.queue import FIFOSignalQueue
from psiaudio.stim import load_wav
from ncrar_audio.babyface import Babyface


#device = Babyface()
fs = 44e3

wav = load_wav(fs, 'stim/da_40ms.wav')
queue = FIFOSignalQueue(fs=fs)
queue.append(wav, 10, delays=1)

b = queue.pop_buffer(int(fs*10))
plt.plot(b)
plt.show()
