import logging
log = logging.getLogger(__name__)

import json
from pathlib import Path
import re
from threading import Lock

import numpy as np
from scipy.interpolate import interp1d

from .osc_client import OSCClient
from .sound_device import SoundDevice
import sounddevice as sd

P_VOLUME = re.compile('/1/volume(\d+)Val')
P_MICGAIN = re.compile('/1/micgain(\d+)Val')


def load_volume_map():
    VOLUME_MAP_FILE = Path(__file__).parent / 'totalmix_fx_volume_map.json'
    VOLUME_MAP = json.loads(VOLUME_MAP_FILE.read_text())
    scale = np.fromiter(VOLUME_MAP.keys(), 'float32')
    db = np.fromiter(VOLUME_MAP.values(), 'float32')
    return interp1d(db, scale, 'linear')


class Babyface(SoundDevice):

    def __init__(self, ip_address=None, send_port=7001, recv_port=9001,
                 use_osc=True):
        self._volume_db = {}
        self._mic_gain_db = {}
        self._lock = Lock()
        self._volume_map = load_volume_map()

        if use_osc:
            self.osc_client = OSCClient(ip_address, send_port, recv_port)
            self.osc_client.dispatch.map('/1/volume*Val', self._volume_updated)
            self.osc_client.dispatch.map('/1/micgain*Val', self._mic_gain_updated)

        name = 'ASIO Fireface USB'
        super().__init__(name, name, input_scale=0.3395)

    def set_volume(self, db, channels=None):
        if channels is None:
            channels = np.arange(12) + 1
        volume = float(self._volume_map(db))
        messages = [(f'/1/volume{i}', volume) for i in channels]
        self.osc_client.send_messages(messages)

    def set_mic_gain(self, db, channels=None):
        if channels is None:
            channels = np.arange(12) + 1
        volume = float(self._volume_map(db))
        messages = [(f'/1/micgain{i}', volume) for i in channels]
        self.osc_client.send_messages(messages)

    def _volume_updated(self, address, value):
        with self._lock:
            channel = int(P_VOLUME.match(address).group(1))
            if value == '-oo':
                value = -np.inf
            else:
                value = float(value.split(' ')[0])
            self._volume_db[channel] = value

    def _mic_gain_updated(self, address, value):
        with self._lock:
            channel = int(P_MICGAIN.match(address).group(1))
            if value == '-oo':
                value = -np.inf
            else:
                value = float(value.split(' ')[0])
            self._mic_gain_db[channel] = value


def test_ramped_tone():
    import logging
    logging.basicConfig(level='INFO')

    import matplotlib.pyplot as plt
    import numpy as np

    from psiaudio.api import ramped_tone, FlatCalibration
    calibration = FlatCalibration.unity()

    device = Babyface()
    frequencies = np.array([1e3, 2e3])[:, np.newaxis]
    tone = ramped_tone(device.fs, frequency=frequencies, duration=1,
                       rise_time=0.1, level=-6, calibration=calibration)
    recording = device.acquire(tone.T)
    plt.plot(recording)
    plt.axis(ymin=-10, ymax=10)
    plt.show()
