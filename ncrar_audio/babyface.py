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
from . import triggers

import sounddevice as sd

P_VOLUME = re.compile(r'/1/volume(\d+)Val')
P_MICGAIN = re.compile(r'/1/micgain(\d+)Val')


def load_volume_map():
    VOLUME_MAP_FILE = Path(__file__).parent / 'totalmix_fx_volume_map.json'
    VOLUME_MAP = json.loads(VOLUME_MAP_FILE.read_text())
    scale = np.fromiter(VOLUME_MAP.keys(), 'float32')
    db = np.fromiter(VOLUME_MAP.values(), 'float32')
    return interp1d(db, scale, 'linear')


class Babyface(SoundDevice):

    output_map = {
        'XLR': [0, 1],
        'XLR1': [0],
        'XLR2': [1],
        'earphones': [2, 3],
        'earphones_left': [2],
        'earphones_right': [3],
        None: [],
    }

    def __init__(self, output_channels='earphones', trigger_channels=None,
                 ip_address=None, send_port=7001, recv_port=9001,
                 use_osc=True):
        self._volume_db = {}
        self._mic_gain_db = {}
        self._lock = Lock()
        self._volume_map = load_volume_map()

        if use_osc:
            self.osc_client = OSCClient(ip_address, send_port, recv_port)
            self.osc_client.dispatch.map('/1/volume*Val', self._volume_updated)
            self.osc_client.dispatch.map('/1/micgain*Val', self._mic_gain_updated)

        self._output_channels = []
        self._trigger_channels = []
        self.set_output(output_channels)
        self.set_trigger(trigger_channels)

        name = 'ASIO Fireface USB'
        super().__init__(name, name, input_scale=0.3395)

    def set_output(self, output_channels):
        if isinstance(output_channels, str):
            output_channels = [output_channels]

        self._output_channels = []
        for oc in output_channels:
            self._output_channels.extend(self.output_map[oc])
        self._output_map = self._output_channels + self._trigger_channels

    def set_trigger(self, trigger_channels):
        self._trigger_channels = self.output_map[trigger_channels]
        self._output_map = self._output_channels + self._trigger_channels

    def add_trigger(self, waveform):
        trigger = triggers.make_trigger(self.fs, waveform.shape[-1])
        trigger = np.repeat(trigger[np.newaxis],
                            len(self._trigger_channels), 0)
        return np.vstack((waveform, trigger))

    def play(self, waveform, cb=None):
        if self._trigger_channels:
            waveform = self.add_trigger(waveform)
        return super().play(waveform, self._output_map, cb=cb)

    def play_queue(self, queues):
        return super().play_queue(queues, self._output_map)

    def acquire(self, waveform, input_channels):
        if self._trigger_channels:
            waveform = self.add_trigger(waveform)
        return super().acquire(waveform, input_channels, self._output_map)

    def play_mono(self, waveform, side):
        if waveform.ndim != 1:
            raise ValueError('Only a single waveform can be provided')
        if len(self._output_channels) == 1:
            self.play(waveform)
        silence = np.zeros_like(waveform)
        if side == 'left':
            waveform = np.vstack((waveform, silence))
        elif side == 'right':
            waveform = np.vstack((silence, waveform))
        else:
            raise ValueError('side must be "right" or "left"')
        self.play(waveform)

    def play_stereo(self, waveform):
        if waveform.ndim != 1:
            raise ValueError('Only a single waveform can be provided')
        if len(self._output_channels) == 1:
            raise ValueError('Only a single output channel available')
        waveform = np.vstack((waveform, waveform))
        self.play(waveform)

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
