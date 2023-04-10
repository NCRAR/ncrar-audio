import logging
log = logging.getLogger(__name__)

from collections.abc import Iterable
import threading

import numpy as np
import sounddevice as sd
sd.default.latency = 'low'


class BaseCallbackContext:

    def __init__(self, cb=None):
        self.i = 0
        if cb is None:
            cb = lambda x: x
        self.cb = cb

    def _valid_samples(self, samples, status):
        if status:
            log.warning('portaudio callback status: %r', status)

        # Calculate the number of valid samples remaining
        samples_remaining = self.n - self.i
        log.info('%r: %r', self.n, self.i)
        if samples_remaining == 0:
            raise sd.CallbackStop

        valid_samples = min(samples_remaining, samples)
        result = self.i, valid_samples
        self.i += valid_samples
        self.cb(valid_samples)
        return result

    def __call__(self, *args):
        raise NotImplementedError


class RecordCallbackContext(BaseCallbackContext):

    def __init__(self, input_buffer, input_scale, cb=None):
        super().__init__(cb=cb)
        self.input_buffer = input_buffer
        self.input_scale = input_scale
        self.n = len(input_buffer)

    def __call__(self, indata, samples, time, status):
        # Read the next segment to the input buffer
        i, valid_samples = self._valid_samples(samples, status)
        self.input_buffer[i:i + valid_samples] = indata[:valid_samples] / self.input_scale


class QueueCallbackContext:

    def __init__(self, queues):
        super().__init__()
        self.queues = queues

    def __call__(self, outdata, samples, time, status):
        if status:
            log.warning('portaudio callback status: %r', status)
        for i, q in enumerate(self.queues):
            outdata[:, i] = q.pop_buffer(samples)

        # Check if any queues are empty
        for q in self.queues:
            if not q._empty:
                break
        else:
            raise sd.CallbackStop


class PlayCallbackContext(BaseCallbackContext):

    def __init__(self, output_buffer, cb=None):
        super().__init__(cb=cb)
        self.output_buffer = output_buffer
        self.n = len(output_buffer)

    def __call__(self, outdata, samples, time, status):
        # Write the next segment to the output buffer
        i, valid_samples = self._valid_samples(samples, status)
        outdata[:valid_samples] = self.output_buffer[i:i + valid_samples]
        outdata[valid_samples:] = 0


class PlayRecordCallbackContext(BaseCallbackContext):

    def __init__(self, input_buffer, output_buffer, input_scale, cb=None):
        super().__init__(cb=cb)
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.input_scale = input_scale
        self.n = len(output_buffer)

    def __call__(self, indata,  outdata, samples, time, status):
        # Read/write the next segments
        i, valid_samples = self._valid_samples(samples, status)
        outdata[:valid_samples] = self.output_buffer[i:i + valid_samples]
        outdata[valid_samples:] = 0
        self.input_buffer[i:i + valid_samples] = indata[:valid_samples] / self.input_scale


class SoundDevice:
    '''
    Basic wrapper around an audio device that provides play and record
    functionality.

    Parameters
    ----------
    input_device : string
        Name of input device as seen by portaudio.
    output_device : string
        Name of output device as seen by portaudio.
    input_scale : float
        Scaling factor to apply to input stream. This is primarily used by the
        ASIO Fireface interface for the Babyface since the input data needs to
        be scaled by ~1/3 to match the expected amplitude.
    '''

    def __init__(self, input_device, output_device, input_scale=1, fs=None):
        self.input_device = input_device
        self.input_info = sd.query_devices(self.input_device)
        self.input_scale = input_scale
        log.info('Properties for input device %r: %r', self.input_device, self.input_info)

        self.output_device = output_device
        self.output_info = sd.query_devices(self.output_device)
        log.info('Properties for output device %r: %r', self.output_device, self.output_info)

        if fs is None:
            self.fs = self.input_info['default_samplerate']
        else:
            self.fs = fs

    def _start_stream(self, stream_class, **stream_kw):
        self.event = threading.Event()
        return stream_class(finished_callback=self.event.set, **stream_kw)

    def wait(self, timeout=0.1):
        return self.event.wait(timeout)

    def join(self):
        while True:
            if self.event.wait(0.1):
                break

    def play_async_queue(self, queues, output_channels=None, cb=None):
        output_settings = None
        input_settings = None
        if output_channels is None:
            output_channels = len(queues)
        elif isinstance(output_channels, Iterable):
            if len(output_channels) != len(queues):
                m = 'Output mapping of channels does not match waveform shape'
                raise ValueError(m)
            output_settings = sd.AsioSettings(output_channels)
            output_channels = len(output_channels)
        return self._start_stream(
            sd.OutputStream,
            device=self.output_device,
            samplerate=self.fs,
            blocksize=1024,
            callback=QueueCallbackContext(queues),
            channels=output_channels,
            extra_settings=output_settings,
        )

    def play_queue(self, queues, output_channels=None):
        with self.play_async_queue(queues, output_channels) as stream:
            self.join()

    def play_async(self, waveform, output_channels=None, cb=None):
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis]
        output_settings = None
        input_settings = None
        if output_channels is None:
            output_channels = len(waveform)
        elif isinstance(output_channels, Iterable):
            if len(output_channels) != len(waveform):
                m = 'Output mapping of channels does not match waveform shape'
                raise ValueError(m)
            output_settings = sd.AsioSettings(output_channels)
            output_channels = len(output_channels)

        return self._start_stream(
            sd.OutputStream,
            device=self.output_device,
            samplerate=self.fs,
            blocksize=1024,
            callback=PlayCallbackContext(waveform.T, cb=cb),
            channels=output_channels,
            extra_settings=output_settings,
        )

    def play(self, waveform, output_channels=None, cb=None):
        with self.play_async(waveform, output_channels, cb=cb) as stream:
            self.join()

    def acquire(self, waveform, input_channels=1, output_channels=None):
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis]
        output_settings = None
        input_settings = None
        if output_channels is None:
            output_channels = len(waveform)
        elif isinstance(output_channels, Iterable):
            if len(output_channels) != len(waveform):
                m = 'Output mapping of channels does not match waveform shape'
                raise ValueError(m)
            output_settings = sd.AsioSettings(output_channels)
            output_channels = len(output_channels)
        if isinstance(input_channels, Iterable):
            input_settings = sd.AsioSettings(input_channels)
            input_channels = len(input_channels)

        recording = np.zeros((waveform.shape[-1], input_channels), dtype='double')
        stream = self._start_stream(
            sd.Stream,
            device=(self.input_device, self.output_device),
            samplerate=self.fs,
            blocksize=1024,
            callback=PlayRecordCallbackContext(recording, waveform.T, self.input_scale),
            channels=(input_channels, output_channels),
            extra_settings=(input_settings, output_settings),
        )
        with stream:
            self.join()
        return recording.T

    def record(self, n_samples, input_channels):
        input_settings = None
        if isinstance(input_channels, Iterable):
            input_settings = sd.AsioSettings(input_channels)
            input_channels = len(input_channels)

        recording = np.zeros((n_samples, input_channels))
        stream = self._start_stream(
            sd.InputStream,
            device=self.input_device,
            samplerate=self.fs,
            blocksize=1024,
            callback=RecordCallbackContext(recording, self.input_scale),
            channels=input_channels,
            extra_settings=input_settings,
        )
        with stream:
            self.join()
        return recording.T
