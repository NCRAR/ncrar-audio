from .sound_device import SoundDevice


class Dante(SoundDevice):

    def __init__(self):
        super().__init__('Dante Virtual Soundcard (x64)',
                         'Dante Virtual Soundcard (x64)', fs=48e3)

    def play_azimuth(self, signal, azimuth):
        pass
