from .osc_client import OSCClient
from .sound_device import SoundDevice


class Babyface(SoundDevice):

    def __init__(self, ip_address=None, send_port=7001, recv_port=9001):
        self.osc_client = OSCClient(ip_address, send_port, recv_port)
        name = 'ASIO Fireface USB'
        super().__init__(name, name, input_scale=0.3395)
