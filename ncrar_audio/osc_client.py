import logging
log = logging.getLogger(__name__)

import socket
import threading
import time

from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client


class OSCClient:
    '''
    Sets up bidirectional communication with a device supporting the Open Sound
    Control protocol.
    '''

    def __init__(self, ip_address=None, send_port=7001, recv_port=9001):
        if ip_address is None:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
        self.send_port = send_port
        self.recv_port = recv_port
        self.ip_address = ip_address
        if self.recv_port is not None:
            self._configure_server()
        self._configure_client()

    def _configure_server(self):
        # The dispatcher is responsible for checking whether the datagram
        # (i.e., message recieved on a socket) is in valid OSC format. If it's
        # valid OSC format, then it will dispatch the message to a callback. By
        # default, all messages will get passed to a callback that logs the
        # unhandled message (useful for knowing whether we're "missing out".
        self.dispatch = dispatcher.Dispatcher()
        self.dispatch.set_default_handler(self._log_unhandled_message)

        addr = (self.ip_address, self.recv_port)
        self.server = osc_server.ThreadingOSCUDPServer(addr, self.dispatch)
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        # Be sure to set in daemon so that thread exits cleanly when the master
        # thread exits.
        self.server_thread.daemon = True
        self.server_thread.start()
        log.info('Set up client to recv on %s:%d', *addr)

    def _configure_client(self):
        self.client = udp_client.SimpleUDPClient(self.ip_address,
                                                 self.send_port)
        log.info('Set up client to send on %s:%d', self.ip_address,
                 self.send_port)

    def _log_unhandled_message(self, *args):
        m = 'Recieved an unhandled OSC response: address=%s, value=%r'
        log.warning(m, *args)

    def send_message(self, address, value, response_address=None):
        if response_address is not None:
            # TODO: We need to figure out how to attach the response to the
            # specific request. asyncio is perfect for this type of dispatch.
            raise NotImplementedError
        log.info('Sending OSC command: address=%s, value=%r', address, value)
        self.client.send_message(address, value)

    def send_messages(self, messages):
        '''
        Send a list of messages

        Each message must be a two or three element iterable. The following are
        valid message formats:
            ('/1/volume1', 1.0, '/1/volume1'),
            ('/1/busOutput', 1),
        '''
        for message in messages:
            self.send_message(*message)
            time.sleep(0.001)


if __name__ == '__main__':
    # Set up basic logging config so we can see the logging messages printed
    # out
    logging.basicConfig(level='INFO')

    device = OSCClient()
    osc_messages = [
        ["/1/busPlayback", 1.0],
        ["/1/volume1",0.7],
        ["/1/busInput", 1.0],
        ["/1/volume1",0.7],
        ["/1/busOutput", 1.0],
        ["/1/volume1",0.7],
    ]
    device.send_messages(osc_messages)
    time.sleep(1)
