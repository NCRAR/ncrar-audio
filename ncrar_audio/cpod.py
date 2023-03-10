import pyxid2


class CodeManager:

    def __init__(self, cpod):
        self.cpod = cpod

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cpod.clear_code()


class CPod:

    def __init__(self, pulse_duration=0):
        # Get a list of all attached XID devices then use the first one. This
        # assumes that we have only one XID device (i.e., the Cedrus C-Pod)
        # attached.
        self.dev = pyxid2.get_xid_devices()[0]
        self.dev.reset_base_timer()
        self.dev.reset_rt_timer()
        self.dev.set_pulse_duration(int(pulse_duration * 1e3))

    def set_code(self, code):
        bitmask = (code << 1) | 1
        self.dev.set_lines(bitmask)
        return CodeManager(self)

    def clear_code(self):
        self.dev.set_lines(0)
