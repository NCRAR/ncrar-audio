from pathlib import Path
import struct

import numpy as np


general_header = [
    ('stim_type', 'i'),
    ('description', '200s'),
    ('cal_info', '20s'),
    ('fs', 'd'),
    ('duration', 'i'),
    ('max_amplitude', 'i'),
    ('component_count', 'i'),
    ('zero_position', 'i'),
    ('_reserved', '200b')
]


component_header = [
    ('component_type', 'i'),
    ('frequency', 'd'),
    ('phase', 'd'),
    ('delay', 'i'),
    ('duration', 'i'),
    ('window', 'i'),
    ('rise_fall_time', 'i'),
    ('per_amp', 'd'),
    ('resp_freq', 'd'),
    ('SPL_to_HL', 'i'),
    ('_reserved', '36b'),
    ('active', 'b'),
    ('amp_mod', 'b'),
    ('freq_mod', 'b'),
    ('resp_seq', 'b'),
    ('calculate', 'b'),
    ('_reserved', '9b'),
    ('amp_mod_freq', 'd'),
    ('amp_mod_per', 'd'),
    ('amp_mod_phase', 'd'),
    ('_reserved', '40b'),
    ('freq_mod_freq', 'd'),
    ('freq_mod_per', 'd'),
    ('freq_mod_phase', 'd'),
    ('_reserved', '40b'),
    ('rep_seq_duration', 'i'),
    ('rep_seq_count', 'i'),
    ('rep_seq_freq', 'd'),
    ('rep_seq_isi', 'i'),
    ('rep_seq_ramp_level', 'i'),
    ('group_delay', 'i'),
    ('rep_seq_freq_range', 'd'),
    ('_reserved', '24b'),
    ('_reserved', '200b'),
]


cal_description = [
    ('right_cal', 'h'),
    ('right_max', 'h'),
    ('right_min', 'h'),
    ('left_cal', 'h'),
    ('left_max', 'h'),
    ('left_min', 'h'),
    ('spl_conversion', 'h'),
    ('offset_cal_flag', 'h'),
    ('ref_freq', 'i'),
    ('right_offset', 'h'),
    ('left_offset', 'h'),
    ('_reserved', '190b'),
]


def _read_description(description, fh):
    keys, fmt = zip(*description)
    descr = '=' + ''.join(fmt)
    nbytes = struct.calcsize(descr)
    values = struct.unpack(descr, fh.read(nbytes))

    result = {}
    for key, typecode, value in zip(keys, fmt, values):
        if key.startswith('_'):
            continue
        if typecode.endswith('s'):
            value = value.decode('utf-8').lstrip('\r').rstrip('\x00')
        result[key] = value
    return result


def read_stm(filename):
    with Path(filename).open('rb') as fh:
        # Get number of bytes in file. This is a sanity-check.
        fh.seek(0, 2)
        fh_nbytes = fh.tell()
        fh.seek(0, 0)

        b_version = fh.read(2)
        version, = struct.unpack('h', b_version)
        if version != 310:
            raise ValueError('Unsupported version')

        general = _read_description(general_header, fh)
        components = {}
        for i in range(general['component_count']):
            components[i] = _read_description(component_header, fh)

        caldata = {}
        for i in range(50):
            caldata[i] = _read_description(cal_description, fh)

        waveform_start = fh.tell()
        b_waveform = fh.read(general['duration'] * 2)
        waveform_end = fh.tell()

        waveform = np.frombuffer(b_waveform, 'h')

        # Make sure the file was parsed properly
        if len(waveform) != general['duration']:
            raise ValueError('Data missing')

        remaining = fh_nbytes - fh.tell()
        if remaining > 0:
            raise ValueError(f'File has {remaining} byes of unread data')

    return {
        'version': version,
        'header': general,
        'component_headers': components,
        'calibration_data': caldata,
        'waveform': waveform,
        'file_stats': {
            'waveform_start': waveform_start,
            'waveform_end': waveform_end,
        },
    }
