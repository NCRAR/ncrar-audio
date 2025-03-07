import logging
log = logging.getLogger(__name__)

from pathlib import Path
import struct

import numpy as np
import pandas as pd
from scipy import signal

# Note, most of this code is derived from code written by Sam Gordon. I don't
# have the original reference documentation from IHS (despite asking for it),
# so I've done my best. The important thing is that the epochs are properly
# returned!

def read_cam(filename, header_only=False):
    description = [
        ('std_ihs_ep', 1000, 'h'),
        ('cont_data_suppl', 4000, 'i'),
        ('seq1_locs', 5000, 'i'),
        ('seq1_align', 5000, 'i'),
        ('seq2_locs', 5000, 'i'),
        ('seq2_align', 5000, 'i'),
        ('marker_locc', 4000, 'i'),
        ('marker_types', 4000, 'i'),
        ('additional_dloc', 12500, 'i'),
        ('short_string', None, f'{125*32}s'),
        ('remainder_string', None, '16000s'),
    ]

    recording_type_map = {
        0: 'raw',
        1: 'std average',
        2: 'clad',
        3: 'sequential stimulation',
        4: 'complex sequence average'
    }


    with Path(filename).open('rb') as fh:
        fh.seek(0, 2)
        fh_nbytes = fh.tell()
        fh.seek(0, 0)

        result = {}
        for (name, count, dtype) in description:
            if count is None:
                nbytes = struct.calcsize(dtype)
                value = struct.unpack(dtype, fh.read(nbytes))[0]
                if dtype.endswith('s'):
                    value = value.decode('utf-8').replace('\x00', ' ')
                result[name] = value
            else:
                result[name] = np.fromfile(fh, dtype=dtype, count=count)

        n_channels = result['n_channels'] = result['cont_data_suppl'][0]

        if not header_only:
            # There are four channels. the int16 (2-byte) samples are stored in
            # interleaved fashion.
            fh.seek(200000, 0)
            recording = np.fromfile(fh, dtype='h').reshape((-1, n_channels)).T
            result['dio'] = recording[-1]
            result['recording'] = recording[:-1]

    result['sample_time'] = result['cont_data_suppl'][6] / 1e9
    result['fs'] = 1 / result['sample_time']

    recording_type = result['cont_data_suppl'][1]
    result['recording_type'] = recording_type_map[recording_type]

    # 1 rare, 2 cond, 3 alt
    result['recording_phase'] = result['cont_data_suppl'][2]

    # 0 off, 1 on
    result['digital_channel'] = result['cont_data_suppl'][3]

    result['n_sequences'] = result['cont_data_suppl'][4]
    result['n_unique_stimuli'] = result['cont_data_suppl'][5]
    result['seq_sweep_length'] = result['cont_data_suppl'][249]
    result['seq_sweep_amount'] = result['cont_data_suppl'][250]
    result['n_stimuli'] = result['cont_data_suppl'][251]
    result['stim_rate'] = result['cont_data_suppl'][252] / 1000
    result['seq_ear_send'] = result['cont_data_suppl'][253]

    # 0 alt, 2 cond, 1 rare, 3 complex
    result['seq_stim_mode'] = result['cont_data_suppl'][254]
    result['seq_stim_send'] = result['cont_data_suppl'][255]

    result['stim_settings'] = stim_settings = []
    for i in range(result['n_stimuli']):
        stim_settings.append({
            # intensity in db
            'intensity': result['cont_data_suppl'][310 + i * 15],
            # 1 is click, 2 is toneburst
            'type': result['cont_data_suppl'][311 + i * 15],
            # frequency of tone pip, 0 if click
            'frequency': result['cont_data_suppl'][312 + i * 15],
            # duration of stimulus in number of points
            'duration': result['cont_data_suppl'][313 + i * 15],
            # calibration value in db
            'right_calibration': result['cont_data_suppl'][314 + i * 15],
            # calibration value in db
            'left_calibration': result['cont_data_suppl'][315 + i * 15],
            # hl = 0, spl = 1
            'hlspl_flag': result['cont_data_suppl'][316 + i * 15],
            # rise/fall time (in points) of stim
            'stim_rise_fall_time': result['cont_data_suppl'][317 + i * 15],
            # stimulator type ???
            'stimulator_type': result['cont_data_suppl'][318 + i * 15],
        })

    # Apparently channel #4 is the digital channel
    channel_settings = []
    for i in range(result['n_channels']):
        channel_settings.append({
            # gain in uv?
            'gain': result['cont_data_suppl'][400 + i * 10] / 100,
            # what units???
            'high_pass': result['cont_data_suppl'][401 + i * 10] / 1000,
            'low_pass': result['cont_data_suppl'][402 + i * 10] / 1000,
            # notch filter on or off
            'notch': result['cont_data_suppl'][403 + i * 10] / 1000,
        })
    result['channel_settings'] = pd.DataFrame(channel_settings)
    return result



def extract_timestamps(dio):
    ts = {}

    # These are the start and end markers of the sequence. In at least one file
    # I looked at, the end marker was missing for some reason. We need to
    # handle that edge-case.
    m_start_end = (dio > -30e3) & (dio <= -20e3)
    bounds = np.flatnonzero(m_start_end)
    if len(bounds) == 2:
        lb, _ = ts['start'], ts['end'] = bounds
    elif len(bounds) == 1:
        # If end marker is missing, then set it to None
        lb, _ = ts['start'], ts['end'] = bounds[0], None

    else:
        raise ValueError('Unexpected marker found in DIO')

    ts['stim'] = {}
    get_id = lambda x: int(np.abs(x) % 10)

    # Check to see if start marker of sequence also encodes start marker of
    # stimulus.
    if get_id(lb) != 0:
        ts['stim'].setdefault(get_id(dio[lb]), []).append(lb)

    # Now, pull out the stimulus start times.
    m_stim = (dio > -17e3) & (dio <= -16e3)
    for i in np.flatnonzero(m_stim):
        ts['stim'].setdefault(get_id(dio[i]), []).append(i)
    for k, v in ts['stim'].items():
        ts['stim'][k] = np.array(v)

    return ts


def extract_cam_epochs(result, offset=0, duration=0.5, filter_lb=80,
                       filter_ub=3000, filter_order=800):

    # Find the timestamps we want to extract.
    ts = extract_timestamps(result['dio'])

    # First, filter the signal so we don't have to worry about padding to
    # control for filter edges when epoching.
    if filter_order is not None:
        b = signal.firwin(filter_order, (filter_lb, filter_ub),
                          fs=result['fs'], pass_zero='bandpass')
        w = signal.filtfilt(b, 1, result['recording'], axis=-1)
    else:
        w = result['recording']
    dio = result['dio']

    # Find the start/end edges of the epochs
    o = int(round(result['fs'] * offset))
    d = int(round(result['fs'] * duration))

    all_epochs = {}
    n_error = 0
    for key, indices in ts['stim'].items():
        epochs = []
        i_start = indices + o
        i_end = i_start + d

        for i, (lb, ub) in enumerate(zip(i_start, i_end)):
            m = dio[lb:ub] < -32000
            if m.any():
                n_error += 1
            else:
                e = w[..., lb:ub][np.newaxis]
                if e.shape[-1] != d:
                    log.warn('Dropping truncated epoch %d', i)
                else:
                    epochs.append(e)

        log.warn('Found %d epochs for stim ID %d', len(epochs), key)
        all_epochs[key] = np.concatenate(epochs)

    if n_error:
        log.warn('Could not read %d epochs due to dropped buffer', n_error)
    return all_epochs
