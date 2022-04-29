import struct

import numpy as np
import pandas as pd
from scipy import signal

# Note, most of this code is derived from code written by Sam Gordon. I don't
# have the original reference documentation from IHS (despite asking for it),
# so I've done my best. The important thing is that the epochs are properly
# returned!

def read_cam(filename):
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


    with filename.open('rb') as fh:
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

        # There are four channels. the int16 (2-byte) samples are stored in
        # interleaved fashion.
        fh.seek(200000, 0)
        recording = np.fromfile(fh, dtype='h').reshape((-1, 4)).T
        dio = recording[-1] / (2 ** (16-1))
        stim_indices = np.flatnonzero(dio < -0.1)
        result['recording'] = recording[:3]
        # scale to fraction of sys.maxint
        result['dio'] = dio
        result['stim_indices'] = stim_indices

        result['n_channels'] = result['cont_data_suppl'][0]
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


def extract_cam_epochs(result, offset=0, duration=0.5, filter_lb=80,
                       filter_ub=3000, filter_order=800):
    # First, filter the signal so we don't have to worry about padding to
    # control for filter edges when epoching.
    b = signal.firwin(filter_order, (filter_lb, filter_ub), fs=result['fs'],
                      pass_zero='bandpass')
    w_filt = signal.filtfilt(b, 1, result['recording'], axis=-1)

    # Find the start/end edges of the epochs
    o = int(round(result['fs'] * offset))
    d = int(round(result['fs'] * duration))
    i_start = result['stim_indices'] + o
    i_end = i_start + d
    i_max = w_filt.shape[-1]
    m = (0 <= i_start) & (i_start < i_max) & (0 <= i_end) & (i_end < i_max)

    epochs = [w_filt[..., lb:ub][np.newaxis] \
              for (lb, ub) in zip(i_start[m], i_end[m])]
    epochs = np.concatenate(epochs)
    if len(epochs) % 2:
        epochs = epochs[:-1]

    return epochs
