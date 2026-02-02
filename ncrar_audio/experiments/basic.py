import numpy as np
from tqdm import tqdm

from psiaudio.queue import BlockedFIFOSignalQueue

from ncrar_audio.babyface import Babyface
from ncrar_audio.cpod import DummyCPod
from ncrar_audio import triggers


def play_repeat(n_stim, stim_cb, stim_params, extra_gain, test_ear,
                stim_rate=None, stim_iti=None, n_blocks=2):
    '''
    Parameters
    ----------
    n_stim : int
        Number of stimuli to play
    stim_cb : callable
        Callable that accepts sampling rate, amplitude, and the parameters in
        `stim_params` (passed in by keyword), e.g., `stim_cb(fs, amplitude,
        **stim_params)`.
    stim_params : dict
        Values to pass in by keyword to `stim_params`.
    extra_gain : float
        Extra gain to apply, in dB, to achive desired level. This is converted
        to a scaling factor (i.e., amplitude) and passed in to `stim_cb`.
    test_ear : {'right', 'left', 'binaural'}
        Which ear to test?
    stim_rate : {None, float}
        Rate at which to present stimuli (Hz). If provided, `stim_iti` must be
        set to None.
    stim_iti : {None, float}
        Interstimulus interval (from onset of one stimulus to onset of next
        stimulus). If provided, `stim_iti` must be set to None.
    n_blocks : int
        Number of replicates to acquire.
    '''
    # Check values for input parameters.
    if stim_rate is not None and stim_iti is not None:
        raise ValueError('Must provide either stim_rate or stim_iti')

    if test_ear not in ('none', 'right', 'left', 'binaural'):
        raise ValueError('Unrecognized test ear')

    # Connect to the Babyface. Specify that the audio goest through the
    # earphones and the trigger goes through the XLR1 output. This means that
    # the babyface is configured to have two outputs (earphone for right or
    # left ear and trigger to XLR1). When output_channels is set to
    # `earphones`, two analog output channels for audio are used (2 for left
    # and 3 for right).  When output_channels is set to either earphones_left
    # or earphones_right, only one analog output channel for audio is used (2
    # for left, 3 for right). The Babyface will handle routing channel 2 to the
    # left ear and 3 to the right ear.
    bface = Babyface(output_channels='earphones',
                     trigger_channels='XLR1', use_osc=False)

    # Eventually it would be nice to have a unified calibration suite that
    # handles the details behind the scenes. his is an example of how the
    #calibration = FlatCalibration.as_attenuation()
    #amplitude = level = calibration.get_sf(extra_gain)
    cp = DummyCPod()

    amplitude = 10 ** (extra_gain / 20)
    print(f'Sampling rate is {bface.fs} Hz')
    print(f'Stimulus amplitude is {amplitude}')


    # Create the stimuli and triggers. Ensure they are all equivalent length so
    # that they align properly.
    stim = stim_cb(bface.fs, amplitude, **stim_params)
    t1 = triggers.make_trigger(bface.fs, shape='cos')
    t2 = triggers.make_trigger(bface.fs, shape='cos', shape_settings={'repeat': 2})
    n_samples = max(len(stim), len(t1), len(t2))
    stim = np.pad(stim, (0, n_samples - len(stim)), mode='constant', constant_values=0)
    t1 = np.pad(t1, (0, n_samples - len(t1)), mode='constant', constant_values=0)
    t2 = np.pad(t2, (0, n_samples - len(t2)), mode='constant', constant_values=0)
    duration = n_samples / bface.fs

    if stim_rate is not None:
        stim_iti = (1 / stim_rate, 1 / stim_rate)
    elif isinstance(stim_iti, (float, int)):
        stim_iti = (stim_iti, stim_iti)
    delay_lb = stim_iti[0] - duration
    delay_ub = stim_iti[1] - duration

    def delays(seed):
        nonlocal delay_lb
        nonlocal delay_ub
        rng = np.random.default_rng(seed=seed)
        while True:
            yield rng.uniform(delay_lb, delay_ub)

    #print(f'True stimulus rate is {bface.fs / total_samples:.2f} Hz')
    #stim = np.zeros(total_samples)
    #stim[:len(waveform)] = waveform

    for block in range(n_blocks):
        with cp.set_code(block):
            left_queue = BlockedFIFOSignalQueue()
            left_queue.set_fs(bface.fs)
            right_queue = BlockedFIFOSignalQueue()
            right_queue.set_fs(bface.fs)

            if test_ear in ('left', 'binaural'):
                left_queue.append(stim, n_stim // 2, delays=delays(1))
                left_queue.append(-stim, n_stim // 2, delays=delays(2))
            else:
                # Otherwise, play silence
                left_queue.append(np.zeros_like(stim), n_stim // 2, delays=delays(1))
                left_queue.append(np.zeros_like(-stim), n_stim // 2, delays=delays(2))

            if test_ear in ('right', 'binaural'):
                right_queue.append(stim, n_stim // 2, delays=delays(1))
                right_queue.append(-stim, n_stim // 2, delays=delays(2))
            else:
                # Otherwise, play silence
                right_queue.append(np.zeros_like(stim), n_stim // 2, delays=delays(1))
                right_queue.append(np.zeros_like(-stim), n_stim // 2, delays=delays(2))

            # Remember how we made sure the trigger waveform was exactly the same
            # length as the stimulus? That makes it easy to align the trigger onsets
            # with the waveform onsets.
            trig_queue = BlockedFIFOSignalQueue()
            trig_queue.set_fs(bface.fs)
            trig_queue.append(t1, n_stim // 2, delays=delays(1))
            trig_queue.append(t2, n_stim // 2, delays=delays(2))

            queues = [left_queue, right_queue, trig_queue]

            # tqdm makes a nice console-based progress bar so you can easily tell how
            # far along you are in the experiment.
            cb = tqdm(total=n_stim, desc=f'Replicate {block+1}')
            queues[0].connect(lambda e: cb.update())

            # Play the queues. Be sure to pass them in in the order that we specified
            # the channels when creating the babyface instance. Trigger channels are
            # always last. That means that the trigger queue must be last. Depending on
            # whether we're doing monaural (left or right) or binaural, we will have
            # either a list of two queues (monaural) or three (binuaral). For binaural,
            # the queues are left, right, trigger.
            bface.play_queue(queues)

            # Properly close out the tqdm callback.
            cb.close()
