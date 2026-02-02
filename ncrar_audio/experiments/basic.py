import numpy as np
from tqdm import tqdm

from psiaudio.queue import BlockedFIFOSignalQueue

from ncrar_audio.babyface import Babyface
from ncrar_audio.cpod import DummyCPod
from ncrar_audio import triggers


def play_repeat(n_stim, stim_cb, stim_params, extra_gain, test_ear,
                stim_rate=None, stim_isi=None, n_blocks=2):
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
        Rate at which to present stimuli (Hz). If provided, `stim_isi` must be
        set to None.
    stim_isi : {None, float}
        Interstimulus interval. If provided, `stim_rate` must be set to None.
    n_blocks : int
        Number of replicates to acquire.
    '''
    # Check values for input parameters. 
    if stim_rate is not None and stim_isi is not None:
        raise ValueError('Must provide either stim_rate or stim_isi')

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

    amplitude = 10 ** (extra_gain / 20)
    print(f'Sampling rate is {bface.fs} Hz')
    print(f'Stimulus amplitude is {amplitude}')
    waveform = stim_cb(bface.fs, amplitude, **stim_params)
    cp = DummyCPod()

    # In other scripts, we may use the `delays` parameter of the queue `append`
    # method to control stimulus timing. Since the ABR stimulus is shorter than
    # the default duration of the cos trigger designed by Sam Gordon, we need
    # to encode the delay in the stimulus itself so that the stimulus and
    # trigger queues can be properly aligned.
    if stim_rate is not None:
        total_samples = int(bface.fs // stim_rate)
    elif stim_isi is not None:
        if len(stim_isi) == 2:
            lb, ub = stim_isi
            def delays(seed):
                # Build the delays array
                rng = np.random.default_rng(seed=seed)
                nonlocal lb
                nonlocal ub
                while True:
                    yield next(rng.uniform(lb, ub))
        else:
            isi_samples = int(bface.fs * stim_isi)
            total_samples = isi_samples + waveform.shape[-1]
    else:
        raise ValueError('Must provide stim_rate or stim_isi')

    print(f'True stimulus rate is {bface.fs / total_samples:.2f} Hz')
    stim = np.zeros(total_samples)
    stim[:len(waveform)] = waveform

    # Now, let's make a trigger waveform of the same length as the stimulus.
    # (i.e., same number of samples). By ensuring the trigger waveform is the
    # same length as the wav file, it makes it super-easy to ensure the timing
    # is accurate.
    t1 = triggers.make_trigger(bface.fs, stim.shape[-1], shape='cos')
    t2 = triggers.make_trigger(bface.fs, stim.shape[-1], shape='cos',
                               shape_settings={'repeat': 2})

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
            trig_queue.append(t1, n_stim // 2, delays=1)
            trig_queue.append(t2, n_stim // 2, delays=2)

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
