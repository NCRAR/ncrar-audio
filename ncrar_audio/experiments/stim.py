import numpy as np

from psiaudio import stim


def make_click(fs, amplitude, duration):
    samples = int(round(fs * duration))
    return amplitude * np.ones(samples)


def make_tone_pip(fs, amplitude, frequency, duration):
    samples = int(round(fs * duration))
    time = np.arange(samples) / fs
    tone = amplitude * np.cos(2 * np.pi * frequency * time)
    envelope = signal.windows.blackmanharris(samples)
    return tone * envelope


def make_ram_efr(fs, amplitude, fc, fm, duration, duty_cycle, alpha):
    samples = int(round(fs * duration))
    time = np.arange(samples) / fs
    tone = amplitude * np.cos(2 * np.pi * fc * time)
    envelope = stim.square_wave(offset=0, fs=fs, samples=samples, depth=1, fm=fm,
                                duty_cycle=duty_cycle, alpha=alpha)
    return tone * envelope


def make_sam_efr(fs, amplitude, fc, fm, duration):
    return stim.apply_cos2envelope(
        waveform=stim.sam_tone(
            fs=fs,
            fc=fc,
            fm=fm,
            level=amplitude,
            depth=1,
            duration=duration,
        ),
        fs=fs,
        rise_time=5e-3,
        duration=duration
    )


def load_wav(fs, amplitude, wavfile):
    waveform = amplitude * stim.load_wav(fs, wavfile)
    if waveform.ndim != 1:
        raise ValueError('Not designed to handle N-dimensional waveforms')
    return waveform


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('stim', choices=['RAM', 'SAM'])
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    from psiaudio import util
    fs = 100e3

    if args.stim == 'RAM':
        stim = make_ram_efr(fs, 1, 4e3, 110, 0.5, 0.25, 0)
    elif args.stim == 'SAM':
        stim = make_sam_efr(fs, 1, 4e3, 110, 0.5)

    psd = util.psd_df(stim, fs)
    figure, axes = plt.subplots(1, 2)
    axes[0].plot(stim)
    axes[1].plot(util.db(psd))
    axes[1].set_xscale('octave')
    plt.show()
