


def test_ramped_tone():
    import logging
    logging.basicConfig(level='INFO')

    import matplotlib.pyplot as plt
    from scipy import signal
    import numpy as np

    from psiaudio.api import ramped_tone, FlatCalibration
    calibration = FlatCalibration.unity()

    device = Babyface()
    frequencies = np.array([1e3, 2e3])[:, np.newaxis]
    tone = ramped_tone(device.fs, frequency=frequencies, duration=15,
                       rise_time=0.1, level=-16.7, calibration=calibration)
    print(tone.max(axis=1))
    device.play(tone)
    '''
    #recording = device.record(int(device.fs)*2, 2)
    recording = signal.detrend(recording)

    print(np.abs(recording).max(axis=-1))
    from psiaudio import util
    print(util.rms(recording))

    psd = util.db(util.psd_df(recording, device.fs))
    print(psd.max().max())

    plt.plot(psd.columns, psd.iloc[0].values, label='In 1')
    plt.plot(psd.columns, psd.iloc[1].values, label='In 2')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Signal (dBVrms)')
    plt.show()
    '''

def play_cal_tone():
    import pickle
    import numpy as np
    from psiaudio.api import ramped_tone, FlatCalibration
    from psiaudio import util
    calibration = FlatCalibration.unity()

    device = Babyface()
    frequencies = np.array([1e3])[:, np.newaxis]
    tone = ramped_tone(device.fs, frequency=frequencies, duration=15,
                       rise_time=0.1, level=-18.94, calibration=calibration)
    recording = device.record(tone.shape[-1], 2)
    print(recording.shape)
    print(util.rms(recording))
    np.savetxt('ER7c calibration.txt', recording)
    with open('ER7c calibration.pkl', 'wb') as fh:
        pickle.dump(recording, fh)


def test_volume_control():
    mapping = load_volume_map()
    print(mapping(0))


if __name__ == '__main__':
    test_volume_control()
    play_cal_tone()

