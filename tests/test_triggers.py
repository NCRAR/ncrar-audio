import pytest

import numpy as np

from ncrar_audio import triggers


@pytest.fixture(scope='module', params=[512, 16384])
def fs(request):
    return request.param


@pytest.fixture(scope='module', params=['cos', 'square'])
def trigger_shape(request):
    return request.param


@pytest.fixture
def n_unique_trigs():
    return 3


@pytest.fixture
def n_trigs(n_unique_trigs):
    # Must be a multiple of the number of unique trigger groups
    return 4 * n_unique_trigs


@pytest.fixture
def trig_iti():
    return 0.1


@pytest.fixture
def trigger_train(fs, trigger_shape, n_unique_trigs, n_trigs, trig_iti=0.1):
    n_samples = int(fs * trig_iti)

    trigs = []
    for repeat in range(n_unique_trigs):
        t = triggers.make_trigger(fs, n_samples, shape=trigger_shape, shape_settings={'repeat': repeat+1})
        trigs.append(t)

    t = np.concatenate(trigs, axis=-1)
    t = np.tile(t, int(n_trigs / n_unique_trigs))
    t = np.pad(t, 10, 'constant')
    return t


def test_extract_triggers(fs, n_unique_trigs, n_trigs, trig_iti, trigger_train):
    group_window = int(0.025 * fs)
    trigs = triggers.extract_triggers(trigger_train, group_window=group_window)
    assert len(trigs) == n_unique_trigs
    for code in range(n_unique_trigs):
        assert len(trigs[code + 1]) == n_trigs / n_unique_trigs
        iti = np.mean(np.diff(trigs[code + 1]) / fs)
        assert pytest.approx(iti, abs=1e-2) == (trig_iti * n_unique_trigs)
