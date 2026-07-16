import numpy as np
from delta._inputs import as_layers


def test_as_layers_masked_array_no_masked_values():
    arr = np.ma.MaskedArray(np.ones((4, 4), dtype=np.float32))
    data, variance, mask = as_layers(arr)
    assert data.shape == (4, 4)
    assert variance is None
    assert mask is None


def test_as_layers_masked_array_with_masked_values():
    raw = np.ones((4, 4), dtype=np.float32)
    bad = np.zeros((4, 4), dtype=bool)
    bad[1, 2] = True
    arr = np.ma.MaskedArray(raw, mask=bad)
    data, variance, mask = as_layers(arr)
    assert mask is not None
    assert mask.shape == (4, 4)
    assert mask[1, 2] == 1
    assert mask.sum() == 1


def test_as_layers_fully_masked_array():
    arr = np.ma.masked_all((4, 4), dtype=np.float32)
    data, variance, mask = as_layers(arr)
    assert mask is not None
    assert mask.shape == (4, 4)
    assert np.all(mask == 1)
