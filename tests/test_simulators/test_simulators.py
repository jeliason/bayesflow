import keras
import numpy as np


def test_two_moons(simulator, batch_size):
    samples = simulator.sample((batch_size,))

    assert isinstance(samples, dict)
    assert list(samples.keys()) == ["parameters", "observables"]
    assert all(isinstance(value, np.ndarray) for value in samples.values())

    assert samples["parameters"].shape == (batch_size, 2)
    assert samples["observables"].shape == (batch_size, 2)


def test_sample(simulator, batch_size):
    samples = simulator.sample((batch_size,))

    # test output structure
    assert isinstance(samples, dict)

    for key, value in samples.items():
        print(f"{key}.shape = {keras.ops.shape(value)}")

        # test type
        assert isinstance(value, np.ndarray)

        # test shape
        assert value.shape[0] == batch_size

        # test batch randomness
        assert not np.allclose(value, value[0])
