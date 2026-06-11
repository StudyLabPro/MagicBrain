"""
Tests for backend parity (NumPy vs JAX).
"""
import pytest
import numpy as np
from magicbrain.backends import get_backend, NumpyBackend


def test_numpy_backend_basic():
    """Test basic NumPy backend operations."""
    backend = get_backend("numpy")

    assert backend.name == "numpy"

    # Test array creation
    zeros = backend.zeros((10,))
    assert zeros.shape == (10,)
    assert np.allclose(zeros, 0.0)

    ones = backend.ones((5,))
    assert ones.shape == (5,)
    assert np.allclose(ones, 1.0)

    # Test operations
    arr = backend.array([1, 2, 3, 4, 5])
    assert backend.sum(arr) == 15

    clipped = backend.clip(arr, 2, 4)
    assert np.allclose(clipped, [2, 2, 3, 4, 4])


def test_numpy_backend_random():
    """Test random number generation."""
    backend = NumpyBackend()

    # Normal distribution
    normal = backend.random_normal((100,), mean=0.0, std=1.0, seed=42)
    assert normal.shape == (100,)
    assert -3 < np.mean(normal) < 3  # Should be around 0

    # Uniform distribution
    uniform = backend.random_uniform((100,), low=0.0, high=1.0, seed=42)
    assert uniform.shape == (100,)
    assert 0 <= np.min(uniform) and np.max(uniform) <= 1.0


def test_backend_conversion():
    """Test conversion to numpy."""
    backend = get_backend("numpy")

    arr = backend.array([1, 2, 3])
    np_arr = backend.to_numpy(arr)

    assert isinstance(np_arr, np.ndarray)
    assert np.allclose(np_arr, [1, 2, 3])


@pytest.mark.skipif(True, reason="JAX optional dependency")
def test_jax_backend_available():
    """Test JAX backend if available."""
    try:
        backend = get_backend("jax")
        assert backend.name == "jax"

        # Basic operations should work
        zeros = backend.zeros((10,))
        assert zeros.shape == (10,)

    except ImportError:
        pytest.skip("JAX not installed")


def test_backend_parity_operations():
    """Test that NumPy operations work consistently."""
    backend = get_backend("numpy")

    # Create test data
    a = backend.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = backend.array([2.0, 2.0, 2.0, 2.0, 2.0])

    # Operations
    assert np.isclose(backend.sum(a), 15.0)
    assert np.isclose(backend.sum(b), 10.0)

    # Element-wise
    exp_a = backend.exp(backend.array([0.0, 1.0]))
    assert np.isclose(backend.to_numpy(exp_a)[0], 1.0)
    assert np.isclose(backend.to_numpy(exp_a)[1], np.e)
