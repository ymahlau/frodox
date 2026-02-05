import jax.numpy as jnp
import numpy as np

from frodox import gaussian_wave_1d, plane_wave_1d

# --- Test Data & Constants ---
SIZE = 100.0
DX = 1.0
EXPECTED_N = 100  # round(100.0 / 1.0)


class TestPlaneWave:
    def test_plane_wave_basic_initialization(self):
        """Test that plane_wave returns correct shape and value (1+0j)."""
        arr = plane_wave_1d(size=SIZE, dx=DX, double_precision=False)

        # Check Shape
        assert arr.shape == (EXPECTED_N,)

        # Check Dtype (default should be complex64)
        assert arr.dtype == jnp.complex64

        # Check Values (should be exactly 1.0 + 0.0j everywhere)
        # We verify real part is 1 and imaginary part is 0
        assert jnp.all(arr == 1.0 + 0j)

    def test_plane_wave_rounding(self):
        """Test that the size/dx rounding logic works as expected."""
        # 10.0 / 0.3 = 33.333... -> should round to 33
        arr = plane_wave_1d(size=10.0, dx=0.3)
        assert arr.shape == (33,)

    def test_plane_wave_double_precision(self):
        """Test that double_precision=True returns complex128."""
        # Warning: This test alters global JAX state to x64
        arr = plane_wave_1d(size=10.0, dx=1.0, double_precision=True)
        assert arr.dtype == jnp.complex128


class TestGaussianWave:
    def test_gaussian_shape_and_center(self):
        """Test shape and that the peak is perfectly centered."""
        size = 10.0
        dx = 1.0
        std = 2.0

        # n = 10. Indices 0..9. Center index should be 4.5?
        # The grid is x = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        # Center of physical domain is size/2 = 5.0.
        # So the peak should occur exactly at index 5 (where x=5.0).

        arr = gaussian_wave_1d(size=size, dx=dx, std=std)

        assert arr.shape == (10,)

        # The value at index 5 (x=5.0) should be 1.0 (the peak)
        # Because exp( - (5-5)^2 ... ) = exp(0) = 1
        assert jnp.isclose(arr[5], 1.0 + 0j)

        # Check symmetry: value at 4 should equal value at 6
        assert jnp.isclose(arr[4], arr[6])

    def test_gaussian_std_width(self):
        """Verify the physics: At x = center + std, value should be exp(-0.5)."""
        size = 20.0
        dx = 1.0
        std = 5.0

        arr = gaussian_wave_1d(size=size, dx=dx, std=std)

        # We want to check the value at x = 15.0 (center + std)
        # Since dx=1.0, x=15.0 corresponds to index 15.
        target_val = np.exp(-0.5)

        # Use abs() because arr is complex
        assert jnp.isclose(jnp.abs(arr[15]), target_val, atol=1e-5)

    def test_gaussian_cutoff(self):
        """Test that values below the cutoff are strictly zero."""
        size = 100.0
        dx = 1.0
        std = 5.0
        cutoff = 0.1

        arr = gaussian_wave_1d(size=size, dx=dx, std=std, cutoff=cutoff)

        # Check that no non-zero values exist below cutoff
        # We look at the magnitude
        magnitudes = jnp.abs(arr)

        # Create a mask of values that 'survived' the cutoff
        survivors = magnitudes[magnitudes > 0]

        # All survivors must be >= cutoff
        assert jnp.all(survivors >= cutoff)

        # Check that values that SHOULD be zero ARE zero
        # At the edges (e.g. index 0), x=0. Center=50. dist=50.
        # exp(-0.5 * (50/5)^2) = exp(-50) which is tiny, definitely < 0.1
        assert arr[0] == 0j

    def test_gaussian_double_precision(self):
        """Test complex128 support."""
        arr = gaussian_wave_1d(size=10.0, dx=1.0, std=2.0, double_precision=True)
        assert arr.dtype == jnp.complex128
        assert arr.real.dtype == jnp.float64


class TestEdgeCases:
    def test_single_point(self):
        """Test behavior when grid results in a single point."""
        # size=1.0, dx=1.0 -> n=1
        arr = plane_wave_1d(size=1.0, dx=1.0)
        assert arr.shape == (1,)
        assert arr[0] == 1.0 + 0j

    def test_tiny_dx_memory_safe(self):
        """
        Ensure code doesn't crash with slightly smaller dx,
        but still keep it small enough for unit tests (n=1000).
        """
        arr = gaussian_wave_1d(size=10.0, dx=0.01, std=1.0)
        assert arr.shape == (1000,)

    def test_cutoff_none_vs_cutoff_zero(self):
        """cutoff=None and cutoff=0.0 should theoretically behave similarly for positive functions,
        but cutoff=0.0 invokes jnp.where explicitly."""
        arr_none = gaussian_wave_1d(size=10.0, dx=1.0, std=2.0, cutoff=None)
        arr_zero = gaussian_wave_1d(size=10.0, dx=1.0, std=2.0, cutoff=0.0)

        assert jnp.array_equal(arr_none, arr_zero)
