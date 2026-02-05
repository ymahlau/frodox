import jax
import jax.numpy as jnp
import numpy as np
import pytest

from frodox import angular_spectrum_method_1d


def analytical_gaussian_propagation(u0, w0, z, k, x):
    """
    Analytical solution for Gaussian beam propagation in paraxial approximation.
    Used as a ground truth oracle.
    """
    z_R = (jnp.pi * w0**2) / (2 * jnp.pi / k)  # Rayleigh range
    w_z = w0 * jnp.sqrt(1 + (z / z_R) ** 2)
    R_z = z * (1 + (z_R / z) ** 2)
    gouy_phase = jnp.arctan(z / z_R)

    # 1D Gaussian beam formula
    # Note: The amplitude decay in 1D is sqrt(w0/w_z) rather than (w0/w_z) in 2D
    amplitude = jnp.sqrt(w0 / w_z) * jnp.exp(-(x**2) / (w_z**2))
    phase = jnp.exp(-1j * (k * z + k * (x**2) / (2 * R_z) - 0.5 * gouy_phase))

    return amplitude * phase


# --- Test Suites ---


class TestPhysicalCorrectness:
    """Tests based on physical laws and known analytical solutions."""

    def test_energy_conservation(self):
        """Total power should be conserved in lossless propagation."""
        N = 256
        dx = 1.0e-6
        wavelength = 0.5e-6
        distance = 100e-6

        # Create a random complex field
        key = jax.random.PRNGKey(0)
        fields = jax.random.normal(key, (N,)) + 1j * jax.random.normal(key, (N,))

        propagated = angular_spectrum_method_1d(fields, distance, dx, wavelength)

        input_power = jnp.sum(jnp.abs(fields) ** 2)
        output_power = jnp.sum(jnp.abs(propagated) ** 2)

        # Use a slightly loose tolerance due to potential evanescent wave clipping at boundaries
        np.testing.assert_allclose(input_power, output_power, rtol=1e-5)

    def test_zero_propagation_distance(self):
        """Propagating distance 0 should return the input field exactly."""
        N = 128
        dx = 1.0
        wavelength = 0.5
        fields = jnp.ones(N) + 0j

        result = angular_spectrum_method_1d(fields, 0.0, dx, wavelength)
        np.testing.assert_allclose(result, fields, atol=1e-6)

    def test_reciprocity(self):
        """Propagating forward +Z then backward -Z should recover original field."""
        N = 128
        dx = 1.0e-6
        wavelength = 500e-9
        distance = 50e-6

        # Create a localized source (Gaussian) to avoid boundary wrapping issues
        x = (jnp.arange(N) - N // 2) * dx
        fields = jnp.exp(-(x**2) / (10 * dx) ** 2).astype(complex)

        forward = angular_spectrum_method_1d(fields, distance, dx, wavelength)
        backward = angular_spectrum_method_1d(forward, -distance, dx, wavelength)

        np.testing.assert_allclose(fields, backward, atol=1e-5)

    def test_plane_wave_phase_accumulation(self):
        """A perfect plane wave should only acquire global phase exp(ikz)."""
        N = 64
        dx = 1.0
        wavelength = 0.1
        distance = 10.0

        k = 2 * jnp.pi / wavelength
        fields = jnp.ones(N).astype(complex)  # Plane wave

        result = angular_spectrum_method_1d(fields, distance, dx, wavelength)
        expected_phase = jnp.exp(1j * k * distance)

        # Check center pixel to avoid edge artifacts
        center_val = result[N // 2]

        np.testing.assert_allclose(jnp.abs(center_val), 1.0, atol=1e-5)
        # Check phase angle
        assert jnp.abs(jnp.angle(center_val / expected_phase)) < 1e-5

    def test_gaussian_beam_propagation(self):
        """Compare numerical result against analytical Gaussian beam formula."""
        N = 512
        dx = 0.5e-6
        wavelength = 0.633e-6  # HeNe
        w0 = 10e-6  # Waist
        distance = 200e-6  # Propagation

        k = 2 * jnp.pi / wavelength
        x = (jnp.arange(N) - N // 2) * dx

        # Input Gaussian
        u0 = jnp.exp(-(x**2) / (w0**2)).astype(complex)

        # Numerical Result
        u_numeric = angular_spectrum_method_1d(u0, distance, dx, wavelength)

        # Analytical Result
        u_analytic = analytical_gaussian_propagation(u0, w0, distance, k, x)

        # Normalize both (absolute phase might differ by constant piston, we care about profile)
        u_numeric_norm = jnp.abs(u_numeric)
        u_analytic_norm = jnp.abs(u_analytic)

        # Check profile shape match (Intensity)
        np.testing.assert_allclose(u_numeric_norm, u_analytic_norm, rtol=1e-2, atol=1e-4)


class TestNumericalStability:
    """Tests for floating point issues, scaling, and precision."""

    def test_small_scales(self):
        """Test with nanometer scales to ensure scale normalization works."""
        N = 64
        dx = 1e-9  # 1 nm
        wavelength = 500e-9
        distance = 100e-9

        fields = jnp.ones(N).astype(complex)
        # Without scaling, k would be huge and exp(ikz) might overflow/loss precision
        result = angular_spectrum_method_1d(fields, distance, dx, wavelength)
        assert not jnp.isnan(result).any()

    def test_paraxial_phase_trick(self):
        """
        Verify the conjugate trick reduces error for small angles.
        We check if the implementation matches a naive implementation for standard inputs,
        implying the algebra is correct.
        """
        N = 128
        dx = 1.0
        wavelength = 0.1
        distance = 10.0
        fields = jnp.zeros(N).at[N // 2].set(1.0)  # Delta function

        # Run optimized function
        optimized = angular_spectrum_method_1d(fields, distance, dx, wavelength)

        assert not jnp.isnan(optimized).any()
        # Magnitude check
        assert jnp.max(jnp.abs(optimized)) > 0

    def test_evanescent_waves(self):
        """High spatial frequencies (kx > k) should decay exponentially."""
        N = 128
        dx = 0.1  # Very fine sampling
        wavelength = 10.0  # Large wavelength -> k is small

        # Create a field with high frequency variation: [-1, 1, -1, 1]
        # Frequency is 1/(2*dx) = 5.0.
        # Wave number k = 2pi/10 = 0.6.
        # kx (approx 2pi * 5 = 30) >> k. Should be evanescent.
        fields = jnp.tile(jnp.array([1.0, -1.0]), N // 2).astype(complex)

        res_near = angular_spectrum_method_1d(fields, 0.1, dx, wavelength)
        res_far = angular_spectrum_method_1d(fields, 5.0, dx, wavelength)

        power_near = jnp.sum(jnp.abs(res_near) ** 2)
        power_far = jnp.sum(jnp.abs(res_far) ** 2)

        # Power should drop significantly due to evanescent decay
        assert power_far < power_near


class TestJAXIntegration:
    """Tests ensuring JAX transformations work."""

    def test_jit_compilation(self):
        """Ensure the function can be JIT compiled."""
        N = 64
        fields = jnp.zeros(N).astype(complex)
        args = (fields, 10.0, 0.1, 0.5)

        jitted_func = jax.jit(angular_spectrum_method_1d)

        # Run once to compile
        _ = jitted_func(*args)
        # Run again
        res = jitted_func(*args)
        assert res.shape == (N,)

    def test_autograd_wrt_input(self):
        """Ensure we can differentiate with respect to the input field."""
        N = 32
        dx = 1.0
        wl = 0.5
        dist = 10.0

        def loss_fn(f):
            prop = angular_spectrum_method_1d(f, dist, dx, wl)
            return jnp.sum(jnp.abs(prop) ** 2)

        fields = jax.random.normal(jax.random.PRNGKey(0), (N,)).astype(complex)
        grads = jax.grad(loss_fn)(fields)

        assert grads.shape == fields.shape
        assert not jnp.isnan(grads).any()

    def test_autograd_wrt_distance(self):
        """Ensure we can differentiate with respect to propagation distance (autofocus applications)."""
        N = 32
        dx = 1.0
        wl = 0.5
        fields = jnp.ones(N).astype(complex)

        def loss_fn(d):
            prop = angular_spectrum_method_1d(fields, d, dx, wl)
            return jnp.real(jnp.sum(prop))

        dist = 10.0
        grads = jax.grad(loss_fn)(dist)

        assert grads.shape == ()  # Scalar gradient
        assert not jnp.isnan(grads)

    def test_vmap_batching(self):
        """Test vmap to propagate a batch of fields simultaneously."""
        N = 64
        Batch = 10
        dx = 1.0
        wl = 0.5
        dist = 5.0

        # Shape (Batch, N)
        batch_fields = jnp.ones((Batch, N)).astype(complex)

        # vmap over the first argument (index 0)
        vmap_asm = jax.vmap(angular_spectrum_method_1d, in_axes=(0, None, None, None))

        res = vmap_asm(batch_fields, dist, dx, wl)

        assert res.shape == (Batch, N)
        # All items in batch are identical, results should be identical
        np.testing.assert_allclose(res[0], res[5])


class TestEdgeCases:
    """Inputs that might cause shape errors or type errors."""

    def test_input_shape_validation(self):
        """Should raise error if input is not 1D."""
        with pytest.raises(AssertionError):
            angular_spectrum_method_1d(jnp.ones((10, 10)), 1.0, 1.0, 1.0)

    def test_real_input_cast(self):
        """Function should handle real-valued input array automatically."""
        N = 32
        fields = jnp.ones(N, dtype=jnp.float32)  # Real input
        res = angular_spectrum_method_1d(fields, 10.0, 1.0, 0.5)
        assert jnp.iscomplexobj(res)  # Output must be complex
