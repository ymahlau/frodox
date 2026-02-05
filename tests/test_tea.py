import jax
import jax.numpy as jnp
import numpy as np
import pytest

from frodox import thin_element_approximation_1d

# --- Tests ---


class TestThinElementApproximation:
    def setup_method(self):
        """Standard setup for low-compute tests."""
        self.N = 50
        self.wavelength = 0.5  # microns
        self.n_air = 1.0
        self.n_glass = 1.5

        # Create a simple flat wavefront
        self.fields_flat = jnp.ones(self.N, dtype=jnp.complex64)
        # Create a linear ramp height profile (wedge)
        self.heights_ramp = jnp.linspace(0, 1.0, self.N)

    def test_identity_zero_thickness(self):
        """Physics: If height is 0 everywhere, field should not change."""
        zeros = jnp.zeros(self.N)
        out = thin_element_approximation_1d(self.n_air, self.n_glass, zeros, self.fields_flat, self.wavelength)
        np.testing.assert_allclose(out, self.fields_flat, atol=1e-6)

    def test_identity_matched_index(self):
        """Physics: If n_material == n_background, field should not change."""
        out = thin_element_approximation_1d(
            self.n_glass, self.n_glass, self.heights_ramp, self.fields_flat, self.wavelength
        )
        np.testing.assert_allclose(out, self.fields_flat, atol=1e-6)

    def test_pi_phase_shift(self):
        """
        Physics: Verify a specific height produces exactly a Pi phase shift (sign flip).
        Formula: phi = (2pi/lambda) * (dn) * h
        For phi = pi: h = lambda / (2 * dn)
        """
        dn = self.n_glass - self.n_air
        h_pi = self.wavelength / (2 * dn)

        heights = jnp.full((self.N,), h_pi)

        out = thin_element_approximation_1d(self.n_air, self.n_glass, heights, self.fields_flat, self.wavelength)

        # e^(i*pi) = -1, so output should be negative of input
        expected = -1.0 * self.fields_flat
        np.testing.assert_allclose(out, expected, atol=1e-5)

    def test_2pi_phase_shift_wrapping(self):
        """Physics: A 2Pi phase shift should return the field to its original state."""
        dn = self.n_glass - self.n_air
        h_2pi = self.wavelength / dn

        heights = jnp.full((self.N,), h_2pi)

        out = thin_element_approximation_1d(self.n_air, self.n_glass, heights, self.fields_flat, self.wavelength)

        np.testing.assert_allclose(out, self.fields_flat, atol=1e-5)

    def test_input_types_flexibility(self):
        """JAX/Python: Ensure function handles floats, ints, and 0-rank arrays."""
        # Mix of float, int, and jax arrays
        bg = 1.0  # float
        mat = jnp.array(1.5)  # 0-rank array
        wl = 1  # int

        out = thin_element_approximation_1d(bg, mat, self.heights_ramp, self.fields_flat, wl)
        assert out.shape == (self.N,)
        assert out.dtype == jnp.complex64 or out.dtype == jnp.complex128

    def test_jit_compilation(self):
        """JAX: Ensure the function can be JIT compiled."""
        jit_tea = jax.jit(thin_element_approximation_1d)

        # First run (compilation)
        out1 = jit_tea(self.n_air, self.n_glass, self.heights_ramp, self.fields_flat, self.wavelength)
        # Second run (execution)
        out2 = jit_tea(self.n_air, self.n_glass, self.heights_ramp, self.fields_flat, self.wavelength)

        assert out1.shape == (self.N,)
        np.testing.assert_allclose(out1, out2)

    def test_autodiff_gradients(self):
        """JAX: Ensure we can differentiate with respect to heights (inverse design)."""

        def loss_fn(h):
            field_out = thin_element_approximation_1d(self.n_air, self.n_glass, h, self.fields_flat, self.wavelength)
            # Dummy loss: maximize real part
            return -jnp.sum(jnp.real(field_out))

        # Compute gradient w.r.t heights
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(self.heights_ramp)

        assert grads.shape == self.heights_ramp.shape
        assert not jnp.any(jnp.isnan(grads))

    def test_vmap_batching(self):
        """JAX: Ensure vmap works (e.g., processing a batch of different height profiles)."""
        batch_size = 5
        # Create a batch of heights: (Batch, N)
        batch_heights = jnp.stack([self.heights_ramp * i for i in range(batch_size)])

        # vmap over the 'heights' argument (index 2)
        vmap_tea = jax.vmap(thin_element_approximation_1d, in_axes=(None, None, 0, None, None))

        batch_out = vmap_tea(self.n_air, self.n_glass, batch_heights, self.fields_flat, self.wavelength)

        assert batch_out.shape == (batch_size, self.N)

    def test_shape_mismatch_errors(self):
        """Validation: Ensure assertions catch shape mismatches."""
        bad_heights = jnp.zeros(self.N + 1)  # Mismatched length

        with pytest.raises(AssertionError):
            thin_element_approximation_1d(self.n_air, self.n_glass, bad_heights, self.fields_flat, self.wavelength)

    def test_scalar_validation_errors(self):
        """Validation: Ensure non-scalar indices raise errors."""
        bad_idx = jnp.array([1.0, 1.5])  # Not a scalar

        with pytest.raises(AssertionError):
            thin_element_approximation_1d(bad_idx, self.n_glass, self.heights_ramp, self.fields_flat, self.wavelength)
