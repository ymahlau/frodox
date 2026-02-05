import unittest

import jax
import jax.numpy as jnp
import numpy as np

from frodox import binary_grating_heights_1d


class TestBinaryGrating(unittest.TestCase):
    def setUp(self):
        # Allow checking all values in large arrays if necessary,
        # though we keep arrays small for performance.
        jax.config.update("jax_enable_x64", True)

    def test_basic_geometry_exact_alignment(self):
        """
        Test a simple case where features align perfectly with pixels.
        Grating: 6 pixels wide.
        Material: 1 pixel.
        Gap: 1 pixel.
        Expected: 1, 0, 1, 0, 1, 0
        """
        dx = 1.0
        result = binary_grating_heights_1d(grating_width=6.0, material_width=1.0, gap_width=1.0, dx=dx)
        expected = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        self.assertEqual(result.shape, (6,))
        np.testing.assert_array_equal(result, expected)

    def test_duty_cycle_variation(self):
        """
        Test a 75% duty cycle (3 pixels material, 1 pixel gap).
        Total width 8 pixels.
        Expected: 1,1,1,0, 1,1,1,0
        """
        dx = 0.5
        # material = 1.5 (3 pixels), gap = 0.5 (1 pixel)
        result = binary_grating_heights_1d(
            grating_width=4.0,  # 8 pixels
            material_width=1.5,
            gap_width=0.5,
            dx=dx,
        )
        expected = jnp.array([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_full_width_padding_centering(self):
        """
        Test that the grating is centered within a larger full_width.
        Grating: 2 pixels (1 mat, 1 gap).
        Full Width: 6 pixels.
        Padding: (6 - 2) = 4 total. 2 left, 2 right.
        Expected: 0, 0, [1, 0], 0, 0
        """
        dx = 1.0
        result = binary_grating_heights_1d(grating_width=2.0, material_width=1.0, gap_width=1.0, dx=dx, full_width=6.0)
        expected = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_padding_odd_difference(self):
        """
        Test padding when the difference is odd (asymmetric padding).
        Grating: 2 pixels.
        Full Width: 5 pixels.
        Padding: 3 total. 1 left, 2 right (due to integer division // 2).
        Expected: 0, [1, 0], 0, 0
        """
        dx = 1.0
        result = binary_grating_heights_1d(grating_width=2.0, material_width=1.0, gap_width=1.0, dx=dx, full_width=5.0)
        expected = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_cut_off_pattern(self):
        """
        Test when grating_width cuts off the pattern mid-period.
        Period = 4 pixels (2 mat, 2 gap).
        Width = 5 pixels.
        Expected: 1, 1, 0, 0, 1
        """
        dx = 1.0
        result = binary_grating_heights_1d(grating_width=5.0, material_width=2.0, gap_width=2.0, dx=dx)
        expected = jnp.array([1.0, 1.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_edge_case_zero_gap(self):
        """Test a solid block of material (gap_width = 0)."""
        result = binary_grating_heights_1d(grating_width=5.0, material_width=1.0, gap_width=0.0, dx=1.0)
        # Should be all 1s
        np.testing.assert_array_equal(result, jnp.ones(5))

    def test_edge_case_zero_material(self):
        """Test an empty space (material_width = 0)."""
        result = binary_grating_heights_1d(grating_width=5.0, material_width=0.0, gap_width=1.0, dx=1.0)
        # Should be all 0s
        np.testing.assert_array_equal(result, jnp.zeros(5))

    def test_validation_error(self):
        """Ensure ValueError is raised if full_width < grating_width."""
        with self.assertRaises(ValueError):
            binary_grating_heights_1d(
                grating_width=10.0,
                material_width=1.0,
                gap_width=1.0,
                dx=1.0,
                full_width=5.0,  # Too small
            )

    def test_jit_compilation(self):
        """
        Verify the function can be JIT compiled.
        Since arguments determine array shapes (via `round`),
        they must be static arguments for JIT.
        """
        # We wrap it in jit, marking all arguments as static because
        # they determine the shape of the output array (jnp.arange size).
        jit_func = jax.jit(
            binary_grating_heights_1d,
            static_argnames=["grating_width", "material_width", "gap_width", "dx", "full_width"],
        )

        # Run it to trigger compilation
        result = jit_func(grating_width=4.0, material_width=1.0, gap_width=1.0, dx=1.0)
        expected = jnp.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_floating_point_resolution(self):
        """
        Test with non-integer inputs to ensure robust floating point handling.
        dx = 0.1
        Material = 0.2 (2 pixels), Gap = 0.1 (1 pixel)
        """
        dx = 0.1
        result = binary_grating_heights_1d(
            grating_width=0.6,  # 6 pixels
            material_width=0.2,
            gap_width=0.1,
            dx=dx,
        )
        # Period is 3 pixels (110)
        # Sequence: 1, 1, 0, 1, 1, 0
        expected = jnp.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
