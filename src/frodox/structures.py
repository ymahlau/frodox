import jax
import jax.numpy as jnp


def binary_grating_heights_1d(
    grating_width: float,
    material_width: float,
    gap_width: float,
    dx: float,
    full_width: float | None = None,
) -> jax.Array:
    """
    Generates a 1D binary grating height map (0.0 for gaps, 1.0 for material).

    Args:
        grating_width: Total length of the patterned region (meters).
        material_width: Width of the structure/high region (meters).
        gap_width: Width of the gap/low region (meters).
        dx: Grid resolution (meters/pixel).
        full_width: Optional total simulation domain size. If provided, the
                    grating is centered within this width, padded by zeros. Defaults to zero.
    """

    # 1. Create the grating pattern
    n_grating = round(grating_width / dx)
    x = jnp.arange(n_grating) * dx

    period = material_width + gap_width

    # Use modulo arithmetic to create the repeating binary structure
    # Returns 1.0 where we are inside the material_width, 0.0 otherwise
    grating_profile = jnp.where(jnp.mod(x, period) < material_width, 1.0, 0.0)

    # 2. Handle Padding (if full_width is provided)
    if full_width is not None:
        n_total = round(full_width / dx)

        if n_total < n_grating:
            raise ValueError("full_width must be greater than or equal to grating_width")

        # Calculate padding to center the grating
        pad_total = n_total - n_grating
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        # Pad with 0.0 (gaps)
        grating_profile = jnp.pad(grating_profile, (pad_left, pad_right), mode="constant", constant_values=0.0)

    return grating_profile
