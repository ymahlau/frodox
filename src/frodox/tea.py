import jax
import jax.numpy as jnp


def thin_element_approximation_1d(
    background_idx: float | jax.Array,
    material_idx: float | jax.Array,
    heights: jax.Array,
    fields: jax.Array,
    wavelength: float | jax.Array,
) -> jax.Array:
    """
    Applies a spatially dependent phase shift to a 1D optical field using the Thin Element Approximation (TEA).

    This method models the interaction of light with a thin optical component (such as a lens,
    grating, or phase mask) by treating the element as a phase transparency. It assumes the
    element is thin enough that diffraction effects *within* the medium are negligible,
    modulating only the phase of the wavefront based on the Optical Path Difference (OPD).

    The phase shift $\phi(x)$ is calculated as:
    $$ \phi(x) = \frac{2\pi}{\lambda} (n_{material} - n_{background}) \cdot h(x) $$

    Args:
        background_idx (float | jax.Array): The refractive index of the surrounding medium.
            Must be a scalar.
        material_idx (float | jax.Array): The refractive index of the element material.
            Must be a scalar.
        heights (jax.Array): A 1D array representing the physical thickness profile $h(x)$
            of the element at each spatial coordinate. Shape: ``(N,)``.
        fields (jax.Array): The incident complex optical field. Shape: ``(N,)``.
        wavelength (float | jax.Array): The vacuum wavelength ($\lambda$) of the light.
            Must be a scalar.

    Returns:
        jax.Array: The complex optical field after passing through the element.
        Shape: ``(N,)``.
    """
    # --- Input Sanitization (from your snippet) ---
    if isinstance(background_idx, float | int):
        background_idx = jnp.asarray(background_idx).flatten()
    background_idx = background_idx.flatten()

    if isinstance(material_idx, float | int):
        material_idx = jnp.asarray(material_idx).flatten()
    material_idx = material_idx.flatten()

    # Sanitization for the new wavelength parameter
    if isinstance(wavelength, float | int):
        wavelength = jnp.asarray(wavelength).flatten()
    wavelength = wavelength.flatten()

    # --- Assertions ---
    assert background_idx.size == 1
    assert material_idx.size == 1
    assert wavelength.size == 1
    assert heights.ndim == 1
    assert fields.ndim == 1
    assert heights.shape[0] == fields.shape[0]

    # --- Physics Implementation ---

    # k0 = 2 * pi / lambda
    k0 = 2 * jnp.pi / wavelength

    # OPD = (n_material - n_background) * height(x)
    delta_n = material_idx - background_idx
    opd = delta_n * heights

    # phi(x) = k0 * OPD
    phase_shift = k0 * opd

    # Field_out = Field_in * exp(i * phi)
    fields_out = fields * jnp.exp(1j * phase_shift)

    return fields_out
