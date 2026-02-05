import jax
import jax.numpy as jnp


def angular_spectrum_method_1d(
    fields: jax.Array,
    distance: jax.Array | float,
    dx: jax.Array | float,
    wavelength: jax.Array | float,
) -> jax.Array:
    """
    Propagates a 1D optical field over a specified distance using the Angular Spectrum Method (ASM).

    This function solves the Rayleigh-Sommerfeld diffraction problem by decomposing the input
    field into plane waves via the Fourier domain. It is an exact solution to the Helmholtz
    equation for a homogeneous medium.

    The implementation includes specific numerical enhancements for stability:
    1. **Coordinate Scaling:** Spatial variables are scaled internally by `min(wavelength, dx)`
       to prevent numerical underflow/overflow with small optical dimensions.
    2. **Phase Precision:** The transfer function is decomposed into an envelope and a
       carrier phase term ($e^{ik_z z} = e^{i(k_z - k)z} \cdot e^{ikz}$). The envelope phase
       $(k_z - k)$ is computed using the conjugate algebraic trick $\frac{-k_x^2}{k_z + k}$
       to maintain high precision in the paraxial regime where $k_z \approx k$.

    Args:
        fields (jax.Array): A 1D array of complex or real values representing the optical
            field amplitude/phase distribution at the source plane. Shape: ``(N,)``.
        distance (jax.Array | float): The axial propagation distance ($z$). Positive values
            indicate forward propagation. Must be a scalar.
        dx (jax.Array | float): The spatial sampling interval (pixel pitch) of the input
            field. Must be a scalar.
        wavelength (jax.Array | float): The optical wavelength ($\lambda$) of the source
            light. Must be a scalar.

    Returns:
        jax.Array: The propagated complex optical field at the destination plane.
        Shape: ``(N,)``.

    Notes:
        - The method automatically handles evanescent waves (where $k_x > k$) by using complex
          arithmetic, resulting in exponential decay for high spatial frequencies.
    """
    # Validate input dimensions
    assert fields.ndim == 1, "fields must be a 1D array"

    distance = jnp.asarray(distance).flatten()
    assert distance.size == 1

    dx = jnp.asarray(dx).flatten()
    assert dx.size == 1

    wavelength = jnp.asarray(wavelength).flatten()
    assert wavelength.size == 1

    # scale = 1
    scale = jnp.minimum(wavelength, dx)
    dx_scaled = dx / scale
    wavelength_scaled = wavelength / scale
    distance_scaled = distance / scale

    n = fields.shape[0]

    # kx ranges from -pi/dx to +pi/dx
    kx = 2 * jnp.pi * jnp.fft.fftfreq(n, d=dx_scaled)

    # 2. Wave Vector Magnitude (k = 2 * pi / lambda)
    # Reshape k for broadcasting: (N_wavelengths, 1) against kx's (N_pixels)
    k = (2 * jnp.pi / wavelength_scaled)[:, None]

    # 3. Axial Wave Vector (kz)
    # kz = sqrt(k^2 - kx^2).
    # We cast to complex to handle evanescent waves (where kx > k) automatically.
    # If kx^2 > k^2, the term inside sqrt is negative, resulting in imaginary kz (decay).
    kz = jnp.sqrt((k**2 - kx[None, :] ** 2) + 0j)

    # 4. Transfer Function (Propagator)
    # H = exp(i * kz * z)
    # To avoid "large number - large number" error, we use the conjugate trick:
    # sqrt(k^2 - kx^2) - k  ==  -kx^2 / (kz + k)
    delta_kz = -(kx[None, :] ** 2) / (kz + k)
    H_envelope = jnp.exp(1j * distance_scaled * delta_kz)
    global_phase = jnp.exp(1j * distance_scaled * k).flatten()

    # 5. Angular Spectrum Propagation
    # FFT -> Apply Phase Shift -> Inverse FFT
    fields_fft = jnp.fft.fft(fields)

    # Broadcast multiplication: H is (Waves, N), fields_fft is (N,)
    propagated_fft = fields_fft * H_envelope

    output_fields = jnp.fft.ifft(propagated_fft, axis=-1)
    result = output_fields[0] * global_phase  # remove size 1 axis

    assert result.ndim == 1
    assert result.shape[0] == n
    return result
