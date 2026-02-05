import jax
import jax.numpy as jnp


def plane_wave_1d(
    size: float,
    dx: float,
    double_precision: bool = False,
) -> jax.Array:
    """Initialize a 1D plane wave (uniform field) on a discretized grid.

    This function creates a 1-dimensional JAX array filled with complex ones
    (1 + 0j), representing a plane wave with a wave vector of k=0.

    Args:
        size: The total physical size (length) of the 1D domain.
        dx: The spatial discretization step size (grid spacing).
        double_precision: If True, uses `jnp.complex128` and enables JAX 64-bit
            support. If False, uses `jnp.complex64`. Defaults to False.

    Returns:
        A JAX array of shape `(n,)`, where `n = round(size / dx)`, initialized
        with values of 1.0.

    Warning:
        Setting `double_precision=True` will execute
        `jax.config.update('jax_enable_x64', True)`. This enables 64-bit
        precision **globally** for the current JAX session, which may affect
        arrays created elsewhere in your code.
    """
    n = round(size / dx)
    if double_precision:
        jax.config.update("jax_enable_x64", True)
        dtype = jnp.complex128
    else:
        dtype = jnp.complex64
    arr = jnp.ones(shape=(n,), dtype=dtype)
    return arr


def gaussian_wave_1d(
    size: float,
    dx: float,
    std: float,
    cutoff: float | None = None,
    double_precision: bool = False,
) -> jax.Array:
    """
    Generates a centered 1D Gaussian wave profile.

    This function creates a Gaussian amplitude distribution centered within a domain
    of specified physical size. The output is cast to a complex data type suitable
    for wave propagation simulations.

    Args:
        size (float): The total physical length of the spatial domain.
        dx (float): The spatial sampling interval (pixel pitch).
        std (float): The standard deviation ($\sigma$) of the Gaussian profile, controlling
            the beam width.
        cutoff (float | None, optional): A threshold value. Amplitudes below this value
            are set to zero. Defaults to None.
        double_precision (bool, optional): If True, enables JAX's 64-bit precision mode
            (``jax_enable_x64``) and returns ``complex128``. If False, returns
            ``complex64``. Defaults to False.

    Returns:
        jax.Array: A 1D array representing the complex optical field.
        Shape: ``(round(size/dx),)``.

    Warning:
        Setting ``double_precision=True`` modifies the global JAX configuration
        via ``jax.config.update('jax_enable_x64', True)``. This will affect all
        subsequent JAX operations in the current process.
    """
    n = round(size / dx)

    if double_precision:
        jax.config.update("jax_enable_x64", True)
        dtype = jnp.complex128
        real_dtype = jnp.float64
    else:
        dtype = jnp.complex64
        real_dtype = jnp.float32

    # Generate the spatial grid
    x = jnp.arange(n, dtype=real_dtype) * dx
    center = size / 2.0

    # Calculate Gaussian envelope: exp( - (x - mu)^2 / (2 * sigma^2) )
    # We maintain peak amplitude of 1.0
    argument = -0.5 * ((x - center) / std) ** 2
    arr = jnp.exp(argument)

    # Apply cutoff if provided (zero out values below the threshold)
    if cutoff is not None:
        arr = jnp.where(arr < cutoff, 0.0, arr)

    # Cast to complex type to ensure consistency with physics solvers expecting complex arrays
    return arr.astype(dtype)
