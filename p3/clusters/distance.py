from jax import Array

from jax import jit
import jax.numpy as jnp


@jit
def _euclidean_distances(A, B) -> Array:
    """
    Compute the Euclidean distances between two sets of points.

    Parameters
    ----------
    A: jnp.ndarray - The first set of points
    B: jnp.ndarray - The second set of points

    Returns
    -------
    distances: jnp.ndarray - The Euclidean distances between two sets of points
    """
    distances = jnp.sqrt(
        jnp.sum(
            (A[:, None, :] - B[None, :, :]) ** 2,
            axis=-1,
        )
    )
    return distances


@jit
def _pairwise_distance(X: Array) -> Array:
    """
    Compute the pairwise distance between the points.

    Parameters
    ----------
    X: Array - The input data

    Returns
    -------
    distances: Array - The pairwise distance between the points
    """
    X_norm = jnp.sum(X**2, axis=1).reshape(-1, 1)
    distances = X_norm + X_norm.T - 2 * X @ X.T
    distances = jnp.sqrt(jnp.maximum(distances, 0.0))

    return distances
