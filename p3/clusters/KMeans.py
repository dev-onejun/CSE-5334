from jax import Array

import jax
import jax.numpy as jnp
from jax import jit

from .distance import _euclidean_distances


@jit
def _weighted_choice(key, a, p):
    """
    Choose a random element from a with the probabilities p.

    Parameters
    ----------
    key: jax.random.KeyArray - The key for reproducibility
    a: jnp.ndarray - The array of elements
    p: jnp.ndarray - The array of probabilities

    Returns
    -------
    a[idx]: jnp.ndarray - The random element from a with the probabilities p
    """
    cumulative_probs = jnp.cumsum(p)
    r = jax.random.uniform(key, ()) * cumulative_probs[-1]
    idx = jnp.searchsorted(cumulative_probs, r, side="right")

    return a[idx]


@jit
def _assign_clusters(distances) -> Array:
    """
    Assign each data point to the nearest centroid.

    Parameters
    ----------
    distances: jnp.ndarray - The Euclidean distances between the data points and the centroids

    Returns
    -------
    labels: jnp.ndarray - The assigned cluster labels
    """
    return jnp.argmin(distances, axis=1)


@jit
def _compute_SSE(X, centroids, labels) -> Array:
    """
    Compute the sum of squared errors (SSE) of the current KMeans model.

    Returns
    -------
    sse: float - The sum of squared errors (SSE) of the current KMeans model
    """
    assigned_centroids = centroids[labels]
    errors = X - assigned_centroids
    squared_errors = jnp.sum(errors**2, axis=1)

    return jnp.sum(squared_errors)


class KMeans:
    def __init__(self, key, n_clusters=3, max_iter=0) -> None:
        """
        Initialize the KMeans class.

        Parameters
        ----------
        key: jax.random.KeyArray - The key for reproducibility
        n_clusters: int - The number of clusters
        max_iter: int - The maximum number of iterations
        """
        assert n_clusters > 0, "The number of clusters must be larger than 0"

        self.__key = key
        self.__n_clusters: int = n_clusters
        self.__max_iter: int = max_iter

        self.__centroids: Array = jnp.array([])
        self.__labels: Array = jnp.array([])
        self.__X: Array = jnp.array([])
        self.__SSE: Array = jnp.array([])

    def __init_centroids(self) -> Array:
        """
        Initialize the centroids using the KMeans++ algorithm.

        Returns
        -------
        centroids: jnp.ndarray - The initialized centroids
        """
        n_clusters, X = self.__n_clusters, self.__X

        n_samples = X.shape[0]
        keys = jax.random.split(self.__key, n_clusters)

        idx = jax.random.randint(keys[0], (), 0, n_samples)
        centroids = [X[idx]]

        for i in range(1, n_clusters):
            distances = jnp.min(
                jnp.stack([jnp.sum((X - c) ** 2, axis=1) for c in centroids], axis=1),
                axis=1,
            )
            total_distance = jnp.sum(distances)
            probabilities = distances / total_distance

            idx = _weighted_choice(keys[i], jnp.arange(n_samples), probabilities)
            centroids.append(X[idx])

        return jnp.array(centroids)

    def __compute_distances(self):
        """
        Compute the Euclidean distances between the data points and the centroids.

        Returns
        -------
        distances: jnp.ndarray - The Euclidean distances between the data points and the centroids
        """
        return _euclidean_distances(self.__X, self.__centroids)

    def __compute_new_centroids(self) -> Array:
        """
        Compute the new centroids based on the assigned clusters.

        Returns
        -------
        centroids: jnp.ndarray - The new centroids
        """
        X, labels, n_clusters = self.__X, self.__labels, self.__n_clusters

        centroids_sum = jax.ops.segment_sum(X, labels, num_segments=n_clusters)
        counts = jnp.bincount(labels, minlength=n_clusters).reshape(-1, 1)
        counts = jnp.where(counts == 0, 1, counts)
        centroids = centroids_sum / counts

        return centroids

    def fit(self, X):
        """
        Train the KMeans model to the given data.

        Parameters
        ----------
        X: jnp.ndarray - The data to fit the KMeans model

        Returns
        -------
        iter_num: int - The number of iterations to converge
        """
        self.__X: Array = X
        self.__centroids = self.__init_centroids()

        iter_num = 0
        for i in range(1, self.__max_iter + 1):
            iter_num = i

            distances = self.__compute_distances()
            self.__labels = _assign_clusters(distances)
            new_centroids: Array = self.__compute_new_centroids()

            if jnp.allclose(self.__centroids, new_centroids, atol=1e-6):
                break
            self.__centroids = new_centroids

        return iter_num

    def evaluate(self) -> Array:
        """
        Evaluate the KMeans model by computing the sum of squared errors (SSE).

        Returns
        -------
        float - The sum of squared errors (SSE) of the KMeans model
        """
        self.__SSE = _compute_SSE(self.__X, self.__centroids, self.__labels)

        return self.__SSE
