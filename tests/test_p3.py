from p3.utils import load_data, set_seed
from p3.clusters import KMeans, AgglomerativeClustering
from p3.clusters.KMeans import _euclidean_distances

from sklearn.cluster import KMeans as sk_KMeans
from sklearn.cluster import AgglomerativeClustering as sk_AgglomerativeClustering

import time
import jax.numpy as jnp
import numpy as np

key = set_seed(1)
X, _ = load_data()
X_np = np.array(X)


def test_euclidean_distances():
    # Test Case 1: Single points
    A = jnp.array([[1.0, 2.0]])
    B = jnp.array([[4.0, 6.0]])
    expected_distance = jnp.sqrt((4.0 - 1.0) ** 2 + (6.0 - 2.0) ** 2)
    result = _euclidean_distances(A, B)
    assert jnp.isclose(result, expected_distance).all(), f"Test Case 1 Failed: {result}"

    # Test Case 2: Zero distance
    A = jnp.array([[1.0, 2.0]])
    B = jnp.array([[1.0, 2.0]])
    expected_distance = 0.0
    result = _euclidean_distances(A, B)
    assert jnp.isclose(result, expected_distance).all(), f"Test Case 2 Failed: {result}"

    # Test Case 3: Higher dimensions
    A = jnp.array([[1.0, 2.0, 3.0]])
    B = jnp.array([[4.0, 5.0, 6.0]])
    expected_distance = jnp.sqrt((4.0 - 1.0) ** 2 + (5.0 - 2.0) ** 2 + (6.0 - 3.0) ** 2)
    result = _euclidean_distances(A, B)
    assert jnp.isclose(result, expected_distance).all(), f"Test Case 3 Failed: {result}"


def test_kmeans():
    sk_results, results = [], []
    for K in range(2, 9):
        start_time = time.time()

        sk_kmeans = sk_KMeans(
            n_clusters=K,
            init="k-means++",
            max_iter=1000,
        )
        sk_kmeans.fit(X_np)
        print(f"sklearn: K = {K} took {time.time() - start_time:.4f} seconds")

        sk_results.append(sk_kmeans.inertia_)

        start_time = time.time()

        kmeans = KMeans(key=key, n_clusters=K, max_iter=1000)
        kmeans.fit(X)

        print(f"JAX:\t K = {K} took {time.time() - start_time:.4f} seconds")

        results.append(kmeans.evaluate().tolist())

    print(f"sk_results: {sk_results}")
    print(f"results: {results}")
    assert sk_results == results


def test_agglomerative_clustering():
    sk_agglomerative = sk_AgglomerativeClustering(
        n_clusters=3,
        metric="euclidean",
        linkage="single",
    )
    sk_agglomerative.fit(X)
    sk_agglomerative.labels_

    agglomerative = AgglomerativeClustering(n_clusters=3)
    agglomerative.fit(X)
    agglomerative.labels
