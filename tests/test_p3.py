from p3.utils import load_data, set_seed
from p3.clusters import KMeans, AgglomerativeClustering

from sklearn.cluster import KMeans as sk_KMeans
from sklearn.cluster import AgglomerativeClustering as sk_AgglomerativeClustering

import time
import jax.numpy as jnp

key = set_seed(42)
X, _ = load_data()


def test_kmeans():
    sk_results, results = [], []
    for K in range(2, 9):
        start_time = time.time()

        sk_kmeans = sk_KMeans(
            n_clusters=K,
            init="k-means++",
            max_iter=1000,
        )
        sk_kmeans.fit(X)
        print(f"sklearn: K = {K} took {time.time() - start_time:.4f} seconds")

        sk_results.append(sk_kmeans.inertia_)

        start_time = time.time()

        kmeans = KMeans(key=key, n_clusters=K, max_iter=1000)
        kmeans.fit(X)

        print(f"JAX:\t K = {K} took {time.time() - start_time:.4f} seconds")

        results.append(kmeans.evaluate())

    sk_results, results = jnp.array(sk_results), jnp.array(results)
    assert jnp.allclose(sk_results, results, atol=1e-6)


def test_agglomerative_clustering():
    sk_agglomerative = sk_AgglomerativeClustering(n_clusters=3)
    sk_agglomerative.fit(X)
    sk_agglomerative.labels_

    agglomerative = AgglomerativeClustering(n_clusters=3)
    agglomerative.fit(X)
    agglomerative.labels
