"""
Programming Assignment 3
CSE-5334 Data Mining, Fall 2024
Professor: Dr. Marnim Galib

Wonjun Park
UTA ID: 1002237177
Computer Science and Engineering, University of Texs at Arlington, Arlington, TX, USA
wxp7177@mavs.uta.edu
"""

from jax import Array

import numpy as np

from utils import load_data, set_seed, plot_SSE, plot_dendrogram
from clusters import KMeans, AgglomerativeClustering


def train_kmeans(X, K=3):
    kmeans = KMeans(key=key, n_clusters=K, max_iter=1000)

    iter_num = kmeans.fit(X)

    SSE = kmeans.evaluate()

    print(f"For k = {K} After {iter_num} iterations: SSE = {SSE}")

    return SSE


def train_agglomerative(X, linkage) -> tuple[np.ndarray, float]:
    agglomerative_clustering = AgglomerativeClustering(
        metric="euclidean", linkage=linkage
    )

    linkage = agglomerative_clustering.fit(X)
    labels = agglomerative_clustering.get_cluster_labels(3)
    silhouette_score = agglomerative_clustering.silhouette_coefficient(labels)

    return linkage, silhouette_score


def main():
    X: Array = load_data()[0]

    """
    Task 1: KMeans Clustering
    """
    # SSE_history = []
    # for K in range(2, 9):
    # SSE = train_kmeans(X, K)
    # SSE_history.append((K, SSE))

    # plot_SSE(SSE_history)

    """
    Task 2: Agglomerative Hierarchical Clustering
    """
    # linkage, sihouette_score = train_agglomerative(X, "single")
    # plot_dendrogram(linkage)
    # print(f"Silhouette Score: {sihouette_score}")

    linkage, sihouette_score = train_agglomerative(X, "complete")
    plot_dendrogram(linkage)
    print(f"Silhouette Score: {sihouette_score}")


if __name__ == "__main__":
    key = set_seed(0)

    main()
