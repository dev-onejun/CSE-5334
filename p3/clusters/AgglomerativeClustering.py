from jax import Array

import jax.numpy as jnp
import numpy as np

from .distance import _pairwise_distance


class AgglomerativeClustering:
    def __init__(self, metric, linkage) -> None:
        assert linkage in ["single", "complete"], "Invalid linkage type"
        assert metric in ["euclidean"], "Invalid metric type"

        if linkage == "single":
            self.fit = self._single_linkage
        elif linkage == "complete":
            self.fit = self._complete_linkage

        if metric == "euclidean":
            self.__metric = _pairwise_distance

    def _single_linkage(self, X) -> np.ndarray:
        """
        Single Linkage Clustering

        Parameters
        ----------
        X: jnp.ndarray - The input data

        Returns
        -------
        linkage: np.ndarray - The linkage matrix
        """
        self.__X = X

        n_samples = X.shape[0]
        distance = self.__metric(X)
        distance = np.array(distance)

        clusters = [[i] for i in range(n_samples)]
        linkage = []
        active_clusters = [True] * n_samples
        cluster_ids = list(range(n_samples))

        while len(cluster_ids) > 1:
            # Mask distances
            active_indices = [i for i, active in enumerate(active_clusters) if active]
            distance_active = distance[np.ix_(active_indices, active_indices)]
            np.fill_diagonal(distance_active, np.inf)

            # Find the pair of clusters
            i, j = np.unravel_index(np.argmin(distance_active), distance_active.shape)
            idx_i, idx_j = active_indices[i], active_indices[j]
            if idx_i > idx_j:
                idx_i, idx_j = idx_j, idx_i

            # Merge Clusters
            new_cluster = clusters[idx_i] + clusters[idx_j]
            clusters.append(new_cluster)
            active_clusters.append(True)
            linkage.append([idx_i, idx_j, distance[idx_i, idx_j], len(new_cluster)])

            # Update active clusters
            active_clusters[idx_i] = False
            active_clusters[idx_j] = False

            # Update distance matrix for the new cluster
            new_distances = np.minimum(
                distance[idx_i, :],
                distance[idx_j, :],
            )
            distance = np.vstack((distance, new_distances))
            new_distances = np.append(new_distances, np.inf)
            distance = np.column_stack((distance, new_distances))

            # Update cluster ids
            cluster_ids = [i for i, active in enumerate(active_clusters) if active]

        self.__linkage = np.array(linkage)
        return self.__linkage

    def _complete_linkage(self, X) -> np.ndarray:
        """
        Complete Linkage Clustering

        Parameters
        ----------
        X: jnp.ndarray - The input data

        Returns
        -------
        linkage: np.ndarray - The linkage matrix
        """
        self.__X = X

        n_samples = X.shape[0]
        D_p = self.__metric(X)
        D_p = np.array(D_p)  # Convert to NumPy array for indexing operations
        np.fill_diagonal(D_p, np.inf)  # Set diagonals to infinity to avoid self-merging

        clusters = [[i] for i in range(n_samples)]
        linkage = []

        # Create an array to keep track of cluster IDs
        cluster_ids = np.arange(n_samples)

        for step in range(n_samples - 1):
            # Find the two clusters with the smallest maximum distance
            min_dist = np.inf
            idx_i, idx_j = -1, -1

            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Compute the maximum distance between points in cluster i and cluster j
                    points_i = clusters[i]
                    points_j = clusters[j]
                    distances = D_p[np.ix_(points_i, points_j)]
                    max_distance = np.max(distances)
                    if max_distance < min_dist:
                        min_dist = max_distance
                        idx_i, idx_j = i, j

            # Merge clusters idx_i and idx_j
            new_cluster = clusters[idx_i] + clusters[idx_j]
            clusters.append(new_cluster)

            # Record the linkage information
            linkage.append(
                [cluster_ids[idx_i], cluster_ids[idx_j], min_dist, len(new_cluster)]
            )

            # Remove old clusters and update cluster IDs
            clusters.pop(idx_j)
            clusters.pop(idx_i)
            clusters = [
                clusters[k] for k in range(len(clusters)) if k != idx_i and k != idx_j
            ]
            clusters.append(new_cluster)

            cluster_ids = np.delete(cluster_ids, [idx_i, idx_j])
            cluster_ids = np.append(cluster_ids, n_samples + step)

            # Update the distance matrix for the new cluster
            new_distances = []
            for k in range(len(clusters) - 1):
                points_k = clusters[k]
                distances = D_p[np.ix_(new_cluster, points_k)]
                max_distance = np.max(distances)
                new_distances.append(max_distance)

            # Remove old distances and append new distances
            D_p = np.delete(D_p, [idx_i, idx_j], axis=0)
            D_p = np.delete(D_p, [idx_i, idx_j], axis=1)

            new_row = np.array(new_distances + [np.inf])
            D_p = np.vstack((D_p, new_row[:-1]))
            new_col = np.append(new_row, [np.inf])
            D_p = np.column_stack((D_p, new_col))

        self.__linkage = np.array(linkage)
        return self.__linkage

    def get_cluster_labels(self, K) -> np.ndarray:
        """
        Get the cluster labels after clustering

        Parameters
        ----------
        K: int - The number of clusters

        Returns
        -------
        labels: np.ndarray - The cluster labels
        """
        n_samples, linkage = self.__X.shape[0], self.__linkage

        labels = np.arange(n_samples)
        for step in range(n_samples - K):
            i, j = int(linkage[step, 0]), int(linkage[step, 1])
            labels[labels == i] = n_samples + step
            labels[labels == j] = n_samples + step
        unique_labels = np.unique(labels)
        label_map = {
            old_label: new_label for new_label, old_label in enumerate(unique_labels)
        }
        labels = np.array([label_map[label] for label in labels])

        return labels

    def silhouette_coefficient(self, labels) -> float:
        """
        Compute the silhouette coefficient

        Parameters
        ----------
        labels: jnp.ndarray - The cluster labels

        Returns
        -------
        silhouette_coefficient: float - The silhouette coefficient
        """
        D = self.__metric(self.__X)
        n_samples = self.__X.shape[0]
        S = jnp.zeros(n_samples)

        for i in range(n_samples):
            same_cluster = labels == labels[i]
            other_clusters = labels != labels[i]

            a_i = jnp.mean(D[i, same_cluster & (jnp.arange(n_samples) != i)])
            b_i = jnp.min(
                jnp.array(
                    [
                        jnp.mean(D[i, labels == label])
                        for label in jnp.unique(labels[other_clusters])
                    ]
                )
            )
            S = S.at[i].set((b_i - a_i) / jnp.maximum(a_i, b_i))

        return jnp.mean(S).item()
