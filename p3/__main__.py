"""
Programming Assignment 3
CSE-5334 Data Mining, Fall 2024
Professor: Dr. Marnim Galib

Wonjun Park
UTA ID: 1002237177
Computer Science and Engineering, University of Texs at Arlington, Arlington, TX, USA
wxp7177@mavs.uta.edu
"""

from utils import load_data, set_seed, make_plot
from clusters import KMeans, AgglomerativeClustering


def train_kmeans(X, K=3):
    kmeans = KMeans(key=key, n_clusters=K, max_iter=1000)

    iter_num = kmeans.fit(X)

    SSE = kmeans.evaluate()

    print(f"For k = {K} After {iter_num} iterations: SSE = {SSE}")

    return SSE


def main():
    X, _ = load_data()

    SSE_history = []
    for K in range(2, 9):
        SSE = train_kmeans(X, K)
        SSE_history.append((K, SSE))

    make_plot(SSE_history)


if __name__ == "__main__":
    key = set_seed(42)

    main()
