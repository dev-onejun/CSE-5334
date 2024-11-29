import random

import jax
from jax import numpy as jnp
from jax._src.random import KeyArray

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> KeyArray:
    """
    Set the seed for reproducibility.

    Parameters
    ----------
    seed: int - The seed value

    Returns
    -------
    key: jax.random.key - The key for reproducibility for JAX
    """

    random.seed(seed)
    np.random.seed(seed)

    key = jax.random.key(seed)

    return key


def load_data(path: str = "wine.csv") -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Load the data from the given path.

    Parameters
    ----------
    path: str - The path to the data file

    Returns
    -------
    X: jnp.ndarray - The features of the data
    y: jnp.ndarray - The target of the data
    """
    data = pd.read_csv(path)

    X = data.drop("Wine", axis=1)
    y = data["Wine"]

    X = jnp.array(X)
    y = jnp.array(y)

    return X, y


def make_plot(history: list):
    k = [k for k, _ in history]
    sse_history = [sse for _, sse in history]

    # sse_history_plot = np.array(sse_history)

    plt.plot(range(len(sse_history)), sse_history, marker="o")
    plt.xticks(
        range(len(sse_history)),
        k,
    )

    plt.title("SSE with K values")
    plt.xlabel("K")
    plt.ylabel("SSE")

    plt.show()
