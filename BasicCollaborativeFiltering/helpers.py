from typing import Literal

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(ratings, similarity_metric: Literal['cosine', 'pearson'],
                       kind: Literal['user', 'item']) -> np.ndarray:
    """
    Computes the similarity matrix.

    Args:
        ratings: User-item matrix of ratings.
        similarity_metric: Similarity metric to use for computing the similarity matrix. Choose from 'cosine' or 'pearson'.
        kind: Axis to compute similarity on. Choose 'user' for user similarity or 'item' for item similarity.

    Returns:
        similarity: Computed similarity matrix.
    """
    if kind == 'item':
        matrix = ratings.T
    else:
        matrix = ratings

    if similarity_metric == 'cosine':
        similarity = cosine_similarity(matrix)
    elif similarity_metric == 'pearson':
        similarity = np.corrcoef(matrix, rowvar=True)
    else:
        raise ValueError("Invalid similarity metric. Please choose 'cosine' or 'pearson'.")

    return similarity


def fast_cosine_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    sim = 0
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return sim / norms / norms.T
