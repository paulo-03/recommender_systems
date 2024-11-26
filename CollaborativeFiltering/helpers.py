"""
TODO
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Literal

class DataPreprocess:
    def __init__(self, data: pd.DataFrame):
        self.data = data.fillna(0).to_numpy()

    def compute_similarity(self, similarity_metric: Literal['cosine', 'pearson'],
                           axis: Literal['user', 'item']) -> np.ndarray:
        """
        Computes the similarity matrix.

        Args:
            similarity_metric: Similarity metric to use for computing the similarity matrix. Choose from 'cosine' or 'pearson'.
            axis: Axis to compute similarity on. Choose 'user' for user similarity or 'item' for item similarity.

        Returns:
            similarity: Computed similarity matrix.
        """
        if axis == 'item':
            matrix = self.data.T
        else:
            matrix = self.data

        if similarity_metric == 'cosine':
            similarity = cosine_similarity(matrix)
        elif similarity_metric == 'pearson':
            similarity = np.corrcoef(matrix, rowvar=True)
        else:
            raise ValueError("Invalid similarity metric. Please choose 'cosine' or 'pearson'.")

        return similarity
