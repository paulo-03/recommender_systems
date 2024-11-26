import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Literal

class DataPreprocessor:

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the DataPreprocessor object.

        Args:
            data: DataFrame containing user-item-rating pairs with columns 'user_id', 'book_id', and 'rating'.
        """
        self._data_to_matrix_df(data)
        self._matrix_df_to_numpy()

    def _data_to_matrix_df(self, data: pd.DataFrame) -> None:
        self.user_item_nan_matrix_df = data.pivot(index='user_id', columns='book_id', values='rating')

    def _matrix_df_to_numpy(self) -> None:
        self.user_item_matrix = self.user_item_nan_matrix_df.fillna(0).to_numpy()
        self.user_to_index = {}
        self.index_to_user = {}
        self.item_to_index = {}
        self.index_to_item = {}
        for idx, (user_id, book_id) in enumerate(zip(self.user_item_nan_matrix_df.index, self.user_item_nan_matrix_df.columns)):
            self.user_to_index[user_id] = idx
            self.index_to_user[idx] = user_id
            self.item_to_index[book_id] = idx
            self.index_to_item[idx] = book_id

    def compute_similarity(self, similarity_metric: Literal['cosine', 'pearson'], axis: Literal['user', 'item']) -> np.ndarray:
        """
        Computes the similarity matrix.

        Args:
            similarity_metric: Similarity metric to use for computing the similarity matrix. Choose from 'cosine' or 'pearson'.
            axis: Axis to compute similarity on. Choose 'user' for user similarity or 'item' for item similarity.

        Returns:
            similarity: Computed similarity matrix.
        """
        if axis == 'item':
            matrix = self.user_item_matrix.T
        else:
            matrix = self.user_item_matrix

        if similarity_metric == 'cosine':
            similarity = cosine_similarity(matrix)
        elif similarity_metric == 'pearson':
            similarity = np.corrcoef(matrix, rowvar=True)
        else:
            raise ValueError("Invalid similarity metric. Please choose 'cosine' or 'pearson'.")

        return similarity