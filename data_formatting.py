"""
This script implement a class to allow each person to easily format the data we receive.
"""

import numpy as np
import pandas as pd


class DataFormatting:
    def __init__(self, data_path: str):
        """
        Initializes the DataFormatting object.

        Args:
            data: DataFrame containing user-item-rating pairs with columns 'user_id', 'book_id', and 'rating'.
        """
        self.data = pd.read_csv(data_path)

    def create_matrix_df(self) -> pd.DataFrame:
        """TODO"""
        return self.data.pivot(index='user_id', columns='book_id', values='rating')

    def create_matrix_np(self) -> (np.ndarray, dict):
        """TODO"""
        df = self.data.pivot(index='user_id', columns='book_id', values='rating')
        matrix_np = df.to_numpy()
        maps = {
            'user_to_index': {},
            'index_to_user': {},
            'item_to_index': {},
            'index_to_item': {}
        }

        for idx, user_id in enumerate(df.index):
            maps['user_to_index'][user_id] = idx
            maps['index_to_user'][idx] = user_id

        for idx, book_id in enumerate(df.columns):
            maps['item_to_index'][book_id] = idx
            maps['index_to_item'][idx] = book_id


        return matrix_np, maps
