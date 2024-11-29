import numpy as np
import pandas as pd


class Ibcf:

    def __init__(self, data_path: str):
        """
        Initializes the DataPreprocessor object.

        Args:
                data_path: Path to the CSV file containing user-item-rating pairs with columns 'user_id', 'book_id', and 'rating'.
        """
        data = pd.read_csv(data_path)
        self._data_to_matrix_df(data)
        self._matrix_df_to_numpy()
        self.predictions = None

    def _data_to_matrix_df(self, data: pd.DataFrame) -> None:
        self.user_item_nan_matrix_df = data.pivot(index='user_id', columns='book_id', values='rating')

    def _matrix_df_to_numpy(self) -> None:
        self.user_item_matrix = self.user_item_nan_matrix_df.fillna(0).to_numpy()
        self.user_to_index = {}
        self.index_to_user = {}
        self.item_to_index = {}
        self.index_to_item = {}
        for idx, user_id in enumerate(self.user_item_nan_matrix_df.index):
            self.user_to_index[user_id] = idx
            self.index_to_user[idx] = user_id

        for idx, book_id in enumerate(self.user_item_nan_matrix_df.columns):
            self.item_to_index[book_id] = idx
            self.index_to_item[idx] = book_id

    def fill_ratings_matrix(self, ratings, similarity):
        # Mask to find which items are rated by each user
        rated_mask = (ratings > 0).astype(np.float64)  # Binary mask (1 for rated items, 0 otherwise)

        # Compute weighted sum for all items (dot product)
        weighted_sum = ratings.dot(similarity)  # Shape: (n_users, n_items)

        # Compute the sum of absolute similarities for each item-user pair
        similarity_sum = rated_mask.dot(np.abs(similarity))  # Shape: (n_users, n_items)

        # Avoid division by zero
        similarity_sum[similarity_sum == 0] = 1  # Prevent division errors

        # Normalize weighted sums by similarity sums
        self.predictions = weighted_sum / similarity_sum  # Element-wise division

        return self.predictions

    def generate_submission_csv(self, test_path: str, output_path: str):
        """
        Generates a CSV file for submission to Kaggle.

        Args:
            test_path: Path to the test.csv file.
            output_path: Path to save the submission CSV file.
        """
        if self.predictions is None:
            raise ValueError("Predictions not computed. Please run the fill_ratings_matrix method first.")

        test = pd.read_csv(test_path)
        test['rating'] = test.apply(
            lambda x: self.predictions[self.user_to_index[x['user_id']], self.item_to_index[x['book_id']]], axis=1)
        test[['rating']].to_csv(output_path, index=True)
