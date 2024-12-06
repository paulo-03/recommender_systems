import numpy as np
import pandas as pd


class Ubcf:

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
        # Compute the average rating for each user
        user_average_ratings = np.true_divide(
            ratings.sum(axis=1),
            (ratings > 0).sum(axis=1),
            where=(ratings > 0).sum(axis=1) > 0  # Avoid division by zero
        )

        # Center the ratings matrix by subtracting user's average (only for rated items)
        ratings_centered = ratings - user_average_ratings[:, None]
        ratings_centered[ratings == 0] = 0  # Leave unrated items as zero

        # Weighted sum of ratings by user similarity
        weighted_sum = similarity.dot(ratings_centered)  # Shape: (n_users, n_items)

        # Sum of absolute similarities for normalization
        similarity_sum = np.abs(similarity).dot((ratings > 0).astype(np.float64))  # Shape: (n_users, n_items)
        similarity_sum[similarity_sum == 0] = 1  # Prevent division by zero

        # Compute the normalized predictions
        self.predictions = user_average_ratings[:, None] + (weighted_sum / similarity_sum)

        # Clip predictions to valid range
        self.predictions = np.clip(self.predictions, 1, 5)

        return self.predictions

    def generate_submission_csv(self, test_path: str, output_path: str):
        """
        Generates a CSV file for submission to Kaggle.

        Args:
            test_path: Path to the test.csv file
        """
        if self.predictions is None:
            raise ValueError("Predictions not computed. Please run the fill_ratings_matrix method first.")

        test = pd.read_csv(test_path)
        test['rating'] = test.apply(
            lambda x: self.predictions[self.user_to_index[x['user_id']], self.item_to_index[x['book_id']]], axis=1)
        test[['rating']].to_csv(output_path, index=True)
