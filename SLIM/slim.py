import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from tqdm import tqdm


class Slim:

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



    def recommender_elasticnet_optimization(self, alpha=0.01, l1_ratio=0.5, max_iter=10):
        """
        Implements SLIM (Sparse Linear Method) with ElasticNet regularization for matrix completion.

        Args:
            alpha (float): Regularization strength for ElasticNet.
            l1_ratio (float): The mix ratio between L1 and L2 regularization (0 <= l1_ratio <= 1).
                              l1_ratio=0 corresponds to Ridge (L2), l1_ratio=1 corresponds to Lasso (L1).
            max_iter (int): Maximum number of iterations for ElasticNet regression.

        Returns:
            numpy.ndarray: Completed user-item matrix.
        """
        num_items = self.user_item_matrix.shape[1]
        W = np.zeros((num_items, num_items))  # Similarity matrix
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, fit_intercept=False,
                                 positive=True,
                                 selection='random', random_state=42, warm_start=True)

        for j in tqdm(range(num_items)):
            # Exclude column j to create X_minus_j
            X_minus_j = self.user_item_matrix.copy()
            X_minus_j[:, j] = 0

            # Target vector (column j of X)
            target = self.user_item_matrix[:, j]

            # Fit ElasticNet for column j
            model.fit(X_minus_j, target)
            W[:, j] = model.coef_

        return W


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
