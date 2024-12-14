"""
Python script giving the implementation of EASE for a project in Distributed Information System at EPFL.

Author: Paulo Ribeiro
"""
import time
import os
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from datetime import timedelta
from tqdm.notebook import tqdm


class EASE:
    def __init__(self, X: csr_matrix, maps: dict):
        self.X = self._process_data(X)
        self.n_user = self.X.shape[0]
        self.n_item = self.X.shape[1]
        self.user_to_index = maps['user_to_index']
        self.book_to_index = maps['item_to_index']
        self.B = None
        self.score = None
        self.rating = None
        self.pred = None

    @staticmethod
    def _process_data(X: csr_matrix) -> csr_matrix:
        """Process the data to make it useful in the EASE pipeline by putting a one for the score higher or equal than
        3 and 0 otherwise"""
        X.data = (X.data >= 3.0).astype(float)
        return X

    def fit(self, lambda_: float = 1):
        """Compute the closed form of the weight matrix"""

        start = time.time()
        # Gram-Matrix computation
        print("Computing the Gram-Matrix... [1/4]")
        G = self.X.T.dot(self.X).toarray()

        print("Adding lambda to the diagonal of Gram-Matrix... [2/4]")
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_

        print("Computing the inverse of Gram-Matrix... [3/4]")
        P = np.linalg.inv(G)

        print("Compute the weight matrix... [4/4]")
        self.B = P / (-np.diag(P))
        self.B[diagIndices] = 0
        end = time.time()

        print(f"\nTraining completed in {str(timedelta(seconds=end - start))}")

    def predict(self):
        """Predict the scores and ratings for all user-item pair given in"""
        start = time.time()
        print("Starting to predict score of all (user,book) pairs...")
        self.score = self.X.dot(self.B)
        print("Map score into a 1 to 4 score range...")
        self.rating = self._map_score_to_rating()
        end = time.time()
        print(f"\nPrediction completed in {str(timedelta(seconds=end - start))}")

    def _map_score_to_rating(self) -> np.ndarray:
        """"""
        min_score, max_score = self.score.min(), self.score.max()
        return 1 + 3 * (self.score - min_score) / (max_score - min_score)

    #######################################################################################
    ####### Just a trace of my attempts to enhance the mapping from score to rating #######
    #######################################################################################

    def __map_score_to_rating(self) -> np.ndarray:
        # Create the map to specify the rank score to rating
        score_rank_to_rating = self._create_map_score_to_rating()

        # Create an empty matrix to hold the ratings
        ratings_matrix = np.empty_like(self.score, dtype=np.float64)

        for user in tqdm(range(self.n_user)):
            # Create the map to specify the rank score to rating
            score_rank_to_rating = self._create_map_score_to_rating()
            # Get the user scores
            row = self.score[user, :]
            # Get the sorted indices of the user scores in descending order
            sorted_indices = np.argsort(row)[::-1]
            # Map the scores to ratings for the user
            user_ratings = np.empty_like(row, dtype=np.float64)
            for rank, item_id in enumerate(sorted_indices):
                user_ratings[item_id] = score_rank_to_rating[rank]
            # Assign the mapped ratings back to the ratings matrix
            ratings_matrix[user, :] = user_ratings

        return ratings_matrix

    def ___map_score_to_rating(self, k=10, c=0.0, rating_min=1.0, rating_max=5.0):
        sigmoid = 1 / (1 + np.exp(-k * (self.score - c)))
        return rating_min + (rating_max - rating_min) * sigmoid

    def _create_map_score_to_rating(self):
        return {position_score: rating
                for position_score, rating in enumerate(np.linspace(start=3, stop=1.5, num=self.n_item))}

    #######################################################################################
    #######################################################################################

    def retrieve_pred(self, df: pd.DataFrame):
        """Retrieve the rating that we want to predict on Kaggle"""
        pairs = [[self.user_to_index[row['user_id']], self.book_to_index[row['book_id']]] for _, row in df.iterrows()]
        self.pred = pd.Series([self.rating[user, book] for user, book in pairs], name='rating')
        self.pred.index.name = 'id'

    def save_pred(self, path: str = ""):
        """Save the prediction into a .csv file"""
        self.pred.to_csv(os.path.join(path, "ease_ratings.csv"), index=True)
