"""
Python script implementing the model to be used in class NCF
to perform Neural Collaborative Filtering training.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# Create a dataset class
class InteractionDataset(Dataset):
    def __init__(self, data_path: str):
        train_df = pd.read_csv(data_path)

        # Map the user_id and book_id to a unique index
        self.user_id_to_idx = {
            user_id: idx
            for idx, user_id
            in enumerate(train_df['user_id'].unique())
        }
        self.num_users = len(self.user_id_to_idx)

        self.item_id_to_idx = {
            book_id: idx
            for idx, book_id
            in enumerate(train_df['book_id'].unique())
        }
        self.num_items = len(self.item_id_to_idx)

        # Retrieve both needed information
        train_df['user_idx'] = train_df['user_id'].map(self.user_id_to_idx)
        train_df['book_idx'] = train_df['book_id'].map(self.item_id_to_idx)

        self.user = train_df["user_idx"].values
        self.item = train_df["book_idx"].values
        self.ratings = (train_df["rating"].values - 1) / 4  # normalize ratings between 0 and 1

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return ((self.user[idx],
                 self.item[idx]),
                np.float32(self.ratings[idx]))


# Define the NCF model
class MLP(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)  # Our choice to add bias
        self.item_bias = nn.Embedding(num_items, 1)  # Our choice to add bias
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1)
        )

        # Initialise accordingly to paper
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        interaction = torch.cat([user_embed, item_embed], dim=-1)
        output = self.fc_layers(interaction)

        # Add biases
        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)
        output += user_bias + item_bias

        return output.squeeze()


class GMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)  # Our choice to add bias
        self.item_bias = nn.Embedding(num_items, 1)  # Our choice to add bias
        self.global_mean = 3
        self.final_layer = nn.Linear(embedding_dim, 1)

        # Initialise accordingly to paper
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.normal_(self.final_layer.weight, std=0.01)

    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)

        # Dot product
        output = self.final_layer(torch.mul(user_embed, item_embed))

        # Add biases
        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)
        output_bias = output + user_bias + item_bias + self.global_mean

        return torch.sigmoid(output_bias).squeeze()


class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim_gmf, latent_dim_mlp):
        super(NeuMF, self).__init__()
        # Embeddings for MLP part
        self.user_embedding_mlp = nn.Embedding(num_users, latent_dim_mlp)
        self.item_embedding_mlp = nn.Embedding(num_items, latent_dim_mlp)

        # Embeddings for GMF part
        self.user_embedding_gmf = nn.Embedding(num_users, latent_dim_gmf)
        self.item_embedding_gmf = nn.Embedding(num_items, latent_dim_gmf)

        # Bias for users and items
        self.user_bias = nn.Embedding(num_users, 1)  # Our choice to add bias
        self.item_bias = nn.Embedding(num_items, 1)  # Our choice to add bias

        # Fully connected layers for MLP
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim_mlp * 2, 64),
            nn.BatchNorm1d(64),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        # Final output layer (after concatenating GMF and MLP outputs)
        self.final_layer = nn.Linear(latent_dim_gmf + 16, 1)

        # Initialise accordingly to paper
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item):
        # MLP embeddings
        user_embed_mlp = self.user_embedding_mlp(user)
        item_embed_mlp = self.item_embedding_mlp(item)

        # GMF embeddings
        user_embed_gmf = self.user_embedding_gmf(user)
        item_embed_gmf = self.item_embedding_gmf(item)

        # GMF interaction (element-wise product)
        gmf_output = torch.mul(user_embed_gmf, item_embed_gmf)

        # MLP interaction
        mlp_input = torch.cat([user_embed_mlp, item_embed_mlp], dim=-1)
        mlp_output = self.fc_layers(mlp_input)

        # Concatenate GMF and MLP outputs
        combined = torch.cat([gmf_output, mlp_output], dim=-1)

        # Final prediction layer
        output = self.final_layer(combined)

        # Add biases
        user_bias = self.user_bias(user)
        item_bias = self.item_bias(item)
        output += user_bias + item_bias

        return output.squeeze()
