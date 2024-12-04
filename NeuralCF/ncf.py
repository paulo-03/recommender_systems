"""
Python script implementing the class NCF to perform Neural Collaborative Filtering training.
"""
import os
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


# Create a dataset class
class InteractionDataset(Dataset):
    def __init__(self, data_path: str):
        train_df = pd.read_csv(data_path)

        # Map the user_id and book_id to a unique index
        user_id_to_idx = {
            user_id: idx
            for idx, user_id
            in enumerate(train_df['user_id'].unique())
        }
        self.num_users = len(user_id_to_idx)

        item_id_to_idx = {
            book_id: idx
            for idx, book_id
            in enumerate(train_df['book_id'].unique())
        }
        self.num_items = len(item_id_to_idx)

        # Retrieve both needed information
        train_df['user_idx'] = train_df['user_id'].apply(lambda x: user_id_to_idx[x])
        train_df['book_idx'] = train_df['book_id'].apply(lambda x: item_id_to_idx[x])

        self.user_item_pairs = train_df[["user_idx", "book_idx"]].values
        self.ratings = (train_df["rating"].values - 1) / 4  # normalize ratings between 0 and 1

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return ((self.user_item_pairs[idx][:, 0].long(),
                self.user_item_pairs[idx][:, 1].long()),
                self.ratings[idx].float())


# Define the NCF model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.BatchNorm1d(128),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def forward(self, user, item):
        user_embed = self.user_embedding(user)
        item_embed = self.item_embedding(item)
        interaction = torch.cat([user_embed, item_embed], dim=-1)
        return self.fc_layers(interaction).squeeze()


# Define the NCFTrainer class
class NCFTrainer:
    def __init__(self, data_path: str, val_ratio: float, batch_size: int, embedding_dim: int, epochs: int,
                 device: torch.device, lr: float, weight_decay: float, plot_folder: str):
        # Define constants
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_ratio = val_ratio
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay

        # Create directory to store the evolution plot of training
        os.makedirs(plot_folder, exist_ok=True)

        # Format the current date to avoid path problems
        date_str = str(datetime.now()) \
            .replace(':', 'h', 1) \
            .replace(':', 'm', 1) \
            .replace('.', 's', 1)
        self.plot_path = os.path.join(plot_folder, date_str)

        # Load data and create dataloader
        dataset = InteractionDataset(data_path=data_path)
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        # Split into training and validation sets (80-20 split)
        self.train_size = int((1 - self.val_ratio) * len(dataset))
        self.val_size = len(dataset) - self.train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [self.train_size, self.val_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initiate model, criterion and optimizer
        self.model = NCF(self.num_users, self.num_items, self.embedding_dim).to(self.device)
        self.criterion = nn.MSELoss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # TODO: I choose the cosine, but might be good to use other schedular for the lr evolution
        self.schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)


        # Initiate list to store training evolution
        self.train_loss_history = []
        self.train_rmse_history = []
        self.lr_history = []
        self.val_loss_history = []
        self.val_rmse_history = []

    def fit(self) -> None:
        """Compute the training of the model"""
        # Start training
        for epoch in range(self.epochs):
            print(f"Start Training Epoch {epoch}...")
            train_loss, lr = self._train_epoch()
            val_loss = self._validate()

            # Give feedback to user during training
            self.display_epoch_perf(epoch, train_loss, val_loss, lr)

            # Store all metrics in array, to plot them at the end of training
            self.train_loss_history.append(train_loss)
            self.train_rmse_history.append(np.sqrt(train_loss))
            self.lr_history.append(lr)
            self.val_loss_history.append(val_loss)
            self.val_rmse_history.append(np.sqrt(val_loss))

            # TODO (Not sure if useful) Save the model
            # if self.model_saving_path is not None:
            #    self._save_model()

        # Plot training curves
        self._plot_training_curves()

    def predict(self):
        """Predict rating from a user and item pair TODO: denormalize score"""
        ...

    @staticmethod
    def display_epoch_perf(epoch, train_loss, val_loss, lr) -> None:
        """Print evolution of training at each epoch"""
        print(f"Epoch {epoch + 1}:\n"
              f"\t\t- Train: MSE loss={np.mean(train_loss):.4f}, RMSE={np.sqrt(np.mean(train_loss))}\n"
              f"\t\t- Val: MSE loss={np.mean(val_loss):.4f}, RMSE={np.sqrt(np.mean(val_loss))}\n"
              f"\t\t- lr={np.mean(lr):.5f}\n")

    def _train_epoch(self) -> (np.ndarray, np.ndarray):
        """Function to train one epoch"""
        # Set the model in training mode
        self.model.train()

        # Initiate the metric
        train_loss = 0

        # Train all batches
        for (user, item), target in self.train_loader:
            # Move the data to the device
            user, item, target = user.to(self.device), item.to(self.device), target.to(self.device)
            # Zero the gradients
            self.optimizer.zero_grad()
            # Compute model output
            output = self.model(user, item)
            # Compute loss
            loss = self.criterion(output, target)
            # Backpropagation loss
            loss.backward()
            # Perform an optimizer step
            self.optimizer.step()
            # Keep track of average loss
            batch_size = len(user)
            train_loss += self.criterion(output, target).item() * batch_size

        return train_loss / self.train_size, self.schedular.get_last_lr()[0]

    @torch.no_grad()
    def _validate(self) -> np.ndarray:
        """Function to validate the epoch"""
        # Set the model in evaluation mode (turn-off the auto gradient computation, ...)
        self.model.eval()

        # Initiate the validation loss
        val_loss = 0

        # Compute metric per batch
        for (user, item), target in self.val_loader:
            user, item, target = user.to(self.device), item.to(self.device), target.to(self.device)
            output = self.model((user, item))
            batch_size = len(user)
            val_loss += self.criterion(output, target).item() * batch_size

        avg_val_loss = val_loss / self.val_size

        # Perform a learning rate scheduler step (if schedular set) TODO: bug ?
        self.schedular.step(avg_val_loss)

        return avg_val_loss

    def _plot_training_curves(self):
        """Display a nice plot to show training evolution"""
        fig, ax = plt.subplots(1, 3, figsize=(10, 8))

        # plot MSE evolution
        ax[0].plot(range(1, self.epochs+1), self.train_loss_history, label='Train')
        ax[0].plot(range(1, self.epochs+1), self.val_loss_history, label='Val')
        ax[0].legend()
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('MSE')

        # plot RMSE evolution
        ax[1].plot(range(1, self.epochs+1), np.sqrt(self.train_loss_history), label='Train')
        ax[1].plot(range(1, self.epochs+1), np.sqrt(self.val_loss_history), label='Val')
        ax[1].legend()
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('RMSE')

        # plot lr evolution
        ax[2].plot(range(1, self.epochs + 1), self.lr_history)
        ax[2].legend()
        ax[2].set_xlabel('Epoch')
        ax[2].set_ylabel('lr')

        plt.savefig(self.plot_path)
        plt.show()
