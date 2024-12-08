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
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from models import InteractionDataset, GMF, MLP, NeuMF


# Define the NCF class
class NCF:
    def __init__(self, data_path: str, val_ratio: float, batch_size: int, epochs: int, device: torch.device,
                 lr: float, weight_decay: float, plot_folder: str, model_to_use: str,
                 embedding_dim_mlp: int = 0, embedding_dim_gmf: int = 0, seed: int = 42):

        # Fix the seed for reproducibility
        torch.manual_seed(seed=seed)

        # Define constants
        self.embedding_dim_mlp = embedding_dim_mlp
        self.embedding_dim_gmf = embedding_dim_gmf
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
        self.user_id_to_idx = dataset.user_id_to_idx
        self.item_id_to_idx = dataset.item_id_to_idx
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.train_size = int((1 - self.val_ratio) * len(dataset))  # Split into training and validation sets
        self.val_size = len(dataset) - self.train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [self.train_size, self.val_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initiate model, criterion, optimizer and schedular
        if model_to_use == 'GMF':
            self.model = GMF(self.num_users, self.num_items, self.embedding_dim_gmf).to(self.device)
        elif model_to_use == 'MLP':
            self.model = MLP(self.num_users, self.num_items, self.embedding_dim_mlp).to(self.device)
        elif model_to_use == 'NeuMF':
            # Ensure embedding dimensions are correctly set by user
            assert self.embedding_dim_gmf != 0 and self.embedding_dim_mlp != 0, \
                "Embedding dimensions (gmf and mlp) must be defined and bigger than 0."
            self.model = NeuMF(self.num_users, self.num_items,
                               self.embedding_dim_gmf, self.embedding_dim_mlp).to(self.device)
        else:
            msg = f"The model chosen ({model_to_use}) does not exist. Please choose between 'GMF', 'MLP' and 'NeuMF'."
            raise ValueError(msg)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.schedular = torch.optim.lr_scheduler.CosineAnnealingLR(
           self.optimizer,
           T_max=len(self.train_loader) * self.epochs)

        #self.schedular = torch.optim.lr_scheduler.ReduceLROnPlateau( TODO
        #    self.optimizer, mode='min', factor=0.9, patience=2
        #)

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
            print(f"Start Training Epoch {epoch + 1}...")
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

        # Plot training curves
        self._plot_training_curves()

    @torch.no_grad()
    def predict(self, test_path: str):
        """Predict rating from a user and item pair"""
        # Load and format the pairs (user, item) to predict the rating
        test_df = pd.read_csv(test_path)
        test_df['user_idx'] = test_df['user_id'].apply(lambda x: self.user_id_to_idx[x])
        test_df['book_idx'] = test_df['book_id'].apply(lambda x: self.item_id_to_idx[x])
        user_item_pairs = test_df[['user_idx', 'book_idx']].values

        # Predict ratings
        self.model.eval()
        submission = []
        for user, item in user_item_pairs:
            with torch.no_grad():
                pred = self.model(
                  torch.tensor([user]).to(self.device), 
                  torch.tensor([item]).to(self.device)
                  )
                rescaled_pred = torch.clamp(pred, min=0, max=1) * 4 + 1  # make sure no prediction are negative
            submission.append(rescaled_pred.item())


        # Create the DataFrame
        submission_df = pd.DataFrame({'id': range(len(submission)), 'rating': submission})

        # Set 'id' as the index
        submission_df.set_index('id', inplace=True)

        # Save the submission in .csv format
        submission_df.to_csv("ncf_ratings.csv", index=True)
        print("Pairs of (user, item) have been successfully predicted.")

    @staticmethod
    def display_epoch_perf(epoch, train_loss, val_loss, lr) -> None:
        """Print evolution of training at each epoch"""
        print(f"\t\t- Train: MSE loss={np.mean(train_loss):.4f}, RMSE={np.sqrt(np.mean(train_loss)):.4f}\n"
              f"\t\t- Val: MSE loss={np.mean(val_loss):.4f}, RMSE={np.sqrt(np.mean(val_loss)):.4f}\n"
              f"\t\t- lr={lr:.10f}\n")

    def _train_epoch(self) -> (float, float):
        """Function to train one epoch and return the average loss of training epoch"""
        # Set the model in training mode
        self.model.train()

        # Initiate the metric
        train_loss = 0

        # Train all batches
        for (user, item), rating in self.train_loader:
            # Move the data to the device
            user, item, rating = user.to(self.device), item.to(self.device), rating.to(self.device)
            # Zero the gradients
            self.optimizer.zero_grad()
            # Compute model output
            output = self.model(user, item)
            # Denormalize ratings and outputs
            rescaled_output = output * 4 + 1
            rescaled_rating = rating * 4 + 1
            # Compute loss
            loss = self.criterion(rescaled_output, rescaled_rating)
            # Backpropagation loss
            loss.backward()
            # Perform an optimizer step
            self.optimizer.step()
            # Step the learning rate scheduler
            self.schedular.step()  # self.schedular.step(avg_val_loss) if plateau used
            # Keep track of average loss
            batch_size = len(user)
            train_loss += loss.item() * batch_size

        return train_loss / self.train_size, self.schedular.get_last_lr()[0]

    @torch.no_grad()
    def _validate(self) -> float:
        """Function to validate the epoch and return the average validation loss"""
        # Set the model in evaluation mode (turn-off the auto gradient computation, ...)
        self.model.eval()

        # Initiate the validation loss
        val_loss = 0

        # Compute metric per batch
        for (user, item), rating in self.val_loader:
            user, item, rating = user.to(self.device), item.to(self.device), rating.to(self.device)
            output = self.model(user, item)
            # Denormalize ratings and outputs
            rescaled_output = torch.clamp(output, min=0, max=1) * 4 + 1  # make sure no prediction are negative
            rescaled_rating = rating * 4 + 1
            batch_size = len(user)
            val_loss += self.criterion(rescaled_output, rescaled_rating).item() * batch_size

        avg_val_loss = val_loss / self.val_size

        return avg_val_loss

    def _plot_training_curves(self):
        """Display a nice plot to show training evolution"""
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))

        plt.title('Training Evolution')

        # plot MSE evolution
        ax[0].plot(range(1, self.epochs + 1), self.train_loss_history, label='Train')
        ax[0].plot(range(1, self.epochs + 1), self.val_loss_history, label='Val')
        ax[0].legend()
        ax[0].set_ylim(0, 1.5)
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('MSE')

        # plot RMSE evolution
        ax[1].plot(range(1, self.epochs + 1), np.sqrt(self.train_loss_history), label='Train')
        ax[1].plot(range(1, self.epochs + 1), np.sqrt(self.val_loss_history), label='Val')
        ax[1].legend()
        ax[1].set_ylim(0, 1.5)
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('RMSE')

        # plot lr evolution
        ax[2].plot(range(1, self.epochs + 1), self.lr_history)
        ax[2].set_xlabel('Epoch')
        ax[2].set_ylabel('lr')

        plt.savefig(self.plot_path)
        plt.show()
