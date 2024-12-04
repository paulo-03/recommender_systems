"""
Python script to run training of NCF
"""
from ncf import *


def training_model(data_path: str,
                   val_ratio: float,
                   batch_size: int,
                   embedding_dim: int,
                   epochs: int,
                   device: torch.device,
                   lr: float,
                   weight_decay: float,
                   plot_folder: str):

    # Initiate NCF model
    ncf = NCFTrainer(data_path=data_path,
                     val_ratio=val_ratio,
                     batch_size=batch_size,
                     embedding_dim=embedding_dim,
                     epochs=epochs,
                     device=device,
                     lr=lr,
                     weight_decay=weight_decay,
                     plot_folder=plot_folder)

    # Start the training
    ncf.fit()


if __name__ == '__main__':
    training_model(data_path="../data/train.csv",
                   val_ratio=0.8,
                   batch_size=256,
                   embedding_dim=5,
                   epochs=30,
                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                   lr=0.001,
                   weight_decay=1e-4,
                   plot_folder='training_evolution')
