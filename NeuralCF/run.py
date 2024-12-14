"""
Python script to run training of NCF
"""
from ncf import *


def training_model(data_path: str,
                   model_to_use: str,
                   val_ratio: float,
                   batch_size: int,
                   embedding_dim_gmf: int,
                   embedding_dim_mlp: int,
                   epochs: int,
                   device: torch.device,
                   lr: float,
                   weight_decay: float,
                   plot_folder: str):
    # Initiate NCF model
    ncf = NCF(data_path=data_path,
              model_to_use=model_to_use,
              val_ratio=val_ratio,
              batch_size=batch_size,
              embedding_dim_gmf=embedding_dim_gmf,
              embedding_dim_mlp=embedding_dim_mlp,
              epochs=epochs,
              device=device,
              lr=lr,
              weight_decay=weight_decay,
              plot_folder=plot_folder)

    # Start the training
    ncf.fit()


if __name__ == '__main__':
    training_model(data_path="../data/train.csv",
                   model_to_use='NeuMF',
                   val_ratio=0.1,
                   batch_size=256,
                   embedding_dim_mlp=3,
                   embedding_dim_gmf=30,
                   epochs=30,
                   device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                   lr=0.001,
                   weight_decay=1e-4,
                   plot_folder='training_evolution')
