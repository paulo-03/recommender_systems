"""
This script is used to fine tune our NCF models and retrieve the best parameters to give to
our methods to successfully train in the best condition these models.
"""

import optuna
from ncf import *


def objective(trial):
    # Define the hyperparameter search space
    embedding_dim_mlp = trial.suggest_int('embedding_dim_mlp', 8, 128)
    embedding_dim_gmf = trial.suggest_int('embedding_dim_gmf', 8, 128)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_int('batch_size', 256, 4096)

    # Initialize the NCF model with the trial parameters
    model = NCF(data_path='../data/train.csv',
                model_to_use='MLP',
                val_ratio=0.05,
                batch_size=batch_size,
                embedding_dim_gmf=embedding_dim_gmf,
                embedding_dim_mlp=embedding_dim_mlp,
                epochs=30,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                lr=lr,
                weight_decay=weight_decay,
                plot_folder='training_evolution')

    # Train the model
    model.fit()

    print(f"Best validation RMSE: {model.best_val_rmse:.4f}\n"
          f"Epoch at which occurs: {model.epoch_for_best_val_rmse}")

    return model.best_val_rmse


# Set up the Optuna study
def tune_hyperparameters():
    study = optuna.create_study(direction='minimize')  # Adjust direction if you want to minimize the metric
    study.optimize(objective, n_trials=50)  # Adjust n_trials as needed

    print("Best hyperparameters: ", study.best_params)
    print("Best score: ", study.best_value)


if __name__ == "__main__":
    tune_hyperparameters()
