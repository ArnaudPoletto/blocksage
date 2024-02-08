import os
import time
import wandb
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from typing import Tuple, Dict
from torch.optim import Optimizer
from abc import ABC, abstractmethod
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.log import log, warn
from src.config import (
    DATA_MODELS_PATH,
    WANDB_PROJECT_NAME,
    WANDB_DATASET_NAME,
    TRAINER_RESULTS_FILE_PATH,
    TRAINER_RESULTS_FOLDER_PATH,
)


class Trainer(ABC):
    """
    Trainer class used to train a model.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        accumulation_steps: int,
        evaluation_steps: int,
        name: str,
        use_scheduler: bool,
        use_scaler: bool,
        print_statistics: bool,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model (nn.Module): Model to train.
            criterion (nn.Module): Loss function to use.
            accumulation_steps (int): Accumulation steps for gradient accumulation.
            evaluation_steps (int): Evaluation steps for evaluation.
            name (str): Name of the model.
            use_scheduler (bool): Whether to use scheduler.
            use_scaler (bool): Whether to use scaler.
            print_statistics (bool): Whether to print statistics during training.

        Raises:
            ValueError: If accumulation_steps is not a positive integer.
            ValueError: If evaluation_steps is not a positive integer.
        """
        super().__init__()

        if accumulation_steps <= 0:
            raise ValueError("❌ Accumulation steps must be a positive integer.")

        if evaluation_steps <= 0:
            raise ValueError("❌ Evaluation steps must be a positive integer.")

        if use_scheduler:
            warn("Using scheduler is recommended for better convergence.")

        if not use_scaler:
            warn("Using scaler is recommended for better performance.")

        self.model = model
        self.criterion = criterion
        self.accumulation_steps = accumulation_steps
        self.evaluation_steps = evaluation_steps
        self.name = name
        self.use_scheduler = use_scheduler
        self.use_scaler = use_scaler
        self.print_statistics = print_statistics

        self.best_eval_val_loss = np.inf
        self.eval_train_loss = 0
        self.eval_val_loss = 0

    def _get_name(
        self, optimizer: Optimizer, num_epochs: int, learning_rate: int
    ) -> str:
        """
        Get the name of the model.

        Args:
            optimizer (Optimizer): Optimizer used.
            num_epochs (int): Number of epochs.
            learning_rate (int): Learning rate.

        Returns:
            str: Name of the model
        """
        return self.name

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        num_epochs: int,
        learning_rate: int,
        save_model: bool = True,
        sweeping: bool = False,
    ) -> Dict:
        """
        Train the model.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            optimizer (Optimizer): Optimizer to use during training.
            num_epochs (int): Number of epochs to train.
            learning_rate (int): Learning rate to use.
            save_model (bool, optional): Whether to save the best model. Defaults to True.
            sweeping (bool, optional): Whether the training is part of a sweeping. Defaults to False.

        Raises:
            ValueError: If num_epochs is not a positive integer.

        Returns:
            Dict: Statistics of the training.
        """
        if num_epochs <= 0:
            raise ValueError("❌ num_epochs must be a positive integer.")

        if not save_model:
            warn("The trained model will not be saved.")

        if sweeping:
            log("Sweeping mode enabled. Not using wandb default logging.")

        name = self._get_name(optimizer, num_epochs, learning_rate)

        # Get a timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{timestamp}_{name}"

        # Create a folder in data models folder with the name of the model
        model_dir = f"{DATA_MODELS_PATH}{self.model.__class__.__name__.lower()}/"
        os.makedirs(model_dir, exist_ok=True)

        save_path = f"{model_dir}{name}.pth" if save_model else None

        # Setup WandB and watch
        if not sweeping:
            wandb.init(
                project=WANDB_PROJECT_NAME,
                config={
                    "architecture": self.__class__.__name__,
                    "name": name,
                    "dataset": WANDB_DATASET_NAME,
                    "epochs": num_epochs,
                    "learning_rate": learning_rate,
                },
            )
        wandb.watch(self.model, log_freq=4, log="all")

        log(f"🚀 Training {self.__class__.__name__} method for {num_epochs} epochs...")

        # Recording statistics for each epoch
        statistics = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "train_loader_length": len(train_loader),
            "val_loader_length": len(val_loader),
            "num_epochs": num_epochs,
            "evaluation_steps": self.evaluation_steps,
        }

        # Scheduler
        scheduler = (
            ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
            if self.use_scheduler
            else None
        )

        # Scaler
        scaler = GradScaler(enabled=self.use_scaler)

        # Training loop
        with tqdm(range(num_epochs), desc="Epochs", unit="epoch") as bar:
            for _ in bar:
                self._train_one_epoch(
                    train_loader,
                    val_loader,
                    optimizer,
                    statistics,
                    scaler,
                    scheduler,
                    save_path,
                    bar,
                )

        wandb.unwatch(self.model)
        if not sweeping:
            wandb.finish()

        return statistics

    def test(
        self,
        model_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> Dict:
        """
        Test the model.

        Args:
            model_path (str): Path to the model.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            test_loader (DataLoader): Test data loader.

        Returns:
            Dict: Statistics of the testing.
        """
        # Load model
        self.model.load_state_dict(torch.load(model_path))
        log("✅ Using model at ", model_path)

        # Create pandas dataframe to store results
        results = pd.DataFrame()
        results.loc[0, "name"] = self.model.__class__.__name__

        metrics = ["loss", "accuracy"]

        print("📊 Results:")
        for name, loader in [
            ("train", train_loader),
            ("val", val_loader),
            ("test", test_loader),
        ]:
            stats = self._evaluate(loader, show_time=True)
            for metric in metrics:
                if metric not in stats:
                    continue

                col_name = f"{name}_{metric}"

                # Print results
                metric_display = metric.capitalize()
                if metric == "loss":
                    print(f"\t{name} {metric_display}: {stats[metric]:.4f}")

                    # Save the results to the dataframe
                    results.loc[0, col_name] = stats[metric]
                else:
                    print(f"\t{name} {metric_display}: {stats[metric]:.4f}")

                    # Save the results to the dataframe
                    number = np.round(stats[metric] * 100, 2)
                    results.loc[0, col_name] = rf"${number}$"  # LaTeX format

        # Create the folder if it does not exist
        os.makedirs(TRAINER_RESULTS_FOLDER_PATH, exist_ok=True)

        # Save the results to a csv file
        if not os.path.isfile(TRAINER_RESULTS_FILE_PATH):
            results.to_csv(TRAINER_RESULTS_FILE_PATH, index=False, sep="&")
        else:
            results.to_csv(
                TRAINER_RESULTS_FILE_PATH, mode="a", header=False, index=False, sep="&"
            )

        return stats

    @abstractmethod
    def _forward_pass(
        self, batch: tuple
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of a batch. This should be implemented by the child class.

        Args:
            batch (tuple): The batch of data.

        Raises:
            NotImplementedError: If not implemented by child class.

        Returns:
            train_loss (torch.Tensor): Loss of the batch.
            pred (torch.Tensor): Prediction of the batch.
            target (torch.Tensor): Target of the batch.
        """
        raise NotImplementedError

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        statistics: Dict,
        scaler: GradScaler,
        scheduler: ReduceLROnPlateau,
        save_path: str,
        bar: tqdm,
    ) -> None:
        """
        Train the model for one epoch.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            optimizer (Optimizer): Optimizer to use during training.
            statistics (Dict): Statistics of the training.
            scaler (GradScaler): Scaler to use.
            scheduler (ReduceLROnPlateau): Scheduler to use.
            save_path (str): Path to save the best model.
            bar (tqdm): Progress bar to use.
        """
        self.model.train()

        total_train_loss = 0
        n_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Zero the gradients
            optimizer.zero_grad()
            train_loss, _, _ = self._forward_pass(batch)
            total_train_loss += train_loss.item()
            n_train_loss += 1

            scaler.scale(train_loss).backward()

            # Optimize every accumulation steps
            if ((batch_idx + 1) % self.accumulation_steps == 0) or (
                batch_idx + 1 == len(train_loader)
            ):
                scaler.step(optimizer)
                scaler.update()

            if bar is not None:
                bar.set_postfix(
                    {
                        "batch": f"{batch_idx + 1}/{len(train_loader)}",
                        "train_loss": f"{self.eval_train_loss:.4f}",
                        "val_loss": f"{self.eval_val_loss:.4f}",
                    }
                )

            if (batch_idx + 1) % self.evaluation_steps == 0 or (
                batch_idx + 1 == len(train_loader)
            ):
                # Get and update training loss
                self.eval_train_loss = total_train_loss / n_train_loss
                statistics["train_loss"].append(train_loss)
                total_train_loss = 0
                n_train_loss = 0

                # Get validation loss ans statistics
                stats = self._evaluate(val_loader, bar, show_time=False)
                self.eval_val_loss = stats["loss"]
                statistics["val_loss"].append(self.eval_val_loss)

                # Update best model
                if (
                    self.eval_val_loss < self.best_eval_val_loss
                    and save_path is not None
                ):
                    log(f"🎉 Saving model with new best loss: {self.eval_val_loss:.4f}")
                    torch.save(self.model.state_dict(), save_path)
                    self.best_eval_val_loss = self.eval_val_loss

                # Log statistics
                if "accuracy" in stats:
                    statistics["val_acc"].append(stats["accuracy"])

                if bar is None and self.print_statistics:
                    print(
                        f"➡️ Training loss: {self.eval_train_loss:.4f}, Validation loss: {self.eval_val_loss:.4f}"
                    )
                else:
                    bar.set_postfix(
                        {
                            "train_batch": f"{batch_idx + 1}/{len(train_loader)}",
                            "train_loss": f"{self.eval_train_loss:.4f}",
                            "val_loss": f"{self.eval_val_loss:.4f}",
                        }
                    )
                wandb_log = {
                    "train_loss": self.eval_train_loss,
                    "val_loss": self.eval_val_loss,
                }
                if "accuracy" in stats:
                    wandb_log["val_acc"] = stats["accuracy"]
                wandb.log(wandb_log)

                # Step the scheduler
                if scheduler is not None:
                    scheduler.step(self.eval_val_loss)

    def _evaluate(
        self, loader: DataLoader, bar: tqdm, show_time: bool
    ) -> Dict[str, float]:
        """
        Evaluate the model on the given loader.

        Args:
            loader (DataLoader): Loader to evaluate on.
            bar (tqdm): Progress bar to use.
            show_time (bool): Whether to show the time taken for evaluation.

        Returns:
            Dict[str, float]: Evaluation statistics, which contain:
                - the validation loss
                - the accuracy
        """
        self.model.eval()

        total_val_loss = 0

        correct = 0
        total = 0
        with torch.no_grad():
            start_time = time.time()
            for batch_idx, batch in enumerate(loader):
                val_loss, pred, target = self._forward_pass(batch)
                total_val_loss += val_loss.item()

                if pred is not None and target is not None:
                    correct += (pred == target).sum().item()
                    total += torch.numel(pred)

                if bar is not None:
                    bar.set_postfix(
                        {
                            "eval_batch": f"{batch_idx + 1}/{len(loader)}",
                        }
                    )

            if show_time:
                print(f"⏲️ Time taken for evaluation: {time.time() - start_time:.4f}s")

        # Get statistics
        total_val_loss /= len(loader)
        stats = {
            "loss": total_val_loss,
        }

        # Other statistics if available
        if pred is not None and target is not None:
            acc = correct / total
            stats["accuracy"] = acc

        return stats
