try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl

import numpy as np
import torch
from torch import nn

from train_3d.models.simple_cnn import SparseConvRegressor


class Hits3DRegressor(pl.LightningModule):
    def __init__(self, model_config, optim_config, eval_config=None):
        super().__init__()
        self.save_hyperparameters()

        self.model = SparseConvRegressor(
            in_channels=model_config["in_channels"],
            output_dim=model_config.get("output_dim", 1),
            base_channels=model_config.get("base_channels", 16),
            dropout=model_config.get("dropout", 0.1),
            backend=model_config.get("backend", "minkowski"),
        )
        loss_name = str(optim_config.get("loss", "huber")).lower()
        if loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_name == "huber":
            self.loss_fn = nn.HuberLoss(delta=float(optim_config.get("huber_delta", 1.0)))
        elif loss_name == "smoothl1":
            self.loss_fn = nn.SmoothL1Loss(beta=float(optim_config.get("huber_delta", 1.0)))
        else:
            raise ValueError(f"Unsupported loss: {loss_name}")

        self.loss_name = loss_name
        self.lr = float(optim_config["lr"])
        self.weight_decay = float(optim_config.get("weight_decay", 0.0))
        self.target_transform = str(optim_config.get("target_transform", "none"))
        self.eval_config = eval_config or {}
        self.val_pred_energies = []
        self.val_true_energies = []

    def forward(self, coords, features):
        return self.model(coords, features)

    def _prediction_and_metrics(self, batch):
        predictions_transformed = self(
            batch["coords"].int(),
            batch["features"].float(),
        ).squeeze(-1)
        targets = batch["y"].float()
        loss = self.loss_fn(predictions_transformed, targets)

        if self.target_transform == "log1p":
            pred_energy = torch.expm1(predictions_transformed)
        elif self.target_transform == "none":
            pred_energy = predictions_transformed
        else:
            raise ValueError(f"Unsupported target_transform: {self.target_transform}")

        true_energy = batch["energy"].float()
        mae = torch.mean(torch.abs(pred_energy - true_energy))
        rmse = torch.sqrt(torch.mean((pred_energy - true_energy) ** 2))
        return {
            "loss": loss,
            "mae": mae,
            "rmse": rmse,
            "pred_energy": pred_energy,
            "true_energy": true_energy,
        }

    def training_step(self, batch, batch_idx):
        metrics = self._prediction_and_metrics(batch)
        loss = metrics["loss"]
        mae = metrics["mae"]
        rmse = metrics["rmse"]

        batch_size = batch["y"].size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train_mae", mae, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train_rmse", rmse, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        return loss

    def on_validation_epoch_start(self):
        self.val_pred_energies = []
        self.val_true_energies = []

    def validation_step(self, batch, batch_idx):
        metrics = self._prediction_and_metrics(batch)
        loss = metrics["loss"]
        mae = metrics["mae"]
        rmse = metrics["rmse"]
        pred_energy = metrics["pred_energy"]
        true_energy = metrics["true_energy"]

        batch_size = batch["y"].size(0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val_mae", mae, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val_rmse", rmse, on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        self.val_pred_energies.append(pred_energy.detach().cpu())
        self.val_true_energies.append(true_energy.detach().cpu())
        return {"val_loss": loss, "val_mae": mae, "val_rmse": rmse}

    def on_validation_epoch_end(self):
        if not self.val_pred_energies:
            return

        pred = torch.cat(self.val_pred_energies).numpy()
        true = torch.cat(self.val_true_energies).numpy()
        eps = float(self.eval_config.get("relative_eps", 1e-6))
        valid = np.abs(true) > eps

        if not np.any(valid):
            return

        pred = pred[valid]
        true = true[valid]
        rel_delta = (pred - true) / true

        val_bias = float(np.mean(rel_delta))
        val_resolution_std = float(np.std(rel_delta))
        q16, q84 = np.quantile(rel_delta, [0.16, 0.84])
        val_resolution_68 = float(0.5 * (q84 - q16))
        val_rel_mae = float(np.mean(np.abs(rel_delta)))

        self.log("val_bias", val_bias, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_resolution_std", val_resolution_std, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_resolution_68", val_resolution_68, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_rel_mae", val_rel_mae, on_step=False, on_epoch=True, prog_bar=False)

        self._log_wandb_resolution_tables(true, pred, rel_delta)

        self.val_pred_energies = []
        self.val_true_energies = []

    def _log_wandb_resolution_tables(self, true, pred, rel_delta):
        logger = getattr(self, "logger", None)
        if logger is None or not hasattr(logger, "experiment"):
            return
        if getattr(self.trainer, "sanity_checking", False):
            return

        try:
            import wandb
        except ImportError:  # pragma: no cover
            return

        experiment = logger.experiment
        n_bins = int(self.eval_config.get("n_energy_bins", 10))
        min_bin_count = int(self.eval_config.get("min_bin_count", 10))

        if len(true) < max(n_bins, min_bin_count):
            return

        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(true, quantiles)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            return

        rows = []
        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            if high <= low:
                continue
            if high == bin_edges[-1]:
                mask = (true >= low) & (true <= high)
            else:
                mask = (true >= low) & (true < high)
            if int(np.sum(mask)) < min_bin_count:
                continue

            rel_bin = rel_delta[mask]
            q16, q84 = np.quantile(rel_bin, [0.16, 0.84])
            rows.append([
                float(low),
                float(high),
                float(0.5 * (low + high)),
                int(np.sum(mask)),
                float(np.mean(rel_bin)),
                float(np.std(rel_bin)),
                float(0.5 * (q84 - q16)),
            ])

        if not rows:
            return

        resolution_table = wandb.Table(
            columns=[
                "energy_bin_low",
                "energy_bin_high",
                "energy_bin_center",
                "count",
                "bias",
                "resolution_std",
                "resolution_68",
            ],
            data=rows,
        )

        scatter_table = wandb.Table(
            columns=["true_energy", "pred_energy"],
            data=np.stack([true, pred], axis=1).tolist(),
        )

        experiment.log(
            {
                "val_resolution_table": resolution_table,
                "val_resolution_vs_energy_std": wandb.plot.line(
                    resolution_table,
                    "energy_bin_center",
                    "resolution_std",
                    title="Validation Resolution vs True Energy (std)",
                ),
                "val_resolution_vs_energy_68": wandb.plot.line(
                    resolution_table,
                    "energy_bin_center",
                    "resolution_68",
                    title="Validation Resolution vs True Energy (68%)",
                ),
                "val_bias_vs_energy": wandb.plot.line(
                    resolution_table,
                    "energy_bin_center",
                    "bias",
                    title="Validation Bias vs True Energy",
                ),
                "val_pred_vs_true_table": scatter_table,
                "val_pred_vs_true": wandb.plot.scatter(
                    scatter_table,
                    "true_energy",
                    "pred_energy",
                    title="Validation Predicted vs True Energy",
                ),
            },
            step=self.global_step,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler_name = str(self.hparams.optim_config.get("scheduler", "none")).lower()
        if scheduler_name == "none":
            return optimizer
        if scheduler_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=str(self.hparams.optim_config.get("scheduler_mode", "min")),
                factor=float(self.hparams.optim_config.get("scheduler_factor", 0.5)),
                patience=int(self.hparams.optim_config.get("scheduler_patience", 2)),
                min_lr=float(self.hparams.optim_config.get("scheduler_min_lr", 1e-6)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": str(self.hparams.optim_config.get("scheduler_monitor", "val_mae")),
                },
            }

        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
