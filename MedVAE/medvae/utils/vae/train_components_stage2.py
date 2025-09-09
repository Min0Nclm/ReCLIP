import os
from time import time
from typing import Any, Dict, List

import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from rich import print
from torch.utils.data import DataLoader
from torchmetrics import Metric
from medvae.utils.extras import (
    get_weight_dtype,
)
from medvae.utils.transforms import to_dict

__all__ = ["training_epoch", "validation_epoch"]

def training_epoch(
    epoch: int,
    global_step: int,
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    default_metrics: List[Metric],
    rec_metrics: List[Metric],
    opt: Optimizer,
    options: Dict[str, Any],
):
    """Train a single epoch of a VAE model."""
    for metric in default_metrics:
        metric.reset()

    for _, metric in rec_metrics:
        metric.reset()

    (
        metric_bc_loss,
        metric_data,
        metric_batch,
    ) = default_metrics

    model.train()
    epoch_start, batch_start = time(), time()
    dtype = get_weight_dtype(accelerator)
    for i, batch in enumerate(dataloader):
        data_time = time() - batch_start

        batch = to_dict(batch)
        images = batch["img"].to(dtype)
        _, _, latent = model(images, decode=False)

        loss = criterion(images, latent=latent)
        loss = loss.sum() / images.shape[0]

        opt.zero_grad()
        accelerator.backward(loss)
        opt.step()

        # Update metrics
        batch_time = time() - batch_start
        metric_data.update(data_time)
        metric_batch.update(batch_time)
        metric_bc_loss.update(loss)

        # Logging values
        print(
            f"\r[Epoch <{epoch:03}/{options['max_epoch']}>: Step <{i:03}/{len(dataloader)}>] - "
            + f"Data(s): {data_time:.3f} ({metric_data.compute():.3f}) - "
            + f"Batch(s): {batch_time:.3f} ({metric_batch.compute():.3f}) - "
            + f"BC Loss: {loss.item():.3f} ({metric_bc_loss.compute():.3f}) - "
            + f"{(((time() - epoch_start) / (i + 1)) * (len(dataloader) - i)) / 60:.2f} m remaining\n"
        )

        if options["is_logging"] and i % options["log_every_n_steps"] == 0:
            log_data = {
                "epoch": epoch,
                "mean_data": metric_data.compute(),
                "mean_batch": metric_batch.compute(),
                "step": i,
                "step_global": global_step,
                "step_data": data_time,
                "step_batch": batch_time,
            }
            log_data["mean_bc_loss"] = metric_bc_loss.compute()

            for name, metric in rec_metrics:
                log_data[name] = metric.compute()

            accelerator.log(log_data)
        global_step += 1

        if global_step % options["ckpt_every_n_steps"] == 0:
            try:
                accelerator.save_state(os.path.join(options["ckpt_dir"], f"step_{global_step}.pt"))
            except Exception as e:
                print(e)

        batch_start = time()

        if options["fast_dev_run"]:
            break

    return global_step


def validation_epoch(
    options: Dict[str, Any],
    epoch: int,
    accelerator: Accelerator,
    dataloader: DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    default_metrics: List[Metric],
    rec_metrics: List[Metric],
    global_step: int = 0,
    postfix: str = "",
):
    """Validate one epoch for the VAE model."""
    for metric in default_metrics:
        metric.reset()

    for _, metric in rec_metrics:
        metric.reset()

    metric_bcloss = default_metrics[0]
    metric_bcloss.reset()

    model.eval()
    criterion.eval()
    epoch_start = time()
    dtype = get_weight_dtype(accelerator)
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = to_dict(batch)

            images = batch["img"].to(dtype)
            _, _, latent = model(images, decode=False)

            loss = criterion(images, latent=latent)
            loss = loss.sum() / images.shape[0]

            metric_bcloss.update(loss)

            # Logging values
            print(
                f"\r Validation{postfix}: "
                + f"\r[Epoch <{epoch:03}/{options['max_epoch']}>: Step <{i:03}/{len(dataloader)}>] - "
                + f"BC Loss: {loss.item():.3f} ({metric_bcloss.compute():.3f}) - "
                + f"{(((time() - epoch_start) / (i + 1)) * (len(dataloader) - i)) / 60:.2f} m remaining\n"
            )

            if options["fast_dev_run"]:
                break

        if options["is_logging"]:
            log_data = {
                f"valid{postfix}/epoch": epoch,
                f"valid{postfix}/mean_bcloss": metric_bcloss.compute(),
            }
            for name, metric in rec_metrics:
                log_data[f"valid{postfix}/{name}"] = metric.compute()

            accelerator.log(log_data)

    return metric_bcloss.compute()