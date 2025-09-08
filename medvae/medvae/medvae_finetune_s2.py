"""
Please note this script is for 2D stage 2 finetuning. If you have a 3D model, please use the medvae_finetune.py script.
"""

import torch
import pyrootutils
from medvae.utils.extras import cite_function
from medvae.utils.factory import create_model
from medvae.utils.extras import sanitize_dataloader_kwargs, set_seed
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import os
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import GradientAccumulationPlugin
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from medvae.utils.vae.train_components_stage2 import training_epoch, validation_epoch

"""
Process configuration from Hydra instead of command line arguments.

Args:
    cfg: Hydra configuration object
    
Returns:
    The processed configuration: hydra config object
"""


def parse_arguments(cfg: DictConfig):
    # Print a message to cite the med-vae paper
    cite_function()

    # Add visual emphasis to important warning message
    print("\n" + "=" * 80)
    print("⚠️  WARNING: This script is for 2D stage 2 finetuning ONLY ⚠️")
    print("If you have a 3D model, please use the medvae_finetune.py script instead.")
    print("=" * 80 + "\n")

    # Validate model_name
    valid_model_names = [
        "medvae_4_1_2d",
        "medvae_4_3_2d",
        "medvae_4_4_2d",
        "medvae_8_1_2d",
        "medvae_8_4_2d",
    ]
    assert cfg.model_name in valid_model_names, (
        f"model_name must be one of {valid_model_names}. Got: {cfg.model_name}."
    )

    assert cfg.stage2 is True, (
        f"stage2 must be True for stage 2 finetuning. This is used for 2D stage 2 finetuning. Got: {cfg.stage2}."
    )

    cfg.stage2_ckpt = os.path.abspath(cfg.stage2_ckpt)
    if not os.path.exists(cfg.stage2_ckpt):
        raise FileNotFoundError(f"stage2_ckpt {cfg.stage2_ckpt} does not exist.")

    return cfg


# Set the project root
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
config_dir = os.path.join(root, "configs")

# Register configuration resolvers
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(
    version_base="1.2", config_path=config_dir, config_name="finetuned_vae.yaml"
)
def main(cfg: DictConfig):
    cfg = parse_arguments(cfg)

    # Instantiating config
    print(f"=> Starting [experiment={cfg.get('task_name', 'default')}]")
    cfg = instantiate(cfg)

    # Seeding
    if cfg.get("seed", None) is not None:
        print(f"=> Setting seed [seed={cfg.seed}]")
        set_seed(cfg.seed)

    torch.backends.cuda.matmul.allow_tf32 = True

    # Setup accelerator
    logger_kwargs = cfg.get("logger", None)
    is_logging = bool(logger_kwargs)
    print(f"=> Instantiate accelerator [logging={is_logging}]")

    gradient_accumulation_steps = cfg.get("gradient_accumulation_steps", 1)
    accelerator = Accelerator(
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=gradient_accumulation_steps,
            adjust_scheduler=False,
        ),
        mixed_precision=cfg.get("mixed_precision", None),
        log_with="wandb" if is_logging else None,
        split_batches=True,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=True,
            )
        ],
    )
    accelerator.init_trackers(
        "medvae", config=cfg, init_kwargs={"wandb": logger_kwargs}
    )

    # Determine the mode
    print(f"=> Mixed precision: {accelerator.mixed_precision}")

    inference_mode = cfg.get("inference", False)
    print(f"=> Running in inference mode: {inference_mode}")

    print(f"=> Instantiating train dataloader [device={accelerator.device}]")
    train_dataloader = DataLoader(
        **sanitize_dataloader_kwargs(cfg["dataloader"]["train"])
    )

    print(f"=> Instantiating valid dataloader [device={accelerator.device}]")
    valid_dataloader = DataLoader(
        **sanitize_dataloader_kwargs(cfg["dataloader"]["valid"])
    )

    # Create loss function
    criterion = cfg.criterion

    # Create model and use prior weight for stage 1 weight for stage 2 finetuning
    model = create_model(
        cfg.model_name,
        existing_weight=cfg.stage2_ckpt,
        training=False,
        state_dict=False,
    )

    # Freeze the encoder, decoder, quant_conv, and post_quant_conv layers, so that only the projection head is trainable
    print(
        "Trainable Params before freeze:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    model.encoder.requires_grad_(False)
    model.decoder.requires_grad_(False)
    model.quant_conv.requires_grad_(False)
    model.post_quant_conv.requires_grad_(False)
    print(
        "Trainable Params after freeze:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Create two optimizers: one for the autoencoder and one for the discriminator
    print(f"=> Instantiating the optimizer [device={accelerator.device}]")

    batch_size, lr = cfg.batch_size, cfg.base_learning_rate
    lr = gradient_accumulation_steps * batch_size * lr

    ae_params = list(model.channel_ds.parameters()) + list(
        model.channel_proj.parameters()
    )
    opt_ae = torch.optim.Adam(ae_params, lr=lr, betas=(0.5, 0.9))

    num_metrics = 3

    # Prepare components for multi-gpu/mixed precision training
    (train_dataloader, valid_dataloader, model, criterion) = accelerator.prepare(
        train_dataloader,
        valid_dataloader,
        model,
        criterion,
    )

    # Create metrics: aeloss, discloss, recloss, data(time), batch(time)
    default_metrics = accelerator.prepare(*[MeanMetric() for _ in range(num_metrics)])
    if len(cfg["metrics"]) > 0:
        names, metrics = list(zip(*cfg["metrics"]))
        metrics = list(zip(names, accelerator.prepare(*metrics)))
    else:
        metrics = []

    options = {
        "max_epoch": cfg["max_epoch"],
        "is_logging": is_logging,
        "log_every_n_steps": cfg["log_every_n_steps"],
        "ckpt_every_n_steps": cfg["ckpt_every_n_steps"],
        "ckpt_dir": cfg["ckpt_dir"],
        "fast_dev_run": cfg["fast_dev_run"],
    }

    print(f"=> Starting model training [epochs={cfg['max_epoch']}]")
    min_loss = None
    global_step = cfg.get("global_step", 0)
    for epoch in range(cfg["max_epoch"]):
        global_step = training_epoch(
            options=options,
            epoch=epoch,
            global_step=global_step,
            accelerator=accelerator,
            dataloader=train_dataloader,
            model=model,
            criterion=criterion,
            default_metrics=default_metrics,
            rec_metrics=metrics,
            opt=opt_ae,
        )

        accelerator.wait_for_everyone()

        loss = validation_epoch(
            options=options,
            epoch=epoch,
            accelerator=accelerator,
            dataloader=valid_dataloader,
            model=model,
            criterion=criterion,
            default_metrics=default_metrics,
            rec_metrics=metrics,
            global_step=global_step,
        )

        # save the best model
        if min_loss is None or loss < min_loss:
            try:
                accelerator.save_state(
                    os.path.join(cfg.ckpt_dir, "best.pt"), safe_serialization=False
                )
            except Exception as e:
                print(e)
            min_loss = loss

        # save checkpoint
        if (epoch + 1) % cfg.get("ckpt_every_n_epochs", 1) == 0:
            print(f"=> Saving checkpoint [epoch={epoch}]")
            try:
                accelerator.save_state(
                    os.path.join(cfg.ckpt_dir, f"epoch-{epoch:04d}.pt"),
                    safe_serialization=False,
                )
            except Exception as e:
                print(e)

    # save last model
    accelerator.save_state(
        os.path.join(cfg.ckpt_dir, "last.pt"), safe_serialization=False
    )

    # save model to output directory
    accelerator.save(
        model, os.path.join(cfg.output, "model.pt"), safe_serialization=False
    )

    print(f"=> Finished model training [epochs={cfg['max_epoch']}, metric={min_loss}]")
    accelerator.end_training()


if __name__ == "__main__":
    main()
