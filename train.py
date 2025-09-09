import torch
import yaml
import argparse
import warnings
import random
import numpy as np
import os
import pprint
from tqdm import tqdm
from easydict import EasyDict
import torch.nn.functional as F
from torch.utils.data import DataLoader


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'deco_diff')))

# ReCLIP original imports
import open_clip
from models.Necker import Necker
from models.Adapter import Adapter
from models.MapMaker import MapMaker
from models.CoOp import PromptMaker
from models.CoOp import PromptMaker
from utils.losses import FocalLoss, BinaryDiceLoss
from datasets.dataset import TrainDataset, ChexpertTestDataset, BusiTestDataset, BrainMRITestDataset
from utils.misc_helper import get_current_time, create_logger, AverageMeter, compute_imagewise_metrics, compute_pixelwise_metrics

# New components based on the integration plan
from medvae.medvae_main import MVAE
from models.new_components.deco_diff_net import UNET_models
from models.new_components.projection_head import ProjectionHead
from utils.masked_forward import apply_masked_noise
import torchvision.transforms.functional as TF

warnings.filterwarnings('ignore')

def train_one_epoch(
    args,
    models,
    optimizer,
    dataloader,
    criteria,
    epoch,
    logger
):
    # Unpack models and criteria
    medvae_encoder, deco_diff_net, projection_head, adapter, prompt_maker, map_maker, clip_model, necker = models
    mse_criterion, focal_criterion, dice_criterion = criteria

    # Set models to training mode
    deco_diff_net.train()
    projection_head.train()
    adapter.train()
    prompt_maker.train()
    necker.train()

    loss_meter = AverageMeter(args.config.print_freq_step)
    loss_deco_meter = AverageMeter(args.config.print_freq_step)
    loss_mediclip_meter = AverageMeter(args.config.print_freq_step)
    loss_cutpaste_meter = AverageMeter(args.config.print_freq_step) # New meter

    for i, input_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.config.epoch}")):
        
        images = input_data['image'].to(clip_model.device)
        masks = input_data['mask'].to(clip_model.device)
        is_cutpaste = input_data['is_cutpaste']

        total_loss = 0
        
        # --- Task 1: CutPaste Path ---
        cutpaste_indices = torch.where(is_cutpaste)[0]
        if len(cutpaste_indices) > 0:
            cp_images = images[cutpaste_indices]
            cp_masks = masks[cutpaste_indices]

            _, image_tokens = clip_model.encode_image(cp_images, out_layers=args.config.layers_out)
            image_features = necker(image_tokens)
            vision_adapter_features = adapter(image_features)
            prompt_adapter_features = prompt_maker(vision_adapter_features)
            anomaly_map = map_maker(vision_adapter_features, prompt_adapter_features)
            
            loss_cutpaste = focal_criterion(anomaly_map, cp_masks.float()) + dice_criterion(anomaly_map[:, 1, :, :], cp_masks.float())
            total_loss = total_loss + loss_cutpaste
            loss_cutpaste_meter.update(loss_cutpaste.item())

        # --- Task 2: Latent Space (Deco-Diff) Path ---
        latentspace_indices = torch.where(~is_cutpaste)[0]
        if len(latentspace_indices) > 0:
            ls_images = images[latentspace_indices]
            
            # 1. Encode normal images
            with torch.no_grad():
                ls_images_gray = TF.rgb_to_grayscale(ls_images)
                z_normal = medvae_encoder.encode(ls_images_gray).sample()

            # 2. Generator Task
            z_noisy, true_eta, true_mask, t = apply_masked_noise(
                z_normal,
                mask_ratio=args.config.mask_ratio,
                mask_patch_size=args.config.mask_patch_size
            )
            
            # Generate context from prompts
            text_features = prompt_maker(None)

            # Reshape context for the model
            text_features = text_features.permute(1, 0) # Shape: (2, 768)
            batch_size = z_noisy.shape[0]
            context = text_features.unsqueeze(0).repeat(batch_size, 1, 1) # Shape: (N, 2, 768)

            pred_eta = deco_diff_net(z_noisy, t, context=context)
            loss_deco = mse_criterion(pred_eta, true_eta)
            
            with torch.no_grad():
                z_anomalous = z_normal + pred_eta.detach()

            # 3. Detector Task
            f_clip_anomalous = projection_head(z_anomalous)
            f_clip_normal = projection_head(z_normal)

            # Anomaly Path
            adapted_anom_features = adapter([f_clip_anomalous])
            prompted_anom_features = prompt_maker(adapted_anom_features)
            anomaly_map_anom = map_maker(adapted_anom_features, prompted_anom_features)
            
            # Resize true_mask to match anomaly_map dimensions
            target_size = anomaly_map_anom.shape[-2:]
            resized_true_mask = F.interpolate(true_mask.float(), size=target_size, mode='bilinear', align_corners=False)

            loss_anom = focal_criterion(anomaly_map_anom, resized_true_mask) + dice_criterion(anomaly_map_anom[:, 1, :, :], resized_true_mask.squeeze(1))

            # Normal Path
            adapted_norm_features = adapter([f_clip_normal])
            prompted_norm_features = prompt_maker(adapted_norm_features)
            anomaly_map_norm = map_maker(adapted_norm_features, prompted_norm_features)

            # Create a resized zero mask for the normal path
            resized_zero_mask = torch.zeros_like(resized_true_mask)

            loss_norm = focal_criterion(anomaly_map_norm, resized_zero_mask) + dice_criterion(anomaly_map_norm[:, 1, :, :], resized_zero_mask.squeeze(1))
            
            loss_mediclip = loss_anom + loss_norm
            
            latentspace_loss = args.config.w1 * loss_mediclip + args.config.w2 * loss_deco
            total_loss = total_loss + latentspace_loss
            
            loss_deco_meter.update(loss_deco.item())
            loss_mediclip_meter.update(loss_mediclip.item())

        # --- Combined Optimization ---
        # --- Combined Optimization ---
        if isinstance(total_loss, torch.Tensor):
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_meter.update(total_loss.item())

        if (i + 1) % args.config.print_freq_step == 0:
            logger.info(
                f"Epoch: [{epoch+1}/{args.config.epoch}] Iter: [{i+1}/{len(dataloader)}] "
                f"Total Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | "
                f"CutPaste Loss: {loss_cutpaste_meter.avg:.4f} | "
                f"Deco Loss: {loss_deco_meter.avg:.4f} | "
                f"MediCLIP Loss: {loss_mediclip_meter.avg:.4f}"
            )

def validate(args, test_dataloaders, models):
    # Unpack models
    _, _, _, adapter, prompt_maker, map_maker, clip_model, necker = models

    adapter.eval()
    prompt_maker.eval()
    results = {}

    for test_dataset_name, test_dataloader in test_dataloaders.items():
        anomaly_maps, anomaly_gts, image_scores, image_labels = [], [], [], []

        with torch.no_grad():
            for i, input_data in enumerate(tqdm(test_dataloader, desc=f"Validating on {test_dataset_name}")):
                images = input_data['image'].to(clip_model.device)
                
                # Standard ReCLIP inference path
                # Standard ReCLIP inference path
                _, image_tokens = clip_model.encode_image(images, out_layers=args.config.layers_out)
                image_features = necker(image_tokens)
                vision_adapter_features = adapter(image_features)
                prompted_adapter_features = prompt_maker(vision_adapter_features)
                anomaly_map = map_maker(vision_adapter_features, prompted_adapter_features)

                B, _, H, W = anomaly_map.shape
                anomaly_map = anomaly_map[:, 1, :, :]
                
                anomaly_maps.append(anomaly_map.cpu().numpy())
                anomaly_gts.append(input_data['mask'].cpu().numpy())
                
                anomaly_score, _ = torch.max(anomaly_map.view(B, H * W), dim=-1)
                image_scores.extend(anomaly_score.cpu().numpy().tolist())
                image_labels.extend(input_data['is_anomaly'].cpu().numpy().tolist())

        metric = compute_imagewise_metrics(image_scores, image_labels)
        if test_dataset_name == 'busi':
            metric.update(compute_pixelwise_metrics(np.concatenate(anomaly_maps), np.concatenate(anomaly_gts)))
        
        results[test_dataset_name] = metric
    return results


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1. Initialize Models ---
    logger = create_logger("logger", os.path.join(args.config.save_root, 'logger.log'))
    logger.info(f"Using device: {device}")
    logger.info(f"Using random seed: {args.config.random_seed}")
    
    # Load and freeze CLIP model
    clip_model, preprocess, clip_model_cfg = open_clip.create_model_and_transforms(args.config.model_name, args.config.image_size, device=device)
    clip_model.requires_grad_(False)
    clip_model.eval()
    
    # Load and freeze MedVAE model
    medvae_model = MVAE(model_name=args.config.medvae_model_name, modality=args.config.medvae_modality).to(device)
    medvae_model.requires_grad_(False)
    medvae_model.eval()
    medvae_encoder = medvae_model.model

    # Initialize trainable components
    deco_diff_net = UNET_models[args.config.deco_diff_model_size](
        latent_size=args.config.latent_size,
        ncls=1 # ncls is not used for unconditional generation
    ).to(device)

    projection_head = ProjectionHead(
        input_dim=1, # VAE latent has 1 channel
        output_dim=clip_model_cfg['vision_cfg']['width'] # Match the Adapter's input channels
    ).to(device)

    adapter = Adapter(clip_model=clip_model, clip_model_cfg=clip_model_cfg, target=args.config.layers_out_adapter, layers_out_config=args.config.layers_out).to(device)
    prompt_maker = PromptMaker(prompts=args.config.prompts, clip_model=clip_model, n_ctx=args.config.n_learnable_token, CSC=args.config.CSC, class_token_position=args.config.class_token_positions).to(device)
    map_maker = MapMaker(image_size=args.config.image_size).to(device)
    necker = Necker().to(device)

    models = (medvae_encoder, deco_diff_net, projection_head, adapter, prompt_maker, map_maker, clip_model, necker)

    # --- 2. Setup Optimizer ---
    trainable_params = [
        {'params': deco_diff_net.parameters()},
        {'params': projection_head.parameters()},
        {'params': adapter.parameters()},
        {'params': prompt_maker.parameters()},
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.config.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    # --- 3. Setup Data ---
    train_dataset = TrainDataset(args=args.config, source=os.path.join(args.config.data_root, args.config.train_dataset), preprocess=preprocess, k_shot=-1)
    train_dataloader = DataLoader(train_dataset, batch_size=args.config.batch_size, shuffle=True, num_workers=2)
    
    test_dataloaders = {}
    for name in args.config.test_datasets:
        if name == 'chexpert':
            test_dataset = ChexpertTestDataset(args=args.config, source=os.path.join(args.config.data_root, name), preprocess=preprocess)
        elif name == 'brainmri':
            test_dataset = BrainMRITestDataset(args=args.config, source=os.path.join(args.config.data_root, name), preprocess=preprocess)
        elif name == 'busi':
            test_dataset = BusiTestDataset(args=args.config, source=os.path.join(args.config.data_root, name), preprocess=preprocess)
        else:
            raise NotImplementedError(f"Dataset {name} not implemented.")
        test_dataloaders[name] = DataLoader(test_dataset, batch_size=args.config.batch_size, num_workers=2)

    # --- 4. Setup Loss Criteria ---
    criteria = (torch.nn.MSELoss(), FocalLoss(), BinaryDiceLoss())

    # --- 5. Run Training and Validation ---
    best_records = {name: None for name in args.config.test_datasets}
    for epoch in range(args.config.epoch):
        train_one_epoch(args, models, optimizer, train_dataloader, criteria, epoch, logger)

        if (epoch + 1) % args.config.val_freq_epoch == 0:
            results = validate(args, test_dataloaders, models)
            
            for name, result in results.items():
                logger.info(f"Validation Results for {name} at Epoch {epoch+1}: {result}")
                # Checkpoint saving logic based on performance
                current_performance = np.mean(list(result.values()))
                if best_records[name] is None or current_performance > best_records[name]:
                    best_records[name] = current_performance
                    logger.info(f"New best performance for {name}: {current_performance:.4f}. Saving checkpoint.")
                    torch.save({
                        "deco_diff_net_state_dict": deco_diff_net.state_dict(),
                        "projection_head_state_dict": projection_head.state_dict(),
                        "adapter_state_dict": adapter.state_dict(),
                        "prompt_state_dict": prompt_maker.state_dict(),
                    }, os.path.join(args.config.save_root, f'best_model_{name}.pkl'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Integrated Anomaly Detection Model")
    parser.add_argument("--config_path", type=str, default='config/reclip_integrated.yaml', help="Path to the integrated config file")
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    args.config = config
    
    # Setup logging and directories
    current_time = get_current_time()
    args.config.save_root = os.path.join(args.config.save_root, current_time)
    if not os.path.exists(args.config.save_root):
        os.makedirs(args.config.save_root)

    # Setup random seed
    seed = args.config.get('random_seed', random.randint(0, 2**32 - 1))
    args.config.random_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main(args)