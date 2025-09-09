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

def image_cutpaste(images, device, area_ratio_range=(0.02, 0.15)):
    """
    Performs CutPaste on image tensors.
    Args:
        images: A batch of image tensors (N, C, H, W).
        device: The torch device.
        area_ratio_range: The range for the area of the patch relative to the total area.
    Returns:
        cp_images: The batch with image patches pasted on.
        cp_masks: The ground truth mask for the pasted patches.
    """
    N, C, H, W = images.shape

    # Shuffle for source-target pairing
    images_source = images
    images_target = images[torch.randperm(N)]

    # Generate random patch sizes and positions
    area = H * W
    patch_area = torch.empty(N, device=device).uniform_(area_ratio_range[0], area_ratio_range[1]) * area
    patch_ratio = (patch_area / area).sqrt()
    patch_h = (patch_ratio * H).int()
    patch_w = (patch_ratio * W).int()

    box_x1 = torch.randint(0, W, (N,), device=device)
    box_y1 = torch.randint(0, H, (N,), device=device)

    # Create masks and apply CutPaste
    cp_masks = torch.zeros((N, 1, H, W), device=device)
    cp_images = images_target.clone() # Start with target images

    for i in range(N):
        x1, y1 = box_x1[i], box_y1[i]
        h, w = patch_h[i], patch_w[i]
        
        x2 = torch.clamp(x1 + w, 0, W)
        y2 = torch.clamp(y1 + h, 0, H)
        
        # Ensure patch dimensions are valid
        h_actual = y2 - y1
        w_actual = x2 - x1

        if h_actual > 0 and w_actual > 0:
            cp_masks[i, :, y1:y2, x1:x2] = 1
            cp_images[i, :, y1:y2, x1:x2] = images_source[i, :, y1:y2, x1:x2]

    return cp_images, cp_masks


def latent_cutpaste(z_batch, device, area_ratio_range=(0.02, 0.15)):
    """
    Performs CutPaste in the latent space.
    Args:
        z_batch: A batch of latent representations (N, C, H, W).
        device: The torch device.
        area_ratio_range: The range for the area of the patch relative to the total area.
    Returns:
        z_anomalous: The batch with latent patches pasted on.
        true_mask: The ground truth mask for the pasted patches.
    """
    N, C, H, W = z_batch.shape

    # Shuffle for source-target pairing
    z_source = z_batch
    z_target = z_batch[torch.randperm(N)]

    # Generate random patch sizes and positions
    area = H * W
    patch_area = torch.empty(N, device=device).uniform_(area_ratio_range[0], area_ratio_range[1]) * area
    patch_ratio = (patch_area / area).sqrt()
    patch_h = (patch_ratio * H).int()
    patch_w = (patch_ratio * W).int()

    box_x1 = torch.randint(0, W, (N,), device=device)
    box_y1 = torch.randint(0, H, (N,), device=device)

    # Create masks
    true_mask = torch.zeros((N, 1, H, W), device=device)
    for i in range(N):
        x1, y1 = box_x1[i], box_y1[i]
        h, w = patch_h[i], patch_w[i]
        
        x2 = torch.clamp(x1 + w, 0, W)
        y2 = torch.clamp(y1 + h, 0, H)
        
        true_mask[i, :, y1:y2, x1:x2] = 1

    # Apply CutPaste
    z_anomalous = z_target * (1 - true_mask) + z_source * true_mask

    return z_anomalous, true_mask


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
    loss_pixel_cp_meter = AverageMeter(args.config.print_freq_step) # Renamed from loss_cutpaste_meter
    loss_latent_cp_meter = AverageMeter(args.config.print_freq_step)
    loss_deco_meter = AverageMeter(args.config.print_freq_step)
    loss_identity_meter = AverageMeter(args.config.print_freq_step)
    loss_mediclip_meter = AverageMeter(args.config.print_freq_step)
    
    # Check task weights and normalize if necessary
    task_weights = args.config.task_weights
    total_weight = sum(task_weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        logger.warning(f"Task weights do not sum to 1.0 (sum={total_weight}). Normalizing...")
        for k in task_weights:
            task_weights[k] /= total_weight

    for i, input_data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.config.epoch}")):
        
        images = input_data['image'].to(clip_model.device)
        # masks = input_data['mask'].to(clip_model.device) # Mask from dataset is no longer directly used for task assignment
        # is_cutpaste = input_data['is_cutpaste'] # No longer directly used for task assignment

        total_loss = 0
        batch_size = images.shape[0]

        # Dynamically assign samples to tasks based on weights
        task_assignment_indices = torch.randperm(batch_size)
        
        current_idx = 0
        task_indices = {}
        for task_name, weight in task_weights.items():
            num_samples = int(batch_size * weight)
            task_indices[task_name] = task_assignment_indices[current_idx : current_idx + num_samples]
            current_idx += num_samples
        
        # Ensure all samples are assigned (assign remaining to the last task)
        if current_idx < batch_size:
            remaining_indices = task_assignment_indices[current_idx:]
            last_task_name = list(task_weights.keys())[-1]
            task_indices[last_task_name] = torch.cat((task_indices[last_task_name], remaining_indices)) # Concatenate to the last task


        # --- Task 1: Pixel-space CutPaste ---
        pixel_cp_batch_indices = task_indices.get('pixel_cutpaste', torch.tensor([], dtype=torch.long))
        if len(pixel_cp_batch_indices) > 0:
            pcp_images = images[pixel_cp_batch_indices]
            
            cp_images, cp_masks = image_cutpaste(pcp_images, clip_model.device)

            _, image_tokens = clip_model.encode_image(cp_images, out_layers=args.config.layers_out)
            image_features = necker(image_tokens)
            vision_adapter_features = adapter(image_features)
            prompt_adapter_features = prompt_maker(vision_adapter_features)
            anomaly_map = map_maker(vision_adapter_features, prompt_adapter_features)
            
            loss_pixel_cp = focal_criterion(anomaly_map, cp_masks.float()) + dice_criterion(anomaly_map[:, 1, :, :], cp_masks.float())
            total_loss = total_loss + loss_pixel_cp
            loss_pixel_cp_meter.update(loss_pixel_cp.item())

        # --- Task 2: Latent-space CutPaste ---
        latent_cp_batch_indices = task_indices.get('latent_cutpaste', torch.tensor([], dtype=torch.long))
        if len(latent_cp_batch_indices) > 0:
            lcp_images = images[latent_cp_batch_indices]

            with torch.no_grad():
                lcp_images_gray = TF.rgb_to_grayscale(lcp_images)
                z_normal = medvae_encoder.encode(lcp_images_gray).sample()
            
            z_anomalous, true_mask = latent_cutpaste(z_normal, clip_model.device)

            f_clip_anomalous = projection_head(z_anomalous)
            adapted_anom_features = adapter([f_clip_anomalous])
            prompted_anom_features = prompt_maker(adapted_anom_features)
            anomaly_map_anom = map_maker(adapted_anom_features, prompted_anom_features)

            target_size = anomaly_map_anom.shape[-2:]
            resized_true_mask = F.interpolate(true_mask.float(), size=target_size, mode='bilinear', align_corners=False)

            loss_latent_cp = focal_criterion(anomaly_map_anom, resized_true_mask) + dice_criterion(anomaly_map_anom[:, 1, :, :], resized_true_mask.squeeze(1))
            total_loss = total_loss + loss_latent_cp
            loss_latent_cp_meter.update(loss_latent_cp.item())

        # --- Task 3: Deco-Diff ---
        deco_diff_batch_indices = task_indices.get('deco_diff', torch.tensor([], dtype=torch.long))
        if len(deco_diff_batch_indices) > 0:
            dd_images = images[deco_diff_batch_indices]
            
            with torch.no_grad():
                dd_images_gray = TF.rgb_to_grayscale(dd_images)
                z_normal = medvae_encoder.encode(dd_images_gray).sample()

            z_noisy, true_eta, true_mask, t = apply_masked_noise(
                z_normal,
                mask_ratio=args.config.mask_ratio,
                mask_patch_size=args.config.mask_patch_size
            )
            
            text_features = prompt_maker(None)
            text_features = text_features.permute(1, 0)
            batch_size_dd = z_noisy.shape[0]
            context = text_features.unsqueeze(0).repeat(batch_size_dd, 1, 1)

            pred_eta = deco_diff_net(z_noisy, t, context=context)
            loss_deco = mse_criterion(pred_eta, true_eta)
            
            with torch.no_grad():
                z_anomalous = z_normal + pred_eta.detach()

            f_clip_anomalous = projection_head(z_anomalous)
            f_clip_normal = projection_head(z_normal)

            adapted_anom_features = adapter([f_clip_anomalous])
            prompted_anom_features = prompt_maker(adapted_anom_features)
            anomaly_map_anom = map_maker(adapted_anom_features, prompted_anom_features)
            
            target_size = anomaly_map_anom.shape[-2:]
            resized_true_mask = F.interpolate(true_mask.float(), size=target_size, mode='bilinear', align_corners=False)
            loss_anom = focal_criterion(anomaly_map_anom, resized_true_mask) + dice_criterion(anomaly_map_anom[:, 1, :, :], resized_true_mask.squeeze(1))

            adapted_norm_features = adapter([f_clip_normal])
            prompted_norm_features = prompt_maker(adapted_norm_features)
            anomaly_map_norm = map_maker(adapted_norm_features, prompted_norm_features)
            resized_zero_mask = torch.zeros_like(resized_true_mask)
            loss_norm = focal_criterion(anomaly_map_norm, resized_zero_mask) + dice_criterion(anomaly_map_norm[:, 1, :, :], resized_zero_mask.squeeze(1))
            
            loss_mediclip = loss_anom + loss_norm
            
            # No w1, w2 anymore, direct sum
            deco_diff_task_loss = loss_mediclip + args.config.deco_loss_weight * loss_deco 
            total_loss = total_loss + deco_diff_task_loss
            
            loss_deco_meter.update(loss_deco.item())
            loss_mediclip_meter.update(loss_mediclip.item()) # Update mediclip meter for deco_diff task

        # --- Task 4: Identity Task ---
        identity_batch_indices = task_indices.get('identity', torch.tensor([], dtype=torch.long))
        if len(identity_batch_indices) > 0:
            id_images = images[identity_batch_indices]

            with torch.no_grad():
                id_images_gray = TF.rgb_to_grayscale(id_images)
                z_normal = medvae_encoder.encode(id_images_gray).sample()
            
            f_clip_normal = projection_head(z_normal)
            adapted_norm_features = adapter([f_clip_normal])
            prompted_norm_features = prompt_maker(adapted_norm_features)
            anomaly_map_norm = map_maker(adapted_norm_features, prompted_norm_features)

            target_size = anomaly_map_norm.shape[-2:]
            resized_zero_mask = torch.zeros((anomaly_map_norm.shape[0], 1, target_size[0], target_size[1]), device=clip_model.device)

            loss_identity = focal_criterion(anomaly_map_norm, resized_zero_mask) + dice_criterion(anomaly_map_norm[:, 1, :, :], resized_zero_mask.squeeze(1))
            total_loss = total_loss + loss_identity
            loss_identity_meter.update(loss_identity.item())


        # --- Combined Optimization ---
        if isinstance(total_loss, torch.Tensor):
            loss_val = total_loss.item()
            total_loss = total_loss / args.config.gradient_accumulation_steps
            total_loss.backward()

            if (i + 1) % args.config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            loss_meter.update(loss_val)

        if (i + 1) % args.config.print_freq_step == 0:
            logger.info(
                f"Epoch: [{epoch+1}/{args.config.epoch}] Iter: [{i+1}/{len(dataloader)}] "
                f"Total Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) | "
                f"PixelCP Loss: {loss_pixel_cp_meter.avg:.4f} | "
                f"LatentCP Loss: {loss_latent_cp_meter.avg:.4f} | "
                f"Deco Loss: {loss_deco_meter.avg:.4f} | "
                f"MedCLIP Loss: {loss_mediclip_meter.avg:.4f} | "
                f"Identity Loss: {loss_identity_meter.avg:.4f}"
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
    print(f"DEBUG: Length of train_dataloader: {len(train_dataloader)}") # Added for debugging
    
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

    logger.info("--- Training Finished ---")
    for name, best_perf in best_records.items():
        if best_perf is not None:
            logger.info(f"Final best performance for {name}: {best_perf:.4f}")
        else:
            logger.info(f"No best performance recorded for {name}.")

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