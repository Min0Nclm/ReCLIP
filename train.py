import open_clip
import torch
import yaml
from easydict import EasyDict
from models.Necker import Necker
from models.Adapter import Adapter
import math
import argparse
import warnings
import random
import numpy as np
from utils.misc_helper import *
from torch.utils.data import DataLoader
from models.MapMaker import MapMaker
from utils.losses import FocalLoss,BinaryDiceLoss
from datasets.dataset import TrainDataset,\
                                ChexpertTestDataset,\
                                BusiTestDataset,\
                                BrainMRITestDataset
import pprint
from tqdm import tqdm
import torchvision
import os
warnings.filterwarnings('ignore')


@torch.no_grad()
def make_vision_takens_info(model,model_cfg,layers_out):

    img = torch.ones((1,3,model_cfg['vision_cfg']['image_size'],
                          model_cfg['vision_cfg']['image_size'])).to(model.device)

    img_feature,tokens = model.encode_image(img,layers_out)

    if len(tokens[0].shape)==3:
        model.token_size= [int(math.sqrt(token.shape[1]-1)) for token in tokens]
        model.token_c= [token.shape[-1]  for token in tokens]
    else:
        model.token_size = [token.shape[2] for token in tokens]
        model.token_c = [token.shape[1] for token in tokens]

    model.embed_dim = model_cfg['embed_dim']
    print("model token size is {}".format(model.token_size)," model token dim is {}".format(model.token_c))


def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, preprocess, model_cfg = open_clip.create_model_and_transforms(args.config.model_name, args.config.image_size, device=device)

    for param in model.parameters():
        param.requires_grad_(False)

    args.config.model_cfg = model_cfg

    make_vision_takens_info(model,
                            args.config.model_cfg,
                            args.config.layers_out)

    current_time = get_current_time()
    args.config.save_root=os.path.join(args.config.save_root,current_time)

    if not os.path.exists(args.config.save_root):
        os.makedirs(args.config.save_root)

    # DEBUG: Set epoch to 1 and create debug directory
    args.config.epoch = 1
    debug_dir = os.path.join(args.config.save_root, 'debug_images')
    os.makedirs(debug_dir, exist_ok=True)

    logger = create_logger("logger",os.path.join(args.config.save_root,'logger.log'))
    logger.info("Epoch is forced to 1 for debugging image generation.")

    # set random seed
    seed = args.config.get('random_seed')
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        args.config.random_seed = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.info("config: {}".format(pprint.pformat(args)))
    logger.info(f"Using random seed: {seed}")

    necker = Necker(clip_model=model).to(model.device)
    adapter = Adapter(clip_model=model,target=args.config.model_cfg['embed_dim']).to(model.device)

    if args.config.prompt_maker=='coop':
        from models.CoOp import PromptMaker
        logger.info("load CoOp")
    else:
        raise NotImplementedError("type of prompt must in ['coop']")

    prompt_maker = PromptMaker(
        prompts=args.config.prompts,
        clip_model=model,
        n_ctx= args.config.n_learnable_token,
        CSC = args.config.CSC,
        class_token_position=args.config.class_token_positions,
    ).to(model.device)

    map_maker = MapMaker(image_size=args.config.image_size).to(model.device)

    optimizer = torch.optim.Adam([
            {'params': prompt_maker.prompt_learner.parameters(),'lr': 0.001},
            {'params': adapter.parameters(),"lr":0.001},
        ], lr=0.001, betas=(0.5, 0.999))

    train_dataset = TrainDataset(args=args.config,
                                    source=os.path.join(args.config.data_root,args.config.train_dataset),
                                    preprocess=preprocess,
                                    # Set k_shot to -1 to use the full dataset, which is required to match the pre-computed features.
                                    k_shot=-1)

    train_dataloader = DataLoader(train_dataset, batch_size=args.config.batch_size, shuffle=True, num_workers=2)

    test_dataloaders = {}
    best_record = {}

    for test_dataset_name in args.config.test_datasets:

        if test_dataset_name == 'chexpert':
            test_dataset = ChexpertTestDataset( args=args.config,
                                            source=os.path.join(args.config.data_root,test_dataset_name),
                                            preprocess=preprocess,
                                            )

        elif test_dataset_name =='brainmri':

            test_dataset = BrainMRITestDataset(
                                            args=args.config,
                                            source=os.path.join(args.config.data_root,test_dataset_name),
                                            preprocess=preprocess,
                                            )
        elif test_dataset_name =='busi':

            test_dataset = BusiTestDataset(
                                            args=args.config,
                                            source=os.path.join(args.config.data_root,test_dataset_name),
                                            preprocess=preprocess)
        else:
            raise NotImplementedError("dataset must in ['chexpert','busi','brainmri'] ")

        test_dataloader = DataLoader(test_dataset, batch_size=args.config.batch_size,num_workers=2)
        test_dataloaders[test_dataset_name]=test_dataloader
        best_record[test_dataset_name]=None

    logger.info("train data ({}) len {}".format(args.config.train_dataset,len(train_dataset)))

    for test_dataset_name in test_dataloaders:
        logger.info("test data ({}) len {}".format(test_dataset_name, len(test_dataloaders[test_dataset_name].dataset)))

    for task_name in args.config.anomaly_tasks:
        logger.info("anomaly syn task is {}, sampling probability is {}".format(task_name,args.config.anomaly_tasks[task_name]))

    for epoch in range(0, args.config.epoch):
        last_iter = epoch * len(train_dataloader)

        train_one_epoch(
            args,
            train_dataloader,
            optimizer,
            epoch,
            last_iter,
            logger,
            model,
            necker,
            adapter,
            prompt_maker,
            map_maker,
        )

        # DEBUG: Skipping validation loop as per debug request.
        pass

    # At the end of training, print the seed
    final_seed_message = f"Training finished. The random seed used for this run was: {args.config.random_seed}"
    logger.info(final_seed_message)
    print(final_seed_message)


def train_one_epoch(
            args,
            train_dataloader,
            optimizer,
            epoch,
            start_iter,
            logger,
            clip_model,
            necker,
            adapter,
            prompt_maker,
            map_maker,
):

    loss_meter = AverageMeter(args.config.print_freq_step)

    focal_criterion = FocalLoss()
    dice_criterion = BinaryDiceLoss()

    adapter.train()
    prompt_maker.train()

    for i, input in enumerate(train_dataloader):
        # DEBUG: Save images from the first batch and break.
        if i == 0:
            logger.info("Saving debug images for the first batch...")
            debug_dir = os.path.join(args.config.save_root, 'debug_images')

            original_images = input['original_image']
            augmented_images = input['image']
            # Unsqueeze mask to have a channel dimension (B, 1, H, W) for grid saving
            masks = input['mask'].unsqueeze(1).float()
            # To make the grid visually correct, we need to convert the single-channel mask to 3-channel
            masks_rgb = masks.repeat(1, 3, 1, 1)

            # Create a grid showing original, augmented, and mask
            comparison_grid = torch.cat([original_images, augmented_images, masks_rgb], dim=0)

            save_path = os.path.join(debug_dir, f'epoch_{epoch+1}_batch_{i}_comparison.png')
            # nrow will be the batch size, so we get rows of [original, augmented, mask]
            torchvision.utils.save_image(comparison_grid, save_path, nrow=original_images.size(0), normalize=True)
            logger.info(f"Saved comparison grid to {save_path}")


        curr_step = start_iter + i

        images = input['image'].to(clip_model.device)
        gt_mask = input['mask'].to(clip_model.device)

        with torch.no_grad():
            _, image_tokens = clip_model.encode_image(images,out_layers=args.config.layers_out)
            image_features = necker(image_tokens)

        vision_adapter_features = adapter(image_features)
        propmt_adapter_features = prompt_maker(vision_adapter_features)
        anomaly_map = map_maker(vision_adapter_features,propmt_adapter_features)

        loss = []

        loss.append(focal_criterion(anomaly_map,gt_mask))
        loss.append(dice_criterion(anomaly_map[:, 1, :, :],gt_mask))

        loss = torch.sum(torch.stack(loss))
        loss_meter.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (curr_step + 1) % args.config.print_freq_step == 0:
            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})\t"
                    .format(
                    epoch+1 ,
                    args.config.epoch,
                    curr_step + 1,
                    len(train_dataloader) * args.config.epoch,
                    loss=loss_meter,
                )
            )
        
        # DEBUG: Break after the first batch
        if i == 0:
            logger.info("Breaking after first batch for debugging.")
            break



def validate(args, test_dataloaders, epoch, clip_model, necker, adapter, prompt_maker, map_maker):

    adapter.eval()
    prompt_maker.eval()
    results = {}

    for test_dataset_name in test_dataloaders:
        test_dataloader = test_dataloaders[test_dataset_name]

        anomaly_maps = []
        anomaly_gts = []

        image_scores = []
        image_labels = []

        with torch.no_grad():
            for i, input in enumerate(tqdm(test_dataloader,desc=test_dataset_name)):

                images = input['image'].to(clip_model.device)

                _, image_tokens = clip_model.encode_image(images, out_layers=args.config.layers_out)
                image_features = necker(image_tokens)
                vision_adapter_features = adapter(image_features)
                propmt_adapter_features = prompt_maker(vision_adapter_features)
                anomaly_map = map_maker(vision_adapter_features, propmt_adapter_features)

                B,_,H,W = anomaly_map.shape

                anomaly_map = anomaly_map[:,1,:,:]
                anomaly_gt = input['mask']

                anomaly_maps.append(anomaly_map.cpu().numpy())
                anomaly_gts.append(anomaly_gt.cpu().numpy())

                anomaly_score,_ = torch.max(anomaly_map.view((B,H*W)), dim=-1)

                image_scores.extend(anomaly_score.cpu().numpy().tolist())
                image_labels.extend(input['is_anomaly'].cpu().numpy().tolist())

        metric = compute_imagewise_metrics(image_scores,image_labels)

        if test_dataset_name=='busi':
            metric.update(compute_pixelwise_metrics(anomaly_maps,anomaly_gts))

        results[test_dataset_name] = metric
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MediCLIP")
    parser.add_argument("--config_path", type=str, default='config/brainmri.yaml', help="model configs")
    parser.add_argument("--k_shot", type=int, default=16, help="normal image number")
    parser.add_argument("--num_support_samples", type=int, default=5, help="Number of similar support samples to use for CutPaste, k.")
    args = parser.parse_args()

    # Add the num_support_samples to the config object so it's available in the dataset class
    with open(args.config_path) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    config.num_support_samples = args.num_support_samples
    args.config = config

    torch.multiprocessing.set_start_method("spawn")
    main(args)