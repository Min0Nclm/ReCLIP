import argparse
from genericpath import isdir
from medvae.utils.extras import create_directory, cite_function
import torch
import nibabel as nib
import os
from tqdm import tqdm
import numpy as np
from os.path import join as pjoin
from medvae import MVAE

def parse_arguments():
    parser = argparse.ArgumentParser(description='Use this to run inference with MedVAE. This function is used when '
                                                 'you want to manually specify a folder containing an pretrained MedVAE '
                                                 'model. ',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. These should contain files that you want the latent to be processed for'
                        'Remember *.pngs are for 2D images and *.nii.gz are for 3D images. ' 
                        'The filename should not have a "." in it apart from suffix')
    
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted latents will '
                             'have the same name as their source images.')
    parser.add_argument(
            '-model_name', type=str, required=True,
            help=(
                "There are six MedVAE models that can be used for inference. Choose between:\n"
                "(1) medvae_4_1_2d: 2D images with a 4x compression in each dim (16x total) with a 1 channel latent.\n"
                "(2) medvae_4_3_2d: 2D images with a 4x compression in each dim (64x total) with a 3 channel latent.\n"
                "(3) medvae_8_1_2d: 2D images with an 8x compression in each dim (64x total) with a 1 channel latent.\n"
                "(4) medvae_8_4_2d: 2D images with an 8x compression in each dim (64x total) with a 4 channel latent.\n"
                "(5) medvae_4_1_3d: 3D images with a 4x compression in each dim (64x total) with a 1 channel latent.\n"
                "(6) medvae_8_1_3d: 3D images with an 8x compression in each dim (64x total) with a 1 channel latent.\n"
            )
        )
    
    parser.add_argument('-modality', type=str, required=True,
                        help='Modality of the input images. Choose between xray, ct, or mri.')
    
    parser.add_argument('-ckpt_path', type=str, required=False,
                        help='Path to the checkpoint file. If provided, the model will be loaded from the weight in this file.' + 
                        'Note: This should be a ckpt after stage 2 2D and 3D finetuning. If you want stage 1, then modification need to be made')
    
    parser.add_argument('-roi_size', type=int, default=160, required=False,
                        help='Region of interest size for 3D models. This is the maximum dimension size allowed for processing on the GPU.')
    
    parser.add_argument('-device', type=str, default='cuda', required=False,
                            help="Use this to set the device the inference should run with. Available options are 'cuda' "
                                "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                                "Use CUDA_VISIBLE_DEVICES=X medvae_inference [...] instead!")    
    
    # Print a message to cite the medvae paper
    cite_function()
    
    args, unknownargs = parser.parse_known_args()
    if unknownargs:
        print(f"Ignoring arguments: {unknownargs}")
    
    # Check if input folder exists
    assert isdir(args.i), f"Input folder {args.i} does not exist."

    # Create output directory if it does not exist
    create_directory(args.o)    
    
    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    
    if args.device == 'cpu':
        # let's allow torch to use lots of threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help medvae if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')
    
    return args, device

def __init__():

    args, device = parse_arguments()
    
    # Build the model and transform
    model = MVAE(args.model_name, args.modality, args.roi_size).to(device)
    
    # If a checkpoint path is provided, load the model from the weight in this file
    if args.ckpt_path:
        model.init_from_ckpt(args.ckpt_path, state_dict=False)
        
    model.requires_grad_(False)
    model.eval()
        
    # Run inference on the input folder
    print("Running inference at {}".format(args.i))
    for fpath in tqdm(os.listdir(args.i), total = len(os.listdir(args.i))):
        
        img = model.apply_transform(pjoin(args.i, fpath)).to(device)
        latent = model(img).detach().cpu().numpy()
                
        # Save the latent
        nib.save(nib.Nifti1Image(latent, np.eye(4)), pjoin(args.o, fpath.split('.')[0] + ".nii.gz"))
    
    print("Inference complete! Output saved at {}".format(args.o))
    
def main():
    __init__()