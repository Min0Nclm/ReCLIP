"""
Miscellaneous utility functions.
"""

import os
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn


def create_directory(directory: str) -> None:
    """
    Create a directory if it does not exist.

    Parameters:
        directory (str): The directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def cite_function():
    """
    Print a message to cite the MedVAE paper.
    """
    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using MedVAE: Varma, M., Kumar, A., van der Sluijs, R., Ostmeier, S., "
        "Blankemeier, L., Chambon, P., Bluethgen, C., Prince, J., Langlotz, C., "
        "Chaudhari, A. (2025). MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders. "
        "arXiv preprint arXiv:2502.14753.\n"
        "#######################################################################\n"
    )


""" 
Calculate the region of interest (ROI) size for each dimension of the input image shape.

This function determines the appropriate ROI size based on the target GPU dimension.
If a dimension exceeds the target GPU dimension, it finds the largest power of 2 that 
results in a size less than the target dimension.

@param image_shape: A tuple or list representing the shape of the input image (e.g., (depth, height, width)).
@param target_gpu_dim: The maximum dimension size allowed for processing on the GPU (default is 160).
@return: A list of calculated ROI sizes for each dimension of the input image.
"""


def roi_size_calc(image_shape, target_gpu_dim=160):
    roi_size = []
    for dim in image_shape:
        if dim > target_gpu_dim:
            target_shape = target_gpu_dim
            # For loop for powers of 2
            for power in [2**i for i in range(8)]:
                if dim // power < target_shape:
                    roi_size.append(dim // power)
                    break
        else:
            roi_size.append(dim)

    return roi_size


"""
This function sanitizes the keyword arguments for a DataLoader by converting the 'num_workers' argument to an integer.
This is necessary when 'num_workers' is retrieved from the OS environment, as it might be stored as a string.
"""


def sanitize_dataloader_kwargs(kwargs):
    if "num_workers" in kwargs:
        kwargs["num_workers"] = int(kwargs["num_workers"])

    return kwargs


def set_seed(seed: int):
    """Seed the RNGs."""

    print(f"=> Setting seed [seed={seed}]")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True

    print("=> Setting a seed slows down training considerably!")


def get_weight_dtype(accelerator):
    """Get the weight dtype from the accelerator."""

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    return weight_dtype
