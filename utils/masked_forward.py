import torch
import numpy as np
import math

# --- Helper functions and classes adapted from deco_diff/diffusion/ --- 

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + torch.zeros(broadcast_shape, device=timesteps.device)

class SimpleGaussianDiffusion:
    def __init__(self, betas):
        self.betas = np.array(betas, dtype=np.float64)
        self.num_timesteps = int(self.betas.shape[0])

        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.deviatoin_coeff = (1-self.sqrt_alphas_cumprod)/self.sqrt_one_minus_alphas_cumprod

    def q_sample(self, x_start, t, mask, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        # The noisy part is calculated using the diffusion formula
        noisy_part = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        
        # The final output is a combination of the original and noisy parts based on the mask
        # Where mask is 1, we keep the original x_start (context), where it's 0, we use the noisy version.
        return (mask * x_start) + ((1 - mask) * noisy_part)

    def direction_of_deviation(self, noise, x_start, mask, t):
        assert noise.shape == x_start.shape
        dod = noise - _extract_into_tensor(self.deviatoin_coeff, t, x_start.shape) * x_start
        return (1 - mask) * dod

# --- Masking function adapted from deco_diff/train_DeCo_Diff.py ---

def random_mask(x, mask_ratios, mask_patch_size=1):
    n, c, w, h = x.shape
    size = int(np.prod(x.shape[2:]) / (mask_patch_size**2))
    mask = torch.zeros((n,c,size)).to(x.device)
    for b in range(n):
        masked_indexes = np.arange(size)
        np.random.shuffle(masked_indexes)
        masked_indexes = masked_indexes[:int(size * (1 - mask_ratios[b]))]
        mask[b,:, masked_indexes] = 1
    mask = mask.reshape(n, c, int(w/mask_patch_size), int(h/mask_patch_size))
    mask = mask.repeat_interleave(mask_patch_size, dim=2).repeat_interleave(mask_patch_size, dim=3)
    return mask

# --- Main utility function to be called from the training script ---

def apply_masked_noise(z_normal, mask_ratio, mask_patch_size, diffusion_steps=10):
    # 1. Create a simplified diffusion object on the fly
    betas = get_named_beta_schedule("squaredcos_cap_v2", diffusion_steps)
    diffusion = SimpleGaussianDiffusion(betas)

    # 2. Create a random mask
    mask_ratios_list = [mask_ratio] * z_normal.shape[0]
    true_mask = random_mask(z_normal, mask_ratios=mask_ratios_list, mask_patch_size=mask_patch_size)

    # 3. Generate timestep and noise
    t = torch.randint(0, diffusion.num_timesteps, (z_normal.shape[0],), device=z_normal.device)
    noise = torch.randn_like(z_normal)

    # 4. Apply masked noise to get z_noisy
    z_noisy = diffusion.q_sample(x_start=z_normal, t=t, mask=true_mask, noise=noise)

    # 5. Calculate the ground truth deviation for the loss function
    true_eta = diffusion.direction_of_deviation(noise=noise, x_start=z_normal, mask=true_mask, t=t)

    return z_noisy, true_eta, true_mask, t