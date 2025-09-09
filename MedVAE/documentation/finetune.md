# 🔧 MedVAE Finetuning Documentation

Our finetuning framework leverages the flexibility and power of [Hydra](https://github.com/facebookresearch/hydra) and [HuggingFace Accelerate](https://github.com/huggingface/accelerate), providing multi-GPU training support and easy integration with [Weights and Biases (wandb)](https://wandb.ai/) for experiment tracking. We recommend using Python 3.9 and installing packages specified in our `pyproject.toml` file.

While MedVAE is primarily tested and validated on medical imaging modalities like X-ray, MRI, and CT, our framework can potentially support finetuning on datasets from other imaging modalities. However, please note that performance has not been validated outside these modalities.

## Configuration Structure (Hydra)

Through Hydra, we are able to modify our hyperparameters for finetuning through [config files](../configs/). Our configuration files for finetuning are organized into three main directories:

- **Criterion:** Contains loss function configurations for various training stages. Notable losses include:

  - `lpips_with_discriminator`: Used for 2D stage 1 and 3D stage 2 finetuning.
  - `biomedclip`: Used specifically for 2D stage 2 finetuning.

- **Dataloader:** Includes preconfigured data loaders for different imaging types:

  - `mmgs.yaml`: Loads 2D Full-Field Digital Mammograms (FFDMs).
  - `mri_ct_3d.yaml`: Loads 3D MRI and CT imaging data.

- **Experiment:** Centralizes and abstracts hyperparameters, allowing easy customization of your finetuning process. You will mainly need to change parameters in your experiment file for different finetuning runs.

## Running Finetuning Stages

We provide example configuration files for both 2D and 3D image finetuning using a 4x downscaled latent representation. Adjust the GPU identifier (`CUDA_VISIBLE_DEVICES`) and batch size according to your hardware capabilities and dataset size. Typically, larger batch sizes are recommended.

### Stage 1 (2D Finetuning)

Finetune the base model using 4x downsizing with either 1-channel or 3-channel latent representations. Key parameters to update in the experiment configuration file include `dataloader`, `dataset_name`, and `task_name`.

- **1-channel latent:**

```python
CUDA_VISIBLE_DEVICES=0 medvae_finetune experiment=medvae_4x_1c_2d_finetuning
```

- **3-channel latent (using LoRA):**

```python
CUDA_VISIBLE_DEVICES=0 medvae_finetune experiment=medvae_4x_3c_2d_finetuning
```

### Stage 2 (2D Finetuning)

Stage 2 involves training a lightweight projection layer to enrich latent representations for downstream tasks. Ensure the `stage2_ckpt` parameter in the experiment file points to your stage 1 finetuning checkpoint.

- **1-channel latent:**

```python
CUDA_VISIBLE_DEVICES=0 medvae_finetune_s2 experiment=medvae_4x_1c_2d_s2_finetuning
```

- **3-channel latent (using LoRA):**

```python
CUDA_VISIBLE_DEVICES=0 medvae_finetune_s2 experiment=medvae_4x_3c_2d_s2_finetuning
```

### Stage 2 (3D Finetuning)

Directly finetune 3D latent representations. Similar to stage 1, ensure appropriate parameters (`dataloader`, `dataset_name`, `task_name`) are correctly configured.

```python
CUDA_VISIBLE_DEVICES=0 medvae_finetune experiment=medvae_4x_1c_3d_finetuning
```

## Multi-GPU Training

Multi-GPU training is seamlessly supported through Accelerate. Configure Accelerate appropriately, then specify multiple GPUs as shown below:

```python
CUDA_VISIBLE_DEVICES=1,2,3,4 medvae_finetune experiment=medvae_4x_1c_2d_finetuning
```

## CSV File Creation

For your data, you will need to create a CSV-file with the appropriate train, val, and test splits. Please see [create_csv.ipynb](create_csv.ipynb) to assist on this task.

## Symbolic Links for Data

If you prefer not to modify the dataloader configuration, you can symbolically link your dataset to the default data directory:

```bash
ln -s <your_data_directory> <medvae_installation_directory>/medvae/data
```

## Creating Custom Data Loaders

For maximum flexibility, we recommend creating your own data loading method. You will need to change the loader in the dataloader configuration file. Refer to the [`loaders.py`](../medvae/utils/loaders.py) file to see examples:

- `load_2d_finetune`
- `load_mri_3d_finetune`
- `load_ct_3d_finetune`

Use these functions as templates for developing loaders tailored to your specific dataset structure and requirements.

## Logging with Weights & Biases

To log training runs with wandb, ensure your wandb API key is set up. Enable logging as follows:

```python
CUDA_VISIBLE_DEVICES=0 medvae_finetune experiment=medvae_4x_1c_2d_finetuning logger=wandb
```

Note: We recommend using `wandb` version `0.14.0`. We have noticed logging errors with other versions with our setup.

## Inference Post-Finetuning

Use our built-in inference engine to perform inference on your finetuned models:

```python
medvae_inference -i INPUT_FOLDER -o OUTPUT_FOLDER -model_name MED_VAE_MODEL -modality MODALITY -ckpt_path YOUR_CKPT_PATH
```

## Troubleshooting Tips

- If you encounter a `state_dict` warning during checkpoint loading, simply wrap your checkpoint weights within a dictionary under the `'state_dict'` key.
- Creating a separate conda environment will help debugging considerably. Especially with hydra / accelerate configurations.
- Input to the VAEs will need to be normalized between \[-1, 1\]. The already provided dataloaders handle this.
- The discriminator currently starts after 3125 steps. If you want it to start earlier, you can adjust it in the main config experiment file. We set it to 3125 for our batch sizes, which was 32. Typically, the discriminator can discriminate pretty quickly, so that is why you set it to train a bit later after the model has finetuned for a bit. The discriminator is randomly initialized based on a small distribution (ln 236 in vae_losses; line 70 in loss components).
- We recommend maintaining gradient accumulation as 1 for numerical stability.
- Do not worry if the L1 loss (reconstruction) and perceptual loss are wildly different. They are on different scales, but this should not affect the backprop, since the gradient directions would stay the same.

## Support

For questions or issues regarding finetuning MedVAE models, please submit a request on our [GitHub issues page](https://github.com/StanfordMIMI/MedVAE/issues).
