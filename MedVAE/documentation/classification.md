# ✨ MedVAE classification documentation

MedVAE includes a flexible classification framework designed to leverage latent representations for downstream classification tasks. The current setup provides examples that you can adapt to your dataset. Please follow the instructions below to configure and run classification models using MedVAE.

## Requirements

- Python 3.9 (Recommended)
- Clone of the MedVAE GitHub repository

## Preparing Your Dataset

The provided example dataset does not include labels or classification examples. You will need to:

- Prepare a CSV file that clearly specifies your dataset splits (train, validation, test) and corresponding labels.
- Adjust the [dataloader configuration](../configs/dataloader/example_dataset.yaml) to reference your CSV file and dataset path.

## Configuration and Customization

The classification framework requires modifications in several key configuration areas:

### Dataloader

- Customize the [example dataloader file](../configs/dataloader/example_dataset.yaml) to match your dataset structure and labeling schema.
- Ensure the CSV file accurately includes columns for data splits and class labels.

### Criterion

- The default loss function is set for binary cross-entropy. To use alternative loss functions (e.g., multi-class cross-entropy), you must update the criterion configuration accordingly.

### Model

- The default [model configuration](../configs/model/monai_seresnet152.yaml) provided in [example_cls.yaml](../configs/experiment/example_cls.yaml) is designed for 3D volume classification.
- For 2D image classification tasks, please adapt this configuration using [default.yaml](../configs/model/default.yaml) or another suitable architecture.
- Identify and select model architectures that best suit your specific dataset and task.

### Experiment

- Modify the experiment configuration file to adjust hyperparameters such as batch size, learning rate, and epochs tailored to your dataset and task.

## Running Classification

Use the following command to start the classification process with your configured experiment:

```bash
CUDA_VISIBLE_DEVICES=0 medvae_classify experiment=example_cls
```

### Multi-GPU Training

The classification framework supports multi-GPU training using HuggingFace Accelerate. Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 medvae_classify experiment=example_cls
```

## Support

For questions or support regarding classification tasks using MedVAE, please submit an issue on our [GitHub issues page](https://github.com/StanfordMIMI/MedVAE/issues).
