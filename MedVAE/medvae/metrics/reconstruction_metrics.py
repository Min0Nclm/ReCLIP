import torch
from torchmetrics import (
    Metric,
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)
from torchmetrics.image.fid import FrechetInceptionDistance

from monai.transforms import SpatialPad


class MSE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, images, reconstructions):
        assert images.shape == reconstructions.shape
        images = images.contiguous()
        reconstructions = reconstructions.contiguous()

        images = images.view(images.shape[0], -1)
        reconstructions = reconstructions.view(reconstructions.shape[0], -1)
        err = ((images - reconstructions) ** 2).mean(-1)

        self.error += err.sum()
        self.total += images.shape[0]

    def compute(self):
        return self.error.float() / self.total


class PSNR(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("psnr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.func = PeakSignalNoiseRatio(data_range=1.0, reduction=None, dim=1)

    def update(self, images, reconstructions):
        assert images.shape == reconstructions.shape
        images = images.contiguous()
        reconstructions = reconstructions.contiguous()

        # Undo normalization and transform images to 0 to 1 range
        images = images.view(images.shape[0], -1)
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)

        # Undo normalization and transform reconstructions to 0 to 1 range
        reconstructions = reconstructions.view(reconstructions.shape[0], -1)
        reconstructions = torch.clamp((reconstructions + 1.0) / 2.0, min=0.0, max=1.0)

        # Compute PSNR
        psnr = self.func(reconstructions, images).sum()

        self.psnr += psnr
        self.total += images.shape[0]

    def compute(self):
        return self.psnr.float() / self.total


class MS_SSIM(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("ms_ssim", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.func = MultiScaleStructuralSimilarityIndexMeasure(
            reduction="none", data_range=1.0
        )

    def update(self, images, reconstructions):
        assert images.shape == reconstructions.shape

        # Undo normalization and transform reconstructions to 0 to 1 range
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)

        # Undo normalization and transform reconstructions to 0 to 1 range
        reconstructions = torch.clamp((reconstructions + 1.0) / 2.0, min=0.0, max=1.0)

        # Compute MS-SSIM
        ms_ssim = self.func(reconstructions, images).sum()

        self.ms_ssim += ms_ssim
        self.total += images.shape[0]

    def compute(self):
        return self.ms_ssim.float() / self.total


class MS_SSIM_SMALL(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("ms_ssim", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.func = MultiScaleStructuralSimilarityIndexMeasure(
            reduction="none", data_range=1.0
        )

    def update(self, images, reconstructions):
        assert images.shape == reconstructions.shape

        # Undo normalization and transform reconstructions to 0 to 1 range
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images = SpatialPad(spatial_size=(1, 192, 192, 192))(images).as_tensor()

        # Undo normalization and transform reconstructions to 0 to 1 range
        reconstructions = torch.clamp((reconstructions + 1.0) / 2.0, min=0.0, max=1.0)
        reconstructions = SpatialPad(spatial_size=(1, 192, 192, 192))(
            reconstructions
        ).as_tensor()

        # Compute MS-SSIM
        ms_ssim = self.func(reconstructions, images).sum()

        self.ms_ssim += ms_ssim
        self.total += images.shape[0]

    def compute(self):
        return self.ms_ssim.float() / self.total


class FID_Inception(Metric):
    def __init__(self):
        super().__init__()
        self.func = FrechetInceptionDistance(normalize=True)

    def update(self, images, reconstructions):
        assert images.shape == reconstructions.shape

        # Undo normalization and transform reconstructions to 0 to 1 range
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).expand(
            -1, 3, -1, -1
        )

        # Undo normalization and transform reconstructions to 0 to 1 range
        reconstructions = torch.clamp(
            (reconstructions + 1.0) / 2.0, min=0.0, max=1.0
        ).expand(-1, 3, -1, -1)

        # Compute FID
        self.func.update(images, real=True)
        self.func.update(reconstructions, real=False)

    def compute(self):
        return self.func.compute()


class FID_Inception_3D(Metric):
    def __init__(self):
        super().__init__()
        self.func = FID_Inception()

    def update(self, images, reconstructions):
        assert images.shape == reconstructions.shape

        for dim_idx, dim_name in enumerate(["depth", "height", "width"], start=2):
            # Iterate over slices along the current dimension
            for j in range(images.size(dim_idx)):
                # Select the appropriate slice along each dimension
                if dim_name == "depth":
                    slice_i = images[:, :, j, :, :]
                    recon_i = reconstructions[:, :, j, :, :]
                elif dim_name == "height":
                    slice_i = images[:, :, :, j, :]
                    recon_i = reconstructions[:, :, :, j, :]
                else:  # dim_name == 'width'
                    slice_i = images[:, :, :, :, j]
                    recon_i = reconstructions[:, :, :, :, j]

                # Compute FID
                self.func.update(slice_i, recon_i)

    def compute(self):
        return self.func.compute()
