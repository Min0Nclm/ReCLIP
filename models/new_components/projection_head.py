import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int = 768):
        super().__init__()
        # Project the channel dimension of the VAE latent to the dimension expected by the Adapter
        self.projection = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def forward(self, x):
        return self.projection(x)
