import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 256, output_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)