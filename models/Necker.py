import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Necker(nn.Module):

    def __init__(self):
        super(Necker, self).__init__()

    @torch.no_grad()
    def forward(self, tokens):
        reshaped_tokens = []
        for token in tokens:
            if len(token.shape) == 3:
                B, N, C = token.shape
                side_length = int(math.sqrt(N - 1))
                token = token[:, 1:, :].view(B, side_length, side_length, C).permute(0, 3, 1, 2)
            reshaped_tokens.append(token)

        target_size = tuple(reshaped_tokens[0].shape[2:])

        align_features = []
        for token in reshaped_tokens:
            align_features.append(F.interpolate(token, size=target_size, mode='bilinear', align_corners=True))
            
        return align_features