import torch

from medvae.utils.vae.diffusionmodels_3d import Decoder, Encoder
from medvae.utils.vae.distributions import DiagonalGaussianDistribution
from torch.utils.checkpoint import checkpoint


class AutoencoderKL(torch.nn.Module):
    def __init__(
        self, ddconfig, embed_dim, ckpt_path=None, ignore_keys=[], apply_channel_ds=True
    ):
        super().__init__()
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv3d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        self.apply_channel_ds = apply_channel_ds
        if self.apply_channel_ds:
            self.channel_ds = torch.nn.Sequential(
                torch.nn.Conv3d(self.embed_dim, 64, 1),
                torch.nn.ReLU(),
                torch.nn.Conv3d(64, 64, 3, padding="same"),
                torch.nn.ReLU(),
                torch.nn.Conv3d(64, self.embed_dim, 1),
            )
            self.channel_proj = torch.nn.Conv3d(self.embed_dim, self.embed_dim, 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, state_dict=True)

    def init_from_ckpt(self, path, ignore_keys=list(), state_dict=False):
        if not state_dict:
            sd = torch.load(path, map_location="cpu")
        else:
            sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]

        model_state_dict = self.state_dict()
        missing = []
        unexpected = []
        for name, param in sd.items():
            if name in model_state_dict:
                if param.size() == model_state_dict[name].size():
                    model_state_dict[name].copy_(param)
                else:
                    # If the dimensions of the checkpoint param is one less than model param, then copy the checkpoint param across
                    # middle dimension of model param
                    if (
                        len(param.size()) == len(model_state_dict[name].size()) - 1
                        and len(model_state_dict[name].size()) == 5
                    ):
                        ckpt_param = param.unsqueeze(4)
                        ckpt_param_repeat = ckpt_param.repeat(
                            1, 1, 1, 1, model_state_dict[name].size(4)
                        )
                        # Only save the weight to the middle kernel slice
                        ckpt_param_repeat = torch.zeros_like(ckpt_param_repeat)
                        ckpt_param_repeat[
                            :, :, :, :, model_state_dict[name].size(4) // 2
                        ] = param

                        model_state_dict[name].copy_(ckpt_param_repeat)
                    else:
                        missing.append(name)
            else:
                unexpected.append(name)

        # Load state
        self.load_state_dict(model_state_dict, strict=False)

        # missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")
        elif len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def compute_latent_proj(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        if self.apply_channel_ds:
            return z, posterior, self.channel_proj(self.channel_ds(z) + z)

        return z, posterior, None

    def forward(self, input, sample_posterior=True, decode=True):
        posterior = checkpoint(self.encode, input, use_reentrant=False)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        latent = checkpoint(
            self.channel_proj, self.channel_ds(z) + z, use_reentrant=False
        )
        if decode:
            dec = checkpoint(self.decode, z, use_reentrant=False)
            return dec, posterior, z
        else:
            return z, posterior, latent

    def get_last_layer(self):
        return self.decoder.conv_out.weight
