import sys
from pathlib import Path

import torch
from loguru import logger

from lightx2v.utils.envs import GET_DTYPE
from lightx2v_platform.base.global_var import AI_DEVICE


class WanV1LatentResizer:
    """Use a trained WanTrajectoryUpsampler checkpoint to switch latent resolution."""

    def __init__(self, upsampler, config):
        self.upsampler = upsampler
        self.config = config
        self.dtype = GET_DTYPE()
        self.device = torch.device(AI_DEVICE)

    @torch.no_grad()
    def resize(self, noisy_latent, sigma, target_latent_shape, step_index=None, changing_resolution_index=None):
        if noisy_latent.dim() != 4:
            raise ValueError(f"Expected WAN latent shape [C, T, H, W], got {tuple(noisy_latent.shape)}")

        target_latent_shape = tuple(target_latent_shape)
        current_shape = tuple(noisy_latent.shape)
        if target_latent_shape[0] != current_shape[0] or target_latent_shape[1] != current_shape[1]:
            raise ValueError(
                "Wan V1 bridge expects channel/time to stay unchanged. "
                f"Current={current_shape}, target={target_latent_shape}"
            )
        if target_latent_shape[2] != current_shape[2] * 2 or target_latent_shape[3] != current_shape[3] * 2:
            raise ValueError(
                "Wan V1 bridge currently only supports a single spatial x2 upscale. "
                f"Current={current_shape}, target={target_latent_shape}"
            )

        logger.info(
            "Wan V1 latent bridge resize: "
            f"step={step_index}, stage={changing_resolution_index}, "
            f"wan_latent={current_shape} -> {target_latent_shape}, sigma={float(sigma):.6f}"
        )

        batch = noisy_latent.unsqueeze(0).to(device=self.device, dtype=self.dtype)
        sigma_tensor = torch.tensor([float(sigma)], device=self.device, dtype=torch.float32)
        pred = self.upsampler(batch, sigma_tensor).squeeze(0)

        if tuple(pred.shape) != target_latent_shape:
            logger.warning(
                "Wan V1 bridge shape mismatch, fallback to trilinear resize: "
                f"{tuple(pred.shape)} -> {target_latent_shape}"
            )
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(0),
                size=target_latent_shape[1:],
                mode="trilinear",
            ).squeeze(0)

        return pred.to(dtype=noisy_latent.dtype, device=noisy_latent.device)
