import sys
from pathlib import Path

import torch
from loguru import logger

from lightx2v.models.runners.wan.wan_runner import WanRunner
from lightx2v.models.schedulers.wan.feature_caching.scheduler import (
    WanSchedulerCaching,
    WanSchedulerTaylorCaching,
)
from lightx2v.models.schedulers.wan.scheduler import WanScheduler
from lightx2v.models.schedulers.wan.wanupsampler_bridge_resolution.scheduler import (
    WanScheduler4WanUpsamplerBridgeInterface,
)
from lightx2v.models.video_encoders.hf.wan.upsampler.wan_v1_latent_bridge import (
    WanV1LatentResizer,
)
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("wan2.1_wanupsampler_bridge")
class WanWanUpsamplerBridgeRunner(WanRunner):
    """WAN changing-resolution runner backed by a trained WanTrajectoryUpsampler."""

    def __init__(self, config):
        super().__init__(config)
        self._validate_bridge_config()

    def _validate_bridge_config(self):
        if self.config["task"] != "t2v":
            raise NotImplementedError("wan2.1_wanupsampler_bridge currently only supports t2v.")
        if self.config.get("use_tae", False):
            raise ValueError("wan2.1_wanupsampler_bridge requires the full WAN VAE encoder/decoder, not TAE.")
        if self.config.get("lazy_load", False) or self.config.get("unload_modules", False):
            raise ValueError("wan2.1_wanupsampler_bridge does not support lazy_load/unload_modules yet.")

        resolution_rate = list(self.config.get("resolution_rate", []))
        if resolution_rate != [0.5]:
            raise ValueError(
                "wan2.1_wanupsampler_bridge expects a single lowres->fullres stage with resolution_rate=[0.5]. "
                f"Got {resolution_rate}"
            )
        if len(self.config.get("changing_resolution_steps", [])) != 1:
            raise ValueError("wan2.1_wanupsampler_bridge expects exactly one changing_resolution step.")
        if self.config["target_height"] % 64 != 0 or self.config["target_width"] % 64 != 0:
            raise ValueError(
                "wan2.1_wanupsampler_bridge expects final target height/width to be divisible by 64."
            )
        if self.config.get("wanupsampler_ckpt") is None:
            raise ValueError("wan2.1_wanupsampler_bridge requires wanupsampler_ckpt in config.")
        if self.config.get("wanupsampler_repo") is None:
            raise ValueError("wan2.1_wanupsampler_bridge requires wanupsampler_repo in config.")

    def init_scheduler(self):
        if self.config["feature_caching"] == "NoCaching":
            scheduler_class = WanScheduler
        elif self.config["feature_caching"] == "TaylorSeer":
            scheduler_class = WanSchedulerTaylorCaching
        elif self.config.feature_caching in ["Tea", "Ada", "Custom", "FirstBlock", "DualBlock", "DynamicBlock", "Mag"]:
            scheduler_class = WanSchedulerCaching
        else:
            raise NotImplementedError(f"Unsupported feature_caching type: {self.config.feature_caching}")

        self.scheduler = WanScheduler4WanUpsamplerBridgeInterface(scheduler_class, self.config)

    def load_wanupsampler(self):
        repo_path = Path(self.config["wanupsampler_repo"])
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        from wan_sr.models import WanNoisyLatentUpsampler
        from wan_sr.training.checkpoint import load_checkpoint
        from wan_sr.training.config import load_yaml

        ckpt_path = self.config["wanupsampler_ckpt"]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model_config = checkpoint.get("config", {}).get("model", {})
        if not model_config and self.config.get("wanupsampler_train_config"):
            train_cfg = load_yaml(self.config["wanupsampler_train_config"])
            model_config = train_cfg.get("model", {})
        if not model_config:
            raise ValueError("Failed to infer wanupsampler model config from checkpoint or wanupsampler_train_config.")

        model = WanNoisyLatentUpsampler(**model_config)
        load_checkpoint(ckpt_path, model, map_location=torch.device(AI_DEVICE))
        if self.config.get("wanupsampler_use_ema", True) and "ema" in checkpoint:
            from wan_sr.training.ema import EMA

            ema = EMA(model)
            ema.load_state_dict(checkpoint["ema"])
            ema.copy_to(model)

        model = model.to(device=torch.device(AI_DEVICE), dtype=GET_DTYPE())
        model.eval()
        logger.info(f"Initialized Wan V1 upsampler from {ckpt_path}")
        return model

    def load_model(self):
        super().load_model()
        self.wanupsampler = self.load_wanupsampler()
        self.clean_latent_resizer = WanV1LatentResizer(
            upsampler=self.wanupsampler,
            config=self.config,
        )
        self.scheduler.set_clean_latent_resizer(self.clean_latent_resizer)
        logger.info("Initialized WAN + WanTrajectoryUpsampler bridge resizer.")

    def init_run(self):
        super().init_run()
        if hasattr(self.scheduler, "set_clean_latent_resizer"):
            self.scheduler.set_clean_latent_resizer(self.clean_latent_resizer)
