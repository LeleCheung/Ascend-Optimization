"""Hardware platform abstraction for multi-vendor GPU/NPU support."""

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PlatformInfo:
    vendor_name: str
    device_name: str
    device_query_cmd: str
    device_visible_envs: list[str] = field(default_factory=list)
    triton_cache_env: str = "TRITON_CACHE_DIR"


PLATFORM_REGISTRY: dict[str, PlatformInfo] = {
    "nvidia": PlatformInfo(
        vendor_name="nvidia",
        device_name="cuda",
        device_query_cmd="nvidia-smi --query-gpu=index --format=csv,noheader",
        device_visible_envs=["CUDA_VISIBLE_DEVICES"],
    ),
    "ascend": PlatformInfo(
        vendor_name="ascend",
        device_name="npu",
        device_query_cmd="npu-smi info -l",
        device_visible_envs=["ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"],
    ),
    "cambricon": PlatformInfo(
        vendor_name="cambricon",
        device_name="mlu",
        device_query_cmd="cnmon info",
        device_visible_envs=["MLU_VISIBLE_DEVICES"],
    ),
    "mthreads": PlatformInfo(
        vendor_name="mthreads",
        device_name="musa",
        device_query_cmd="musa_smi -L",
        device_visible_envs=["MUSA_VISIBLE_DEVICES"],
    ),
    "kunlunxin": PlatformInfo(
        vendor_name="kunlunxin",
        device_name="cuda",
        device_query_cmd="xpu-smi",
        device_visible_envs=["CUDA_VISIBLE_DEVICES"],
    ),
    "hygon": PlatformInfo(
        vendor_name="hygon",
        device_name="hip",
        device_query_cmd="",
        device_visible_envs=["HIP_VISIBLE_DEVICES"],
    ),
    "metax": PlatformInfo(
        vendor_name="metax",
        device_name="maca",
        device_query_cmd="",
        device_visible_envs=["MACA_VISIBLE_DEVICES"],
    ),
    "iluvatar": PlatformInfo(
        vendor_name="iluvatar",
        device_name="cuda",
        device_query_cmd="",
        device_visible_envs=["ILUVATAR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"],
    ),
    "thead": PlatformInfo(
        vendor_name="thead",
        device_name="cuda",
        device_query_cmd="",
        device_visible_envs=["CUDA_VISIBLE_DEVICES"],
    ),
    "tsingmicro": PlatformInfo(
        vendor_name="tsingmicro",
        device_name="txda",
        device_query_cmd="",
        device_visible_envs=["TXDA_VISIBLE_DEVICES"],
    ),
    "sunrise": PlatformInfo(
        vendor_name="sunrise",
        device_name="tang",
        device_query_cmd="",
        device_visible_envs=["TANG_VISIBLE_DEVICES"],
    ),
}


def detect_platform(vendor_hint: str | None = None) -> PlatformInfo:
    if vendor_hint:
        vendor = vendor_hint.lower()
        if vendor not in PLATFORM_REGISTRY:
            raise RuntimeError(
                f"Unknown platform '{vendor}'. "
                f"Available: {', '.join(PLATFORM_REGISTRY)}"
            )
        return PLATFORM_REGISTRY[vendor]

    # Step 1: Check environment variables (highest priority)
    for env_key in ("GEMS_VENDOR", "FLAGGEMS_VENDOR", "KGRUNNER_VENDOR"):
        vendor = os.environ.get(env_key, "").lower()
        if vendor and vendor in PLATFORM_REGISTRY:
            logger.info("Platform: %s (from env %s)", vendor, env_key)
            return PLATFORM_REGISTRY[vendor]

    # Step 2: Probe system commands
    for vendor, info in PLATFORM_REGISTRY.items():
        if not info.device_query_cmd:
            continue
        cmd = info.device_query_cmd.split()[0]
        if shutil.which(cmd):
            logger.info("Platform: %s (auto-detected via %s)", vendor, cmd)
            return info

    raise RuntimeError(
        "No supported GPU/NPU platform detected. "
        "Install device tools (nvidia-smi, npu-smi, etc.) or specify vendor."
    )


def make_device_env(platform: PlatformInfo, device_id: int, base_env: dict | None = None) -> dict:
    env = dict(base_env) if base_env else os.environ.copy()
    for env_var in platform.device_visible_envs:
        env[env_var] = str(device_id)
    return env
