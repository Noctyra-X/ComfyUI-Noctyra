# ComfyUI-Noctyra
# Copyright (C) 2026 Noctyra
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ---------------------------------------------------------------------------
# 本节点封装 vendored MIT 项目 remove-ai-watermarks 的隐形水印去除引擎
# (Copyright (c) 2025 wiltodelta；ctrlregen 基于 yepengliu/CtrlRegen)。
# 详见仓库 THIRD_PARTY_NOTICES.md。
# ---------------------------------------------------------------------------

"""
去除隐形水印（封装 vendored 引擎，支持 default / ctrlregen 两条管线）

- default  : SDXL 低强度 img2img 重生成，打乱 SynthID/StableSignature/TreeRing。
- ctrlregen: CtrlRegen 受控全量重生成(空间ControlNet+语义IP-Adapter)，更彻底。
可选 文字保护 / 人脸保护 / 胶片颗粒(humanize)。模型全部读插件自带 models/ 下
用『隐形水印模型下载器』下好的本地目录(离线)。
"""
import logging
import os
import sys
import tempfile
import time
import types

import numpy as np
import torch
from PIL import Image

from .._vendor import ensure as _ensure_vendor
from .._utils import plugin_models_dir

logger = logging.getLogger("noctyra")

# 本地模型目录名(与下载器统一命名一致)
DIR_SDXL = "stabilityai__stable-diffusion-xl-base-1.0"
DIR_RV = "SG161222__Realistic_Vision_V4.0_noVAE"
DIR_CTRL = "yepengliu__ctrlregen"
DIR_DINOV2 = "facebook__dinov2-giant"
DIR_VAE = "stabilityai__sd-vae-ft-mse"

_ENGINE_CACHE = {}


def _local(dirname):
    return os.path.join(plugin_models_dir(), dirname)


def _pick_device(choice):
    if choice in ("cuda", "cpu"):
        return choice
    return "cuda" if torch.cuda.is_available() else "cpu"


def _stub_mediapipe_face():
    """非侵入式绕过 controlnet_aux 的 mediapipe_face(本环境 mediapipe 不兼容)。

    ctrlregen 仅用到 controlnet_aux.CannyDetector，与 mediapipe 无关。在导入
    controlnet_aux 之前把会触发 mediapipe 的子模块替换成桩，不改动已装的包。
    """
    if "controlnet_aux.mediapipe_face" not in sys.modules:
        stub = types.ModuleType("controlnet_aux.mediapipe_face")

        class _D:
            def __init__(self, *a, **k):
                pass

        stub.MediapipeFaceDetector = _D
        sys.modules["controlnet_aux.mediapipe_face"] = stub


def _wire_ctrlregen_local():
    """把 vendored ctrlregen 引擎里硬编码的 HF 仓库 id 改指本地目录(离线)。"""
    from remove_ai_watermarks.noai.ctrlregen import engine as ce
    from remove_ai_watermarks.noai.ctrlregen import ip_adapter as ia
    ce.CTRLREGEN_HF_REPO = _local(DIR_CTRL)
    ce.DEFAULT_BASE_MODEL = _local(DIR_RV)
    ce.CUSTOM_VAE_ID = _local(DIR_VAE)
    ce.DINOV2_MODEL_ID = _local(DIR_DINOV2)
    ia.DINOV2_MODEL_ID = _local(DIR_DINOV2)


def _get_engine(pipeline, device):
    key = (pipeline, device)
    if key in _ENGINE_CACHE:
        return _ENGINE_CACHE[key]

    _ensure_vendor()
    from remove_ai_watermarks.invisible_engine import InvisibleEngine

    if pipeline == "ctrlregen":
        _stub_mediapipe_face()
        _wire_ctrlregen_local()
        # model_id=ctrlregen 哨兵 -> 走 CtrlRegenEngine(base/vae/dino 已改本地)
        engine = InvisibleEngine(model_id="yepengliu/ctrlregen", pipeline="ctrlregen", device=device)
    else:
        engine = InvisibleEngine(model_id=_local(DIR_SDXL), pipeline="default", device=device)

    _ENGINE_CACHE[key] = engine
    return engine


def _tensor_to_pil(img_tensor):
    arr = np.clip(img_tensor.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr, mode="RGB")


class RemoveInvisibleWatermark:
    """去除隐形水印（SDXL / CtrlRegen）

    用本地扩散模型重生成图像以破坏 SynthID 等不可见水印。default 管线改动极小、
    最保真；ctrlregen 受控全量重生成、更彻底。可选文字/人脸保护与胶片颗粒。
    """

    DESCRIPTION = (
        "去除 SynthID / StableSignature 等『隐形水印』：用扩散模型低强度重生成像素打乱不可见水印。\n"
        "模型需先用『隐形水印模型下载器』下到插件 models/ 目录。需要 GPU。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "疑似含 SynthID 等隐形水印的图像"}),
                "管线": (["default (SDXL)", "ctrlregen"], {"default": "default (SDXL)",
                    "tooltip": "default=SDXL 轻量 img2img(最保真,1个模型)；ctrlregen=受控全量重生成(更彻底,需4个模型)"}),
                "重绘强度": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "img2img 去噪强度。0.05=只动一点点(画面几乎不变即可破坏水印);调大改动越多越可能走样"}),
                "采样步数": ("INT", {"default": 50, "min": 1, "max": 150, "tooltip": "去噪步数，越多越精细越慢"}),
                "seed": ("INT", {"default": 0, "min": -1, "max": 2**31 - 1, "tooltip": "随机种子，-1=每次随机"}),
                "设备": (["自动", "cuda", "cpu"], {"default": "自动", "tooltip": "推理设备。CPU 极慢，建议 cuda"}),
                "文字保护": ("BOOLEAN", {"default": False, "tooltip": "检测文字区并用差分扩散保护，避免文字/CJK 被改坏(首次用下小模型)"}),
                "人脸保护": ("BOOLEAN", {"default": False, "tooltip": "提取人脸、重生成后再贴回，保人脸不变(用 YOLO，首次下小模型)"}),
                "胶片颗粒": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.5,
                    "tooltip": "可选的胶片颗粒/拟真后期强度，0=关。典型 2~6"}),
                "最大边长": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 64,
                    "tooltip": "处理前把长边缩到此像素以省显存，0=原生分辨率(质量最好)。大图 OOM 时才设"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "remove"
    CATEGORY = "Noctyra/水印去除"

    def remove(self, 图像, 管线, 重绘强度, 采样步数, seed, 设备,
               文字保护, 人脸保护, 胶片颗粒, 最大边长):
        try:
            import diffusers  # noqa: F401
        except ImportError:
            raise ImportError("未安装 diffusers，请先 pip install diffusers accelerate")

        if 图像 is None or len(图像) == 0:
            return (torch.zeros((0, 1, 1, 3)),)

        pipeline = "ctrlregen" if 管线.startswith("ctrlregen") else "default"
        device = _pick_device(设备)
        from pathlib import Path
        seed_arg = int(seed) if (seed is not None and seed >= 0) else None

        # 仅在本节点执行期间启用 HF 离线，避免影响进程内其它需要联网的节点
        _prev_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            engine = _get_engine(pipeline, device)
            out = []
            for idx, img_tensor in enumerate(图像):
                pil = _tensor_to_pil(img_tensor)
                stamp = f"{os.getpid()}_{int(time.time()*1000)}_{idx}"
                tmp_in = os.path.join(tempfile.gettempdir(), f"noctyra_in_{stamp}.png")
                tmp_out = os.path.join(tempfile.gettempdir(), f"noctyra_out_{stamp}.png")
                pil.save(tmp_in)
                try:
                    res_path = engine.remove_watermark(
                        image_path=Path(tmp_in),
                        output_path=Path(tmp_out),
                        strength=float(重绘强度),
                        num_inference_steps=int(采样步数),
                        seed=seed_arg,
                        humanize=float(胶片颗粒),
                        protect_faces=人脸保护,
                        protect_text=文字保护,
                        max_resolution=int(最大边长),
                    )
                    result = Image.open(res_path).convert("RGB")
                    if result.size != pil.size:
                        result = result.resize(pil.size, Image.Resampling.LANCZOS)
                    out.append(torch.from_numpy(np.array(result).astype(np.float32) / 255.0).unsqueeze(0))
                finally:
                    for p in (tmp_in, tmp_out):
                        try:
                            if os.path.exists(p):
                                os.remove(p)
                        except Exception:
                            pass
        finally:
            if _prev_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = _prev_offline

        return (torch.cat(out, dim=0),)


NODE_CLASS_MAPPINGS = {
    "RemoveInvisibleWatermark": RemoveInvisibleWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveInvisibleWatermark": "去除隐形水印（SDXL/CtrlRegen）",
}
