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

"""
Noctyra 节点公共工具
"""
import os

import numpy as np
import torch
from PIL import Image, ImageDraw

# 插件根目录（nodes/_utils.py 上溯一级）。各模块统一引用，避免子目录后 __file__ 算错。
PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plugin_models_dir():
    """插件自带模型目录 ComfyUI-Noctyra/models/(去水印的扩散模型放这里)。"""
    return os.path.join(PLUGIN_ROOT, "models")


def plugin_fonts_dir():
    """插件自带字体目录 ComfyUI-Noctyra/fonts/。"""
    return os.path.join(PLUGIN_ROOT, "fonts")


def tensor_to_pil(tensor):
    """单张 IMAGE 张量 -> PIL 图像（仅去 batch 维 / 单通道维，不用裸 squeeze 以免 1px 宽高坍塌）"""
    t = tensor
    if t.dim() == 4:          # [1,H,W,C] -> [H,W,C]
        t = t[0]
    arr = np.clip(255.0 * t.cpu().numpy(), 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:   # 单通道 -> 灰度 [H,W]
        arr = arr[:, :, 0]
    return Image.fromarray(arr)


def pil_to_tensor(pil_image):
    """PIL 图像 -> ComfyUI IMAGE 张量（[1,H,W,C]）"""
    return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor_to_rgb_uint8(img_tensor):
    """单张 IMAGE 张量([H,W,C] 0..1) -> 连续 RGB uint8 ndarray（灰度补 3 通道、去 alpha）。

    供 cv2 系去水印节点共用（contiguous 以便 cv2 就地操作）。
    """
    arr = np.clip(img_tensor.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def rgb_uint8_to_tensor(arr):
    """RGB uint8 ndarray -> ComfyUI IMAGE 张量 [1,H,W,C]"""
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)


def safe_print(text):
    """打印到控制台，自动兼容 GBK 等不支持部分字符的终端编码。"""
    try:
        print(text)
    except Exception:
        import sys
        enc = sys.stdout.encoding or "utf-8"
        print(text.encode(enc, "replace").decode(enc))


def resolve_frame_range(n, start_frame, end_frame):
    """把(帧数, 起始帧, 结束帧)归一为有效 (start, end)；结束帧<0 表示最后一帧。"""
    start = max(0, start_frame)
    end = (n - 1) if end_frame < 0 else min(end_frame, n - 1)
    return start, end


def video_fade_opacity(i, start, end, fade_in, fade_out, base_opacity):
    """第 i 帧的水印不透明度：区间外为 0，区间两端按淡入/淡出帧数线性渐变。"""
    if i < start or i > end:
        return 0.0
    factor = 1.0
    if fade_in > 0:
        factor = min(factor, (i - start + 1) / fade_in)
    if fade_out > 0:
        factor = min(factor, (end - i + 1) / fade_out)
    return base_opacity * max(0.0, min(1.0, factor))


def mask_to_pil_l(mask_tensor, target_size):
    """MASK 张量 -> 与 target_size 匹配的 L 模式 PIL 图像"""
    m = mask_tensor
    if m.dim() == 3:
        m = m[0]
    arr = np.clip(m.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    mask_pil = Image.fromarray(arr, mode="L")
    if mask_pil.size != target_size:
        mask_pil = mask_pil.resize(target_size, Image.Resampling.LANCZOS)
    return mask_pil


def create_default_watermark():
    """占位水印（仅在用户未提供水印图像时使用）"""
    watermark_pil = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark_pil)
    draw.rectangle([0, 0, 100, 100], fill=(0, 0, 0, 100))
    draw.text((20, 40), "Watermark", fill=(255, 255, 255, 200))
    return watermark_pil


def prepare_watermark_rgba(watermark_image, watermark_mask, invert_mask):
    """
    准备 RGBA 水印图像。

    - 未提供 watermark_image：返回占位水印
    - 提供 watermark_image：转 RGBA；若同时提供 mask，则替换 alpha 通道
    - 提供但加载/处理失败：抛异常（避免静默用占位水印覆盖用户成品图）
    """
    if watermark_image is None:
        return create_default_watermark()

    watermark_pil = tensor_to_pil(watermark_image[0]).convert("RGBA")

    if watermark_mask is not None:
        mask_t = watermark_mask[0] if watermark_mask.dim() == 3 else watermark_mask
        mask_l = mask_to_pil_l(mask_t, watermark_pil.size)
        if invert_mask:
            mask_l = mask_l.point(lambda p: 255 - p)
        r, g, b, _ = watermark_pil.split()
        watermark_pil = Image.merge("RGBA", (r, g, b, mask_l))

    return watermark_pil


def apply_opacity(watermark_rgba, opacity):
    """把不透明度系数乘到 RGBA 水印的 alpha 通道（opacity >= 1.0 时直接返回原图）"""
    if opacity >= 1.0:
        return watermark_rgba
    r, g, b, a = watermark_rgba.split()
    a = a.point(lambda p: int(p * opacity))
    return Image.merge("RGBA", (r, g, b, a))


def apply_watermark_frame(
    pil_rgba, watermark_pil, *, size_ratio, opacity, position,
    margin_x, margin_y, fs_angle=-30, fs_density=1.0,
):
    """在单帧上贴水印（图片/视频节点共用）。

    pil_rgba: 已转 RGBA 的底图；watermark_pil: 原始 RGBA 水印(未缩放未调透明度)。
    按 size_ratio 等比缩放、按 opacity 调透明度后，依 position 贴到角落/居中/全屏。
    返回 RGB 模式 PIL 图（就地修改传入的 pil_rgba）。
    """
    import logging
    logger = logging.getLogger("noctyra")

    img_width, img_height = pil_rgba.size
    base_size = min(img_width, img_height)
    target_size = int(base_size * size_ratio)
    wm_w, wm_h = watermark_pil.size
    ratio = min(target_size / wm_w, target_size / wm_h)
    resized = watermark_pil.resize(
        (max(1, int(wm_w * ratio)), max(1, int(wm_h * ratio))), Image.Resampling.LANCZOS
    )
    resized = apply_opacity(resized, opacity)
    wm_w, wm_h = resized.size

    if position == "全屏":
        try:
            rot_img = resized.rotate(fs_angle, expand=True, resample=Image.Resampling.BICUBIC)
            r_w, r_h = rot_img.size
            sx = max(1, int((r_w + margin_x) / fs_density))
            sy = max(1, int((r_h + margin_y) / fs_density))
            offset = sx // 2
            row_idx = 0
            for y in range(-r_h, img_height, sy):
                start_x = -r_w + (offset if (row_idx % 2) else 0)
                for x in range(start_x, img_width, sx):
                    pil_rgba.paste(rot_img, (x, y), rot_img)
                row_idx += 1
        except Exception as e:
            logger.error(f"全屏水印处理失败: {e}")
    else:
        x_pos = (
            margin_x if "左" in position
            else (img_width - wm_w - margin_x if "右" in position else (img_width - wm_w) // 2)
        )
        y_pos = (
            margin_y if "上" in position
            else (img_height - wm_h - margin_y if "下" in position else (img_height - wm_h) // 2)
        )
        x_pos = max(0, min(x_pos, img_width - 1))
        y_pos = max(0, min(y_pos, img_height - 1))
        pil_rgba.paste(resized, (x_pos, y_pos), resized)

    return pil_rgba.convert("RGB")
