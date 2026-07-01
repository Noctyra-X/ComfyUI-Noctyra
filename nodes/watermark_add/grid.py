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
网格水印节点（独立控制行列密度 + 随机偏移防规律裁剪）—— 图片 / 视频
"""
import logging
import random

import torch
from PIL import Image

from .._utils import (
    tensor_to_pil,
    pil_to_tensor,
    prepare_watermark_rgba,
    apply_opacity,
    resolve_frame_range,
    video_fade_opacity,
)

logger = logging.getLogger("noctyra")

# 图片/视频网格水印共用的参数提示
_GRID_TIPS = {
    "图像": "被加水印的底图(或视频帧序列)",
    "水印大小比例": "单个水印短边相对图像短边的比例(0.12=12%)，越小铺得越密",
    "不透明度": "水印不透明度，0=透明 1=不透明(满版建议偏低)",
    "水平密度": "水平方向铺设密度，越大每行水印越多",
    "垂直密度": "垂直方向铺设密度，越大每列水印越多",
    "包围盒倍数": "每个水印占的格子相对水印自身的倍数，越大间隔越疏",
    "最小包围盒比例": "格子最小尺寸下限(相对图像短边)，0=不限制",
    "最大随机偏移": "每个水印在格内随机抖动的幅度(0=整齐网格，1=最大抖动，防规律裁剪)",
    "旋转角度": "水印倾斜角度(度)，-30=常见左下斜向",
    "随机种子": "抖动随机种子，>0 固定布局；0=按图像尺寸自动派生(同尺寸同布局)",
    "反转遮罩": "把水印遮罩黑白反转",
    "水印图像": "作为水印的图(留空则原样输出)",
    "水印遮罩": "水印 alpha/形状，使背景透明",
    "起始帧": "从第几帧开始显示水印(0=第一帧)",
    "结束帧": "到第几帧停止显示，-1=直到最后一帧",
    "淡入帧数": "起始处渐显所用帧数，0=不淡入",
    "淡出帧数": "结束处渐隐所用帧数，0=不淡出",
}


def _t(name):
    """取参数提示，便于在 INPUT_TYPES 里写成 {**spec, **_t('名')}。"""
    return {"tooltip": _GRID_TIPS.get(name, "")}


def _grid_seed(user_seed, img_w, img_h):
    """种子：用户>0 直接用；否则按尺寸派生(同尺寸→同布局)。"""
    if user_seed > 0:
        return user_seed
    return ((img_w * 73856093) ^ (img_h * 19349663)) & 0xffffffffffffffff


def _resize_rotate(watermark_pil, base_size, size_ratio, angle, opacity=None):
    """按短边比例缩放→(可选)调不透明度→旋转，返回 RGBA。

    opacity=None 时保持全不透明(供视频版逐帧再调透明度)；给定时在旋转前应用
    (与图片版历史行为一致)。
    """
    target = int(base_size * size_ratio)
    wm_w, wm_h = watermark_pil.size
    ratio = min(target / wm_w, target / wm_h)
    resized = watermark_pil.resize(
        (max(1, int(wm_w * ratio)), max(1, int(wm_h * ratio))), Image.Resampling.LANCZOS
    )
    if opacity is not None:
        resized = apply_opacity(resized, opacity)
    if angle != 0:
        return resized.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    return resized


def _grid_positions(img_w, img_h, rot_w, rot_h, box_mult, min_box_ratio,
                    h_density, v_density, max_offset, base_size, rng):
    """计算网格各水印左上角坐标(含随机抖动)。图片/视频共用，保证布局一致。"""
    min_box = int(base_size * min_box_ratio)
    box_w = max(int(rot_w * box_mult), min_box)
    box_h = max(int(rot_h * box_mult), min_box)
    base_cols = max(3, int(3 * h_density))
    base_rows = max(3, int(3 * v_density))
    cols = max(2, int(base_cols * img_w / max(img_w, img_h)))
    rows = max(2, int(base_rows * img_h / max(img_w, img_h)))
    start_x = (img_w - cols * box_w) // 2
    start_y = (img_h - rows * box_h) // 2

    positions = []
    for row in range(rows):
        for col in range(cols):
            cx = start_x + col * box_w + box_w / 2
            cy = start_y + row * box_h + box_h / 2
            spare_x = (box_w - rot_w) / 2
            spare_y = (box_h - rot_h) / 2
            max_off_x = spare_x * min(max_offset, 1.0)
            max_off_y = spare_y * min(max_offset, 1.0)
            off_x = rng.uniform(-max_off_x, max_off_x) if max_off_x > 0 else 0
            off_y = rng.uniform(-max_off_y, max_off_y) if max_off_y > 0 else 0
            positions.append((int(cx - rot_w / 2 + off_x), int(cy - rot_h / 2 + off_y)))
    return positions


def _paste_grid(pil_rgba, watermark_rotated, positions, img_w, img_h):
    """把(已含目标不透明度的)旋转水印铺到各位置，越界自动跳过。返回 RGB。"""
    rw, rh = watermark_rotated.size
    for fx, fy in positions:
        if fx + rw > 0 and fx < img_w and fy + rh > 0 and fy < img_h:
            pil_rgba.paste(watermark_rotated, (fx, fy), watermark_rotated)
    return pil_rgba.convert("RGB")


class AddGridWatermark:
    """添加网格水印（独立控制行列密度）

    分别控制水平/垂直密度，包围盒基于水印实际尺寸；每个水印在格内随机偏移，
    排列不规整、更难被规律性裁剪去除。
    """

    DESCRIPTION = (
        "把水印按网格满版平铺，可分别控制水平/垂直密度；每个水印在格内随机抖动，"
        "排列不规整、更难被规律性裁剪去掉。\n"
        "与『全屏水印』区别：网格行列密度独立可调且带随机偏移，更防裁剪。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", _t("图像")),
                "水印大小比例": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 2.0, "step": 0.01, **_t("水印大小比例")}),
                "不透明度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, **_t("不透明度")}),
                "水平密度": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 5.0, "step": 0.1, **_t("水平密度")}),
                "垂直密度": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 5.0, "step": 0.1, **_t("垂直密度")}),
                "包围盒倍数": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1, **_t("包围盒倍数")}),
                "最小包围盒比例": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01, **_t("最小包围盒比例")}),
                "最大随机偏移": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, **_t("最大随机偏移")}),
                "旋转角度": ("INT", {"default": -30, "min": -360, "max": 360, **_t("旋转角度")}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, **_t("随机种子")}),
                "反转遮罩": ("BOOLEAN", {"default": False, **_t("反转遮罩")}),
            },
            "optional": {
                "水印图像": ("IMAGE", _t("水印图像")),
                "水印遮罩": ("MASK", _t("水印遮罩")),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "add_watermark"
    CATEGORY = "Noctyra/添加水印"

    def add_watermark(self, 图像, 水印大小比例=0.12, 不透明度=0.5, 水平密度=1.0, 垂直密度=1.0,
                      包围盒倍数=1.5, 最小包围盒比例=0.0, 最大随机偏移=0.8, 旋转角度=-30,
                      随机种子=0, 反转遮罩=False, 水印图像=None, 水印遮罩=None):
        if 图像 is None or len(图像) == 0:
            return (torch.zeros((0, 1, 1, 3)),)

        watermark_pil = prepare_watermark_rgba(水印图像, 水印遮罩, 反转遮罩)

        # 按分辨率分组，同分辨率共享布局
        groups = {}
        for idx, t in enumerate(图像):
            groups.setdefault((t.shape[1], t.shape[0]), []).append((idx, t))
        out = [None] * len(图像)

        for (w, h), group in groups.items():
            rng = random.Random(_grid_seed(随机种子, w, h))
            base_size = min(w, h)
            # 图片版：缩放→不透明度→旋转(与历史行为一致，逐位不变)
            wm = _resize_rotate(watermark_pil, base_size, 水印大小比例, 旋转角度, opacity=不透明度)
            positions = _grid_positions(w, h, wm.size[0], wm.size[1], 包围盒倍数,
                                        最小包围盒比例, 水平密度, 垂直密度, 最大随机偏移, base_size, rng)
            for idx, t in group:
                pil = tensor_to_pil(t).convert("RGBA")
                out[idx] = pil_to_tensor(_paste_grid(pil, wm, positions, w, h))

        for idx, item in enumerate(out):
            if item is None:
                out[idx] = 图像[idx].unsqueeze(0) if 图像[idx].dim() == 3 else 图像[idx]
        return (torch.cat(out, dim=0),)


class AddVideoGridWatermark:
    """添加网格水印（视频·帧区间+淡入淡出）

    与网格水印一致的随机网格布局；水印只在 [起始帧, 结束帧] 区间显示，可两端渐变。
    同一段视频(同分辨率)共用一套网格布局，逐帧仅改不透明度。
    """

    DESCRIPTION = (
        "给视频帧序列加随机网格水印(布局同图片网格水印)，可只在 [起始帧, 结束帧] 区间"
        "显示并两端淡入淡出。\n"
        "同一段视频(同分辨率)共用一套网格布局，逐帧只改不透明度，保证水印位置稳定不跳动。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", _t("图像")),
                "水印大小比例": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 2.0, "step": 0.01, **_t("水印大小比例")}),
                "不透明度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, **_t("不透明度")}),
                "水平密度": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 5.0, "step": 0.1, **_t("水平密度")}),
                "垂直密度": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 5.0, "step": 0.1, **_t("垂直密度")}),
                "包围盒倍数": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1, **_t("包围盒倍数")}),
                "最小包围盒比例": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01, **_t("最小包围盒比例")}),
                "最大随机偏移": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, **_t("最大随机偏移")}),
                "旋转角度": ("INT", {"default": -30, "min": -360, "max": 360, **_t("旋转角度")}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, **_t("随机种子")}),
                "起始帧": ("INT", {"default": 0, "min": 0, "max": 1000000, **_t("起始帧")}),
                "结束帧": ("INT", {"default": -1, "min": -1, "max": 1000000, **_t("结束帧")}),
                "淡入帧数": ("INT", {"default": 0, "min": 0, "max": 100000, **_t("淡入帧数")}),
                "淡出帧数": ("INT", {"default": 0, "min": 0, "max": 100000, **_t("淡出帧数")}),
                "反转遮罩": ("BOOLEAN", {"default": False, **_t("反转遮罩")}),
            },
            "optional": {
                "水印图像": ("IMAGE", _t("水印图像")),
                "水印遮罩": ("MASK", _t("水印遮罩")),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "add_watermark"
    CATEGORY = "Noctyra/添加水印"

    def add_watermark(self, 图像, 水印大小比例=0.12, 不透明度=0.5, 水平密度=1.0, 垂直密度=1.0,
                      包围盒倍数=1.5, 最小包围盒比例=0.0, 最大随机偏移=0.8, 旋转角度=-30,
                      随机种子=0, 起始帧=0, 结束帧=-1, 淡入帧数=0, 淡出帧数=0,
                      反转遮罩=False, 水印图像=None, 水印遮罩=None):
        if 图像 is None or len(图像) == 0:
            return (torch.zeros((0, 1, 1, 3)),)

        n = len(图像)
        start, end = resolve_frame_range(n, 起始帧, 结束帧)
        watermark_pil = prepare_watermark_rgba(水印图像, 水印遮罩, 反转遮罩)

        # 布局只按分辨率算一次（缓存旋转后的全不透明水印 + 位置）
        layout = {}

        def _layout_for(w, h):
            if (w, h) not in layout:
                rng = random.Random(_grid_seed(随机种子, w, h))
                base = min(w, h)
                rot = _resize_rotate(watermark_pil, base, 水印大小比例, 旋转角度)
                pos = _grid_positions(w, h, rot.size[0], rot.size[1], 包围盒倍数,
                                      最小包围盒比例, 水平密度, 垂直密度, 最大随机偏移, base, rng)
                layout[(w, h)] = (rot, pos)
            return layout[(w, h)]

        out = []
        for i, t in enumerate(图像):
            opacity = video_fade_opacity(i, start, end, 淡入帧数, 淡出帧数, 不透明度)
            if opacity <= 0.0:
                out.append(t.unsqueeze(0) if t.dim() == 3 else t)
                continue
            h, w = t.shape[0], t.shape[1]
            rot_full, positions = _layout_for(w, h)
            wm = apply_opacity(rot_full, opacity)
            pil = tensor_to_pil(t).convert("RGBA")
            out.append(pil_to_tensor(_paste_grid(pil, wm, positions, w, h)))
        return (torch.cat(out, dim=0),)


NODE_CLASS_MAPPINGS = {
    "AddGridWatermark": AddGridWatermark,
    "AddVideoGridWatermark": AddVideoGridWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddGridWatermark": "图片添加网格水印（独立行列密度）",
    "AddVideoGridWatermark": "视频添加网格水印（独立行列密度）",
}
