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
全屏(平铺)水印节点 —— 图片 / 视频

从图片水印、视频水印节点中抽出的『全屏』模式：把水印旋转后斜向平铺铺满整图。
图片版处理单/多帧；视频版另带『帧区间 + 淡入淡出』。
（另有 watermark_grid 的网格水印，是带随机偏移的网格平铺，风格不同，二者并存。）
"""
import logging

import torch

from .._utils import (
    tensor_to_pil,
    pil_to_tensor,
    prepare_watermark_rgba,
    apply_watermark_frame,
    resolve_frame_range,
    video_fade_opacity,
)

logger = logging.getLogger("noctyra")


def _tile_frame(img_tensor, watermark_pil, size_ratio, opacity, angle, sx, sy, density):
    """对单帧做全屏平铺水印，返回 [1,H,W,3] 张量。"""
    pil = tensor_to_pil(img_tensor).convert("RGBA")
    result = apply_watermark_frame(
        pil, watermark_pil,
        size_ratio=size_ratio, opacity=opacity, position="全屏",
        margin_x=sx, margin_y=sy, fs_angle=angle, fs_density=density,
    )
    return pil_to_tensor(result)


class AddFullscreenWatermark:
    """全屏水印（图片）

    把水印旋转后斜向平铺铺满整图，适合防盗用的满版水印。
    """

    DESCRIPTION = (
        "把水印旋转后斜向平铺铺满整图，做防盗用的满版水印。处理单帧或多帧。\n"
        "与『网格水印』区别：全屏是规则斜向密铺；网格可分别控制行列密度且带随机抖动。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "被加水印的底图"}),
                "水印大小比例": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 2.0, "step": 0.01,
                    "tooltip": "单个水印宽度相对底图宽度的比例。越小铺得越密"}),
                "不透明度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "水印不透明度，满版建议偏低(0.3~0.5)"}),
                "旋转角度": ("INT", {"default": -30, "min": -360, "max": 360, "tooltip": "水印倾斜角度(度)，-30=常见的左下斜向"}),
                "水平间距": ("INT", {"default": 10, "min": 0, "max": 4096, "tooltip": "平铺时相邻水印的水平间距(像素)"}),
                "垂直间距": ("INT", {"default": 10, "min": 0, "max": 4096, "tooltip": "平铺时相邻水印的垂直间距(像素)"}),
                "密度": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1, "tooltip": "整体铺设密度倍率，越大越密"}),
                "反转遮罩": ("BOOLEAN", {"default": False, "tooltip": "把水印遮罩黑白反转"}),
            },
            "optional": {
                "水印图像": ("IMAGE", {"tooltip": "作为水印的图(留空则原样输出)"}),
                "水印遮罩": ("MASK", {"tooltip": "水印 alpha/形状，使背景透明"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "add_watermark"
    CATEGORY = "Noctyra/添加水印"

    def add_watermark(self, 图像, 水印大小比例=0.15, 不透明度=0.5, 旋转角度=-30,
                      水平间距=10, 垂直间距=10, 密度=1.0, 反转遮罩=False,
                      水印图像=None, 水印遮罩=None):
        if 图像 is None or len(图像) == 0:
            return (torch.zeros((0, 1, 1, 3)),)
        watermark_pil = prepare_watermark_rgba(水印图像, 水印遮罩, 反转遮罩)
        out = [
            _tile_frame(t, watermark_pil, 水印大小比例, 不透明度, 旋转角度, 水平间距, 垂直间距, 密度)
            for t in 图像
        ]
        return (torch.cat(out, dim=0),)


class AddVideoFullscreenWatermark:
    """全屏水印（视频）

    给视频帧序列铺满版水印，支持只在 [起始帧, 结束帧] 区间显示并做淡入淡出。
    """

    DESCRIPTION = (
        "给视频帧序列铺满版水印，可只在 [起始帧, 结束帧] 区间显示并做淡入淡出。\n"
        "平铺参数同『图片全屏水印』，多出帧区间与淡入淡出控制。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "视频帧序列(批量 IMAGE)"}),
                "水印大小比例": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 2.0, "step": 0.01,
                    "tooltip": "单个水印宽度相对帧宽度的比例"}),
                "不透明度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "水印不透明度(淡入淡出会在此基础上缩放)"}),
                "旋转角度": ("INT", {"default": -30, "min": -360, "max": 360, "tooltip": "水印倾斜角度(度)"}),
                "水平间距": ("INT", {"default": 10, "min": 0, "max": 4096, "tooltip": "相邻水印水平间距(像素)"}),
                "垂直间距": ("INT", {"default": 10, "min": 0, "max": 4096, "tooltip": "相邻水印垂直间距(像素)"}),
                "密度": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1, "tooltip": "铺设密度倍率"}),
                "起始帧": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "从第几帧开始显示水印(0=第一帧)"}),
                "结束帧": ("INT", {"default": -1, "min": -1, "max": 1000000, "tooltip": "到第几帧停止显示，-1=直到最后一帧"}),
                "淡入帧数": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "起始处用多少帧把水印从透明渐显到设定不透明度，0=不淡入"}),
                "淡出帧数": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "结束处用多少帧把水印渐隐到透明，0=不淡出"}),
                "反转遮罩": ("BOOLEAN", {"default": False, "tooltip": "把水印遮罩黑白反转"}),
            },
            "optional": {
                "水印图像": ("IMAGE", {"tooltip": "作为水印的图(留空则原样输出)"}),
                "水印遮罩": ("MASK", {"tooltip": "水印 alpha/形状，使背景透明"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "add_watermark"
    CATEGORY = "Noctyra/添加水印"

    def add_watermark(self, 图像, 水印大小比例=0.15, 不透明度=0.5, 旋转角度=-30,
                      水平间距=10, 垂直间距=10, 密度=1.0,
                      起始帧=0, 结束帧=-1, 淡入帧数=0, 淡出帧数=0, 反转遮罩=False,
                      水印图像=None, 水印遮罩=None):
        if 图像 is None or len(图像) == 0:
            return (torch.zeros((0, 1, 1, 3)),)

        start, end = resolve_frame_range(len(图像), 起始帧, 结束帧)
        watermark_pil = prepare_watermark_rgba(水印图像, 水印遮罩, 反转遮罩)

        out = []
        for i, t in enumerate(图像):
            opacity = video_fade_opacity(i, start, end, 淡入帧数, 淡出帧数, 不透明度)
            if opacity <= 0.0:
                out.append(t.unsqueeze(0) if t.dim() == 3 else t)
                continue
            out.append(_tile_frame(t, watermark_pil, 水印大小比例, opacity, 旋转角度, 水平间距, 垂直间距, 密度))
        return (torch.cat(out, dim=0),)


NODE_CLASS_MAPPINGS = {
    "AddFullscreenWatermark": AddFullscreenWatermark,
    "AddVideoFullscreenWatermark": AddVideoFullscreenWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddFullscreenWatermark": "图片添加水印（全屏）",
    "AddVideoFullscreenWatermark": "视频添加水印（全屏）",
}
