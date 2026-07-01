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
视频加水印节点

视频在 ComfyUI 中即一批帧(IMAGE，B=帧数，常配合 VideoHelperSuite 的
加载视频/合成视频节点)。本节点把水印贴到指定帧区间，并支持淡入淡出。
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


class AddVideoWatermark:
    """给视频(帧序列)添加水印

    与图片水印一致的位置/大小/不透明度控制，另加视频专属的『起止帧 + 淡入淡出』：
    水印只在 [起始帧, 结束帧] 区间显示，可在区间两端做透明度渐变。
    """

    DESCRIPTION = (
        "给视频(帧序列)在九宫格位置叠加水印，位置/大小/不透明度同图片版。\n"
        "另支持只在 [起始帧, 结束帧] 区间显示，并可在两端做淡入淡出。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "视频帧序列(批量 IMAGE)"}),
                "水印大小比例": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 2.0, "step": 0.01,
                    "tooltip": "水印宽度相对帧宽度的比例"}),
                "不透明度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "水印不透明度(淡入淡出会在此基础上缩放)"}),
                "位置": (["左上", "中上", "右上", "左中", "居中", "右中", "左下", "中下", "右下"], {"default": "右下",
                    "tooltip": "水印在画面的九宫格位置"}),
                "水平边距": ("INT", {"default": 30, "min": 0, "max": 4096, "tooltip": "距左/右边缘的像素间距"}),
                "垂直边距": ("INT", {"default": 30, "min": 0, "max": 4096, "tooltip": "距上/下边缘的像素间距"}),
                "起始帧": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "从第几帧开始显示水印(0=第一帧)"}),
                "结束帧": ("INT", {"default": -1, "min": -1, "max": 1000000, "tooltip": "到第几帧停止显示，-1=直到最后一帧"}),
                "淡入帧数": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "起始处渐显所用帧数，0=不淡入"}),
                "淡出帧数": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "结束处渐隐所用帧数，0=不淡出"}),
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

    def add_watermark(
        self, 图像, 水印大小比例=0.25, 不透明度=1.0, 位置="右下",
        水平边距=30, 垂直边距=30,
        起始帧=0, 结束帧=-1, 淡入帧数=0, 淡出帧数=0, 反转遮罩=False,
        水印图像=None, 水印遮罩=None,
    ):
        if 图像 is None or len(图像) == 0:
            return (torch.zeros((0, 1, 1, 3)),)

        start, end = resolve_frame_range(len(图像), 起始帧, 结束帧)
        watermark_pil = prepare_watermark_rgba(水印图像, 水印遮罩, 反转遮罩)

        out = []
        for i, img_tensor in enumerate(图像):
            opacity = video_fade_opacity(i, start, end, 淡入帧数, 淡出帧数, 不透明度)
            if opacity <= 0.0:
                out.append(img_tensor.unsqueeze(0) if img_tensor.dim() == 3 else img_tensor)
                continue

            pil = tensor_to_pil(img_tensor).convert("RGBA")
            result = apply_watermark_frame(
                pil, watermark_pil,
                size_ratio=水印大小比例, opacity=opacity, position=位置,
                margin_x=水平边距, margin_y=垂直边距,
            )
            out.append(pil_to_tensor(result))

        return (torch.cat(out, dim=0),)


NODE_CLASS_MAPPINGS = {
    "AddVideoWatermark": AddVideoWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddVideoWatermark": "视频添加水印（位置）",
}
