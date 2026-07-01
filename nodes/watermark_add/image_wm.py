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
水印处理相关节点
"""
import logging
import torch
from PIL import Image

from .._utils import (
    tensor_to_pil,
    pil_to_tensor,
    prepare_watermark_rgba,
    apply_watermark_frame,
)

logger = logging.getLogger("noctyra")


class AddImageWatermark:
    """添加图像水印节点（支持遮罩）

    在 ZML_添加图像水印的基础上新增 MASK 输入。
    当提供水印图像的遮罩时，遮罩会作为水印的 alpha 通道，
    使得水印图像的背景区域真正透明，而不是直接整张贴到背景图上。
    """

    DESCRIPTION = (
        "把一张水印图叠到图像的九宫格位置之一(角/边/中心)。\n"
        "可选连入『水印遮罩』当作水印 alpha，使水印背景真正透明而非整块贴上。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "被加水印的底图"}),
                "水印大小比例": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 2.0, "step": 0.01,
                    "tooltip": "水印宽度相对底图宽度的比例(0.25=占 1/4 宽)"}),
                "不透明度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "水印整体不透明度，0=透明 1=不透明"}),
                "位置": (["左上", "中上", "右上", "左中", "居中", "右中", "左下", "中下", "右下"],
                    {"tooltip": "水印贴在底图的九宫格位置"}),
                "水平边距": ("INT", {"default": 30, "min": 0, "max": 4096, "tooltip": "距左/右边缘的像素间距(居中位置无效)"}),
                "垂直边距": ("INT", {"default": 30, "min": 0, "max": 4096, "tooltip": "距上/下边缘的像素间距(居中位置无效)"}),
                "反转遮罩": ("BOOLEAN", {"default": False, "tooltip": "把水印遮罩黑白反转(遮罩选区与预期相反时开)"}),
            },
            "optional": {
                "水印图像": ("IMAGE", {"tooltip": "作为水印的图(留空则不加水印，原样输出)"}),
                "水印遮罩": ("MASK", {"tooltip": "水印的 alpha/形状，使其背景透明。配合『文字水印生成』很方便"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "add_watermark"
    CATEGORY = "Noctyra/添加水印"

    def add_watermark(
        self,
        图像,
        水印大小比例=0.25,
        不透明度=1.0,
        位置="右下",
        水平边距=30,
        垂直边距=30,
        反转遮罩=False,
        水印图像=None,
        水印遮罩=None,
    ):
        watermark_pil = prepare_watermark_rgba(水印图像, 水印遮罩, 反转遮罩)

        processed_images = []
        for img_tensor in 图像:
            pil_image = tensor_to_pil(img_tensor).convert("RGBA")
            result = apply_watermark_frame(
                pil_image, watermark_pil,
                size_ratio=水印大小比例, opacity=不透明度, position=位置,
                margin_x=水平边距, margin_y=垂直边距,
            )
            processed_images.append(pil_to_tensor(result))

        return (torch.cat(processed_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "AddImageWatermark": AddImageWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddImageWatermark": "图片添加水印（位置）",
}
