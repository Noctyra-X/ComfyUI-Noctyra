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
图片处理相关节点
"""
import os
import numpy as np
import torch
from PIL import Image
import folder_paths


class SaveImageNoMetadata:
    """保存图片（无元数据）节点
    
    保存图片到输出目录，不写入任何元数据信息
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    DESCRIPTION = (
        "保存图片到输出目录，且不写入任何元数据(无 PNG text/参数/工作流)，"
        "适合发布前清掉 AI 生成痕迹与隐私信息。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "要保存的图片(可批量)"}),
                "filename_prefix": ("STRING", {"default": "Clean_Image", "tooltip": "输出文件名前缀，会自动追加序号"}),
            },
            "optional": {
                "compress_level": ("INT", {"default": 4, "min": 0, "max": 9, "step": 1,
                    "tooltip": "PNG 压缩级别 0~9，越大文件越小越慢(0=不压缩最快)"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Noctyra/图片"

    def save_images(self, images, filename_prefix="Clean_Image", compress_level=4):
        if images is None or len(images) == 0:
            return {"ui": {"images": []}}

        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix,
                self.output_dir,
                images[0].shape[1],
                images[0].shape[0]
            )
        )

        results = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            file = f"{filename}_{counter:05}_.png"

            # pnginfo=None 确保不写入元数据
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=None,
                compress_level=compress_level,
            )

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}


# 节点映射
NODE_CLASS_MAPPINGS = {
    "SaveImageNoMetadata": SaveImageNoMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageNoMetadata": "Save Image (No Metadata)",
}
