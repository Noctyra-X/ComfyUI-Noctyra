# ComfyUI-Noctyra
# Copyright (C) 2026 Noctyra
# GPLv3+ — 详见仓库根目录 LICENSE
"""
无缝贴图查看器:将输入图片 3×3 平铺,用于检查无缝贴图(seamless tile)的接缝。

输入一张材质/贴图图片,节点输出 9 倍面积(3×3 平铺)的合成图并在节点内预览,
看十字交界处是否还有缝、是否有色差线 —— 这是验证无缝贴图最直观的方法。
"""
import os
import random
import numpy as np
import torch
from PIL import Image

import folder_paths


class SeamlessTilePreview:
    """无缝贴图查看器 (3×3)"""

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        # 文件名前缀加随机后缀,避免并发覆盖
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(8)
        )
        self.compress_level = 4

    DESCRIPTION = (
        "把输入图片在 X/Y 两个方向各重复 3 次拼成 3×3 (共 9 张) 平铺图,"
        "用来检查无缝贴图(seamless tile)的接缝。出图后看十字交界处:"
        "纹理连续穿过 = 真无缝 ✓,有缝/色差线 = 没做对。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "要平铺查看的贴图(只取 batch 的第 1 张)"}),
            },
            "optional": {
                "grid": ("INT", {
                    "default": 3, "min": 2, "max": 8, "step": 1,
                    "tooltip": "平铺网格大小(默认 3,即 3×3=9 张)",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("tiled",)
    FUNCTION = "tile_preview"
    OUTPUT_NODE = True
    CATEGORY = "Noctyra/图片"

    def tile_preview(self, image, grid=3):
        if image is None or len(image) == 0:
            return {"ui": {"images": []}, "result": (image,)}

        # 只用 batch 的第 1 张(查无缝时不需要批量)
        # image: [B, H, W, C], 0..1
        single = image[:1]  # [1, H, W, C]
        # X 方向重复 grid 次
        row = torch.cat([single] * grid, dim=2)        # [1, H, grid*W, C]
        # Y 方向重复 grid 次
        tiled = torch.cat([row] * grid, dim=1)          # [1, grid*H, grid*W, C]

        # 保存到 temp 目录用于预览
        full_output_folder, filename, counter, subfolder, _ = (
            folder_paths.get_save_image_path(
                "Noctyra_seamless" + self.prefix_append,
                self.output_dir,
                tiled.shape[2],
                tiled.shape[1],
            )
        )
        results = []
        arr = (255.0 * tiled[0].cpu().numpy()).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        file = f"{filename}_{counter:05}_.png"
        img.save(os.path.join(full_output_folder, file),
                 compress_level=self.compress_level)
        results.append({"filename": file, "subfolder": subfolder, "type": self.type})

        return {"ui": {"images": results}, "result": (tiled,)}


NODE_CLASS_MAPPINGS = {
    "NoctyraSeamlessTilePreview": SeamlessTilePreview,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NoctyraSeamlessTilePreview": "无缝贴图查看器 (3×3)",
}
