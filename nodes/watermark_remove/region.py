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
去除可见水印相关节点

思路参考开源项目 remove-ai-watermarks (MIT, wiltodelta)：对 Gemini / 豆包
等 AI 生图工具留在画面角落或底部的可见水印，用区域掩膜 + 图像修复(inpaint)
的确定性方法抹除。本模块为独立实现，不依赖也不下载任何外部模型。

- 隐形水印(SynthID 等)需扩散重生成，ComfyUI 可直接用原生 VAE编码→低重绘
  KSampler→VAE解码 实现，故不在此重复封装。
- 去元数据可使用 Noctyra 既有的 "Save Image (No Metadata)" 节点。
"""
import logging

import cv2
import numpy as np
import torch

from .._utils import (
    tensor_to_rgb_uint8 as _tensor_to_rgb_uint8,
    rgb_uint8_to_tensor as _rgb_uint8_to_tensor,
)

logger = logging.getLogger("noctyra")

_INPAINT_FLAGS = {
    "Telea": cv2.INPAINT_TELEA,
    "Navier-Stokes": cv2.INPAINT_NS,
}


def _rect_from_preset(preset, 角落位置, 角落大小, 条带位置, 条带高度,
                      自定义X, 自定义Y, 自定义宽, 自定义高, w, h):
    """根据预设返回水印矩形 (x0, y0, x1, y1)，坐标为像素。"""
    if preset == "角落":
        side = max(1, int(min(w, h) * 角落大小))
        side_w = min(side, w)
        side_h = min(side, h)
        if 角落位置 == "右下":
            x0, y0 = w - side_w, h - side_h
        elif 角落位置 == "左下":
            x0, y0 = 0, h - side_h
        elif 角落位置 == "右上":
            x0, y0 = w - side_w, 0
        else:  # 左上
            x0, y0 = 0, 0
        return x0, y0, x0 + side_w, y0 + side_h

    if preset == "条带":
        strip_h = max(1, min(h, int(h * 条带高度)))
        if 条带位置 == "底部":
            return 0, h - strip_h, w, h
        else:  # 顶部
            return 0, 0, w, strip_h

    # 自定义矩形
    x0 = int(np.clip(自定义X, 0.0, 1.0) * w)
    y0 = int(np.clip(自定义Y, 0.0, 1.0) * h)
    x1 = int(min(w, x0 + max(0.0, 自定义宽) * w))
    y1 = int(min(h, y0 + max(0.0, 自定义高) * h))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


class RemoveVisibleWatermark:
    """去除可见水印（区域修复）

    对画面指定区域（角落 / 条带 / 自定义矩形 / 外部遮罩）做图像修复，
    适用于抹除 AI 生图工具留在角落或底部的可见 logo / 文字水印。
    """

    DESCRIPTION = (
        "通用可见水印去除：手动框选水印所在区域(角落/条带/自定义矩形/外部遮罩)后做图像修复(inpaint)。\n"
        "适合抹除任意位置的 logo / 文字水印。坐标按图像比例(0~1)。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "待去水印的图像"}),
                "模式": (["角落", "条带", "自定义矩形", "外部遮罩"], {"default": "角落",
                    "tooltip": "区域选取方式：角落=四角之一；条带=顶/底整条；自定义矩形=按XYWH比例；外部遮罩=用连入的MASK"}),
                "角落位置": (["右下", "左下", "右上", "左上"], {"default": "右下", "tooltip": "【角落模式】水印在哪个角"}),
                "角落大小": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "【角落模式】方块边长占图像短边的比例(0.15=15%)"}),
                "条带位置": (["底部", "顶部"], {"default": "底部", "tooltip": "【条带模式】整条在顶部还是底部"}),
                "条带高度": ("FLOAT", {"default": 0.08, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "【条带模式】条带高度占图像高度的比例"}),
                "自定义X": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "【自定义矩形】左上角 X(占宽比例)"}),
                "自定义Y": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "【自定义矩形】左上角 Y(占高比例)"}),
                "自定义宽": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "【自定义矩形】宽(占宽比例)"}),
                "自定义高": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "【自定义矩形】高(占高比例)"}),
                "后端": (["cv2(快)", "LaMa(高质量)"], {"default": "cv2(快)",
                    "tooltip": "修复后端。cv2=即时无需模型；LaMa=神经修复质量更好(首次用自动下~200MB模型)"}),
                "修复方法": (["Telea", "Navier-Stokes"], {"default": "Telea", "tooltip": "【cv2后端】inpaint 算法"}),
                "修复半径": ("INT", {"default": 6, "min": 1, "max": 64, "tooltip": "【cv2后端】参考邻域半径，越大越平滑越慢"}),
                "遮罩扩张": ("INT", {"default": 4, "min": 0, "max": 128, "tooltip": "把选区向外扩几像素，确保盖住水印的抗锯齿边缘"}),
                "边缘羽化": ("INT", {"default": 4, "min": 0, "max": 128, "tooltip": "修复区与原图边界的羽化，软化接缝"}),
            },
            "optional": {
                "遮罩": ("MASK", {"tooltip": "【外部遮罩模式】白色区域=要修复的水印位置"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像", "遮罩")
    FUNCTION = "remove"
    CATEGORY = "Noctyra/水印去除"

    def _build_mask(self, mode, h, w, preset_args, ext_mask, idx):
        """返回 uint8 单通道掩膜(0/255)，水印区域为 255。"""
        if mode == "外部遮罩":
            if ext_mask is None:
                raise ValueError("模式为『外部遮罩』但未连接 MASK 输入。")
            m = ext_mask[idx] if ext_mask.dim() == 3 and idx < ext_mask.shape[0] else (
                ext_mask[0] if ext_mask.dim() == 3 else ext_mask)
            arr = np.clip(m.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            if arr.shape != (h, w):
                arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
            return (arr > 127).astype(np.uint8) * 255

        mask = np.zeros((h, w), dtype=np.uint8)
        x0, y0, x1, y1 = _rect_from_preset(mode, *preset_args, w=w, h=h)
        mask[y0:y1, x0:x1] = 255
        return mask

    def remove(self, 图像, 模式, 角落位置, 角落大小, 条带位置, 条带高度,
               自定义X, 自定义Y, 自定义宽, 自定义高,
               后端, 修复方法, 修复半径, 遮罩扩张, 边缘羽化, 遮罩=None):
        if 图像 is None or len(图像) == 0:
            return (torch.zeros((0, 1, 1, 3)), torch.zeros((0, 1, 1)))

        flag = _INPAINT_FLAGS.get(修复方法, cv2.INPAINT_TELEA)
        use_lama = 后端.startswith("LaMa")
        preset_args = (角落位置, 角落大小, 条带位置, 条带高度,
                       自定义X, 自定义Y, 自定义宽, 自定义高)

        out_images = []
        out_masks = []
        for idx, img_tensor in enumerate(图像):
            rgb = _tensor_to_rgb_uint8(img_tensor)
            h, w = rgb.shape[:2]

            mask = self._build_mask(模式, h, w, preset_args, 遮罩, idx)

            if 遮罩扩张 > 0:
                k = 2 * 遮罩扩张 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                mask = cv2.dilate(mask, kernel)

            if mask.max() == 0:
                logger.warning("去除可见水印: 掩膜为空，原样返回该帧。")
                out_images.append(_rgb_uint8_to_tensor(rgb))
                out_masks.append(torch.zeros((h, w), dtype=torch.float32))
                continue

            if use_lama:
                # 复用 vendored region_eraser 的 LaMa-ONNX 后端(首次用时自动下 ~200MB)
                from .._vendor import ensure as _ensure_vendor
                _ensure_vendor()
                from remove_ai_watermarks.region_eraser import erase_lama
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                out_bgr = erase_lama(bgr, mask)
                inpainted = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
            else:
                inpainted = cv2.inpaint(rgb, mask, 修复半径, flag)

            if 边缘羽化 > 0:
                # 用模糊后的掩膜做羽化合成，软化修复区域边缘接缝
                ksize = 2 * 边缘羽化 + 1
                soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0,
                                        (ksize, ksize), 0)
                soft = soft[..., None]
                blended = rgb.astype(np.float32) * (1.0 - soft) + \
                    inpainted.astype(np.float32) * soft
                result = np.clip(blended, 0, 255).astype(np.uint8)
            else:
                result = inpainted

            out_images.append(_rgb_uint8_to_tensor(result))
            out_masks.append(torch.from_numpy(mask.astype(np.float32) / 255.0))

        images_out = torch.cat(out_images, dim=0)
        masks_out = torch.stack(out_masks, dim=0)
        return (images_out, masks_out)


NODE_CLASS_MAPPINGS = {
    "RemoveVisibleWatermark": RemoveVisibleWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveVisibleWatermark": "去除可见水印（区域修复）",
}
