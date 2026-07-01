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
水印去空白裁剪节点

把一张周围有大量空白的水印图，自动裁到"刚好包住水印内容"的最小矩形(不旋转)。
输出裁好的 图像 + 遮罩，可直接接到加水印节点。
"""
import logging

import numpy as np
import torch

logger = logging.getLogger("noctyra")


def _content_mask(rgb, mask, mode, threshold):
    """返回 HxW 布尔图：True=水印内容，False=空白。"""
    if mask is not None and mode == "按遮罩":
        return mask >= threshold

    if mode == "按背景色·白":
        thr = (1.0 - threshold) * 255.0
        return ~np.all(rgb >= thr, axis=2)
    if mode == "按背景色·黑":
        thr = threshold * 255.0
        return ~np.all(rgb <= thr, axis=2)

    # 自动：取四角中位色为背景，差异超阈值即为内容
    corners = np.stack([rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]]).astype(np.int16)
    bg = np.median(corners, axis=0)
    diff = np.abs(rgb.astype(np.int16) - bg).max(axis=2)
    return diff > (threshold * 255.0)


def _bbox(content):
    ys, xs = np.where(content)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class TrimWatermark:
    """水印去空白裁剪（自动包围）

    自动检测水印内容的最小包围盒并裁剪掉周围空白（不旋转）。多帧时取并集包围盒，
    保证输出尺寸统一。内容判定：有遮罩用遮罩，否则按背景色(自动/白/黑)。
    """

    DESCRIPTION = (
        "把周围有大量空白的水印图，自动裁到刚好包住水印内容的最小矩形(不旋转)。\n"
        "输出裁好的 图像 + 遮罩，可直接接到加水印节点。多帧取并集包围盒，输出尺寸统一。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "周围带大量空白、待裁剪的水印图"}),
                "内容判定": (["自动(四角背景)", "按遮罩", "按背景色·白", "按背景色·黑"], {"default": "自动(四角背景)",
                    "tooltip": "怎么区分『水印内容』和『空白』：自动=取四角中位色当背景；按遮罩=用连入的MASK；白/黑=白底或黑底水印"}),
                "阈值": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "与背景差异多大才算『内容』(0.1=10%)。调大=只保留明显内容；按遮罩时为遮罩二值化阈值"}),
                "边距": ("INT", {"default": 0, "min": 0, "max": 512, "tooltip": "包围盒外再留几像素(0=贴边裁)"}),
            },
            "optional": {
                "遮罩": ("MASK", {"tooltip": "【按遮罩】用它的非零区域定边界(最准)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像", "遮罩")
    FUNCTION = "trim"
    CATEGORY = "Noctyra/添加水印"

    def trim(self, 图像, 内容判定="自动(四角背景)", 阈值=0.1, 边距=0, 遮罩=None):
        if 图像 is None or len(图像) == 0:
            return (torch.zeros((0, 1, 1, 3)), torch.zeros((0, 1, 1)))

        n = len(图像)
        mode = "按遮罩" if 内容判定.startswith("按遮罩") else 内容判定

        # 遮罩批次数 >1 但短于图像批次：属真正长度不匹配，越界帧回退 mask[0]，只警告一次
        if (遮罩 is not None and 遮罩.dim() == 3
                and 1 < 遮罩.shape[0] < n):
            logger.warning(
                f"水印裁剪: 遮罩批次({遮罩.shape[0]})短于图像批次({n})，"
                "越界帧回退到 mask[0]。")

        # 逐帧检测内容，取并集包围盒（保证多帧输出同尺寸）
        union = None
        per_frame_content = []
        for i, img_tensor in enumerate(图像):
            rgb = np.clip(img_tensor.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            h, w = rgb.shape[:2]
            mk = None
            if 遮罩 is not None:
                m = 遮罩[i] if (遮罩.dim() == 3 and i < 遮罩.shape[0]) else (遮罩[0] if 遮罩.dim() == 3 else 遮罩)
                mk = m.cpu().numpy()
                if mk.shape != (h, w):  # 尺寸不符则忽略遮罩
                    mk = None
            if mode == "按遮罩" and mk is None:
                logger.warning("水印裁剪: 选了『按遮罩』但无有效遮罩，改用自动背景检测。")
                content = _content_mask(rgb, None, "自动", 阈值)
            else:
                content = _content_mask(rgb, mk, mode, 阈值)
            per_frame_content.append((rgb, content, mk))
            bb = _bbox(content)
            if bb is None:
                continue
            if union is None:
                union = list(bb)
            else:
                union[0] = min(union[0], bb[0]); union[1] = min(union[1], bb[1])
                union[2] = max(union[2], bb[2]); union[3] = max(union[3], bb[3])

        h0, w0 = 图像[0].shape[0], 图像[0].shape[1]
        if union is None:
            logger.warning("水印裁剪: 未检测到内容，原样返回。")
            x0, y0, x1, y1 = 0, 0, w0, h0
        else:
            x0 = max(0, union[0] - 边距); y0 = max(0, union[1] - 边距)
            x1 = min(w0, union[2] + 边距); y1 = min(h0, union[3] + 边距)

        out_imgs, out_masks = [], []
        for rgb, content, mk in per_frame_content:
            cropped = rgb[y0:y1, x0:x1]
            out_imgs.append(torch.from_numpy(cropped.astype(np.float32) / 255.0).unsqueeze(0))
            # 遮罩：有输入遮罩用裁后的，否则用检测到的内容作 alpha
            base_mask = mk if mk is not None else content.astype(np.float32)
            out_masks.append(torch.from_numpy(base_mask[y0:y1, x0:x1].astype(np.float32)).unsqueeze(0))

        return (torch.cat(out_imgs, dim=0), torch.cat(out_masks, dim=0))


NODE_CLASS_MAPPINGS = {
    "TrimWatermark": TrimWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TrimWatermark": "遮罩裁剪",
}
