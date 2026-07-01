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
# 本文件移植自 MIT 许可项目 remove-ai-watermarks (Copyright (c) 2025 wiltodelta)
# 的 doubao_engine.py。完整 MIT 声明见仓库根目录 THIRD_PARTY_NOTICES.md。
# ---------------------------------------------------------------------------

"""
去除豆包 "豆包AI生成" 可见水印（定位 → 字形掩膜 → 修复）

豆包(字节)在生成图右下角加 TC260 规定的 AIGC 文字标识，为低饱和浅灰
文字。本引擎几何定位右下角框，用极性感知规则提取浅色字形像素，再用
cv2 inpaint 修复。无字形像素或覆盖率过高(疑似文档密集文字)时不改动。
"""
import logging

import cv2
import numpy as np
import torch

from .._utils import tensor_to_rgb_uint8, rgb_uint8_to_tensor

logger = logging.getLogger("noctyra")

# 几何（相对图像宽度）：豆包水印随宽度缩放、锚定右下角
WM_WIDTH_FRAC = 0.185
WM_HEIGHT_FRAC = 0.065
MARGIN_RIGHT_FRAC = 0.012
MARGIN_BOTTOM_FRAC = 0.014

# 字形外观判定
MAX_SATURATION = 55   # 通道极差小于此值算"灰"
LOGO_MIN_LUMA = 150   # 字形绝对亮度下限
TOPHAT_DELTA = 12     # 字形需比局部背景亮出的级数

DETECT_MIN_COVERAGE = 0.16   # 框内字形覆盖率达此值判定为存在水印
MAX_INPAINT_COVERAGE = 0.50  # 超过则疑似文档背景，拒绝修复以免抹糊内容


class _DoubaoEngine:
    def locate(self, h, w):
        wm_w = max(40, int(w * WM_WIDTH_FRAC))
        wm_h = max(16, int(w * WM_HEIGHT_FRAC))
        margin_r = max(4, int(w * MARGIN_RIGHT_FRAC))
        margin_b = max(4, int(w * MARGIN_BOTTOM_FRAC))
        x = max(0, w - margin_r - wm_w)
        y = max(0, h - margin_b - wm_h)
        return x, y, min(wm_w, w - x), min(wm_h, h - y)

    def extract_mask(self, rgb, box):
        """极性感知地提取框内浅色低饱和字形，返回全图 uint8 掩膜。"""
        h, w = rgb.shape[:2]
        x, y, bw, bh = box
        roi = rgb[y:y + bh, x:x + bw].astype(np.float32)

        luma = roi.mean(axis=2)
        sat = roi.max(axis=2) - roi.min(axis=2)
        grayish = sat < MAX_SATURATION

        sigma = max(4.0, bh * 0.4)
        local_bg = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
        tophat = luma - local_bg

        cand = grayish & (tophat > TOPHAT_DELTA) & (luma > LOGO_MIN_LUMA)
        glyph = cand.astype(np.uint8) * 255
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        mask = np.zeros((h, w), np.uint8)
        mask[y:y + bh, x:x + bw] = glyph
        return mask

    def coverage(self, mask, box):
        x, y, bw, bh = box
        return float((mask[y:y + bh, x:x + bw] > 0).sum()) / float(max(1, bw * bh))


class RemoveDoubaoWatermark:
    """去除豆包 "豆包AI生成" 可见水印

    几何定位右下角文字条 → 提取浅色字形掩膜 → inpaint 修复。
    """

    DESCRIPTION = (
        "专门去除豆包(字节)生图右下角的『豆包AI生成』浅灰文字标识：自动几何定位右下角"
        "→ 提取浅色字形 → inpaint 修复，无需手动框选。\n"
        "检测到覆盖率过高(疑似密集文档文字)时会自动跳过以免抹糊，可用『强制修复』覆盖。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "豆包生图(右下角带『豆包AI生成』标识)"}),
                "修复方法": (["Telea", "Navier-Stokes"], {"default": "Telea", "tooltip": "cv2 inpaint 算法，二者效果接近"}),
                "修复半径": ("INT", {"default": 6, "min": 1, "max": 64, "tooltip": "参考邻域半径，越大越平滑越慢"}),
                "遮罩扩张": ("INT", {"default": 3, "min": 0, "max": 64, "tooltip": "字形掩膜向外扩几像素，确保盖住抗锯齿边缘"}),
                "强制修复": ("BOOLEAN", {"default": False,
                    "tooltip": "无视『检出覆盖率太低/太高』的安全判定，强行修复定位框。误判没去掉时再开"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("图像", "遮罩", "检测信息")
    FUNCTION = "remove"
    CATEGORY = "Noctyra/水印去除"

    def remove(self, 图像, 修复方法, 修复半径, 遮罩扩张, 强制修复):
        if 图像 is None or len(图像) == 0:
            return (torch.zeros((0, 1, 1, 3)), torch.zeros((0, 1, 1)), "")

        engine = _DoubaoEngine()
        flag = cv2.INPAINT_TELEA if 修复方法 == "Telea" else cv2.INPAINT_NS
        out_images, out_masks, infos = [], [], []

        for img_tensor in 图像:
            rgb = tensor_to_rgb_uint8(img_tensor)
            h, w = rgb.shape[:2]

            box = engine.locate(h, w)
            mask = engine.extract_mask(rgb, box)
            cov = engine.coverage(mask, box)
            conf = max(0.0, min(1.0, (cov - 0.06) / 0.20))

            if not mask.any():
                infos.append(f"未发现字形(cov={cov:.3f})，原样返回")
            elif cov > MAX_INPAINT_COVERAGE and not 强制修复:
                infos.append(f"覆盖率过高 cov={cov:.3f}(疑似文档)，跳过；如确需可开启『强制修复』")
            elif cov < DETECT_MIN_COVERAGE and not 强制修复:
                infos.append(f"未检出 cov={cov:.3f} conf={conf:.3f}，原样返回")
            else:
                m = mask
                if 遮罩扩张 > 0:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * 遮罩扩张 + 1,) * 2)
                    m = cv2.dilate(mask, k)
                rgb = cv2.inpaint(rgb, m, 修复半径, flag)
                mask = m
                infos.append(f"已修复 cov={cov:.3f} conf={conf:.3f} 区域={box}")

            out_images.append(rgb_uint8_to_tensor(rgb))
            out_masks.append(torch.from_numpy(mask.astype(np.float32) / 255.0))

        return (torch.cat(out_images, dim=0), torch.stack(out_masks, dim=0), "; ".join(infos))


NODE_CLASS_MAPPINGS = {
    "RemoveDoubaoWatermark": RemoveDoubaoWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveDoubaoWatermark": "去除豆包AIGC文字水印",
}
