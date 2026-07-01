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
# 的 gemini_engine.py，后者又是对 Allen Kuo (allenk) 的 C++ 工具
# GeminiWatermarkTool 反向 alpha 混合算法的移植。完整 MIT 声明见仓库根目录
# THIRD_PARTY_NOTICES.md。
# ---------------------------------------------------------------------------

"""
去除 Gemini 可见水印（反向 alpha 混合）

Gemini/Imagen 的星形水印是 alpha 混合上去的：
    含水印 = a * logo + (1 - a) * 原图
本引擎用内嵌的 alpha 模板(gemini_bg_48/96.png)反解，真正还原底层像素：
    原图 = (含水印 - a * logo) / (1 - a)

定位用多尺度 NCC 模板匹配在右下角搜索；未检出时不改动图像（对没有
水印的区域做反解会产生可见的反相伪影）。
"""
import logging
import os

import cv2
import numpy as np
import torch

from .._utils import tensor_to_rgb_uint8, rgb_uint8_to_tensor

logger = logging.getLogger("noctyra")

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def _calculate_alpha_map(bg_capture):
    """从黑底背景捕获图计算 alpha 模板：alpha = max(R,G,B)/255。"""
    if bg_capture.ndim == 2:
        gray = bg_capture.astype(np.float32)
    elif bg_capture.shape[2] >= 3:
        gray = np.max(bg_capture[:, :, :3], axis=2).astype(np.float32)
    else:
        gray = bg_capture[:, :, 0].astype(np.float32)
    return gray / 255.0


def _load_alpha(name, size):
    path = os.path.join(_ASSET_DIR, name)
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"无法解码内嵌模板: {path}")
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return _calculate_alpha_map(img)


class _GeminiEngine:
    """单例式持有 alpha 模板，避免每帧重复解码。"""

    _instance = None

    def __init__(self, logo_value=255.0):
        self.logo_value = logo_value
        self._alpha_small = _load_alpha("gemini_bg_48.png", 48)
        self._alpha_large = _load_alpha("gemini_bg_96.png", 96)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def interp_alpha(self, size_px):
        src = self._alpha_large
        if size_px == src.shape[1]:
            return src.copy()
        interp = cv2.INTER_LINEAR if size_px > src.shape[1] else cv2.INTER_AREA
        return cv2.resize(src, (size_px, size_px), interpolation=interp)

    def detect(self, rgb):
        """多尺度 NCC 检测右下角水印。返回 (region(x,y,w,h), confidence)。"""
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        search = int(min(min(w, h), 256))
        sx1, sy1 = max(0, w - search), max(0, h - search)
        sr = gray[sy1:h, sx1:w]

        best_scale, best_score, best_raw, best_loc = 0, -1.0, -1.0, (0, 0)
        for scale in range(16, 120, 2):
            if scale > sr.shape[0] or scale > sr.shape[1]:
                continue
            tmpl = cv2.resize(self._alpha_large, (scale, scale), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(sr, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            weight = min(1.0, (scale / 96.0) ** 0.5)
            adj = max_val * weight
            if adj > best_score:
                best_score, best_scale, best_loc, best_raw = adj, scale, max_loc, max_val

        if best_scale == 0:
            return (0, 0, 0, 0), 0.0

        px, py = sx1 + best_loc[0], sy1 + best_loc[1]
        spatial = float(best_raw)
        if spatial < 0.25:
            return (px, py, best_scale, best_scale), max(0.0, spatial * 0.5)

        # 梯度 NCC
        x2, y2 = min(w, px + best_scale), min(h, py + best_scale)
        region = gray[py:y2, px:x2]
        alpha_region = self.interp_alpha(best_scale)[: y2 - py, : x2 - px]
        gx = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
        agx = cv2.Sobel(alpha_region, cv2.CV_32F, 1, 0, ksize=3)
        agy = cv2.Sobel(alpha_region, cv2.CV_32F, 0, 1, ksize=3)
        gmatch = cv2.matchTemplate(cv2.magnitude(gx, gy), cv2.magnitude(agx, agy), cv2.TM_CCOEFF_NORMED)
        _, grad_score, _, _ = cv2.minMaxLoc(gmatch)

        # 方差分析（与水印上方同尺寸参考区比对）
        var_score = 0.0
        ref_h = min(py, best_scale)
        if ref_h > 8:
            ref = cv2.cvtColor(rgb[py - ref_h:py, px:x2], cv2.COLOR_RGB2GRAY)
            _, s_wm = cv2.meanStdDev((region * 255).astype(np.uint8))
            _, s_ref = cv2.meanStdDev(ref)
            if s_ref[0][0] > 5.0:
                var_score = max(0.0, min(1.0, 1.0 - (s_wm[0][0] / s_ref[0][0])))

        conf = spatial * 0.50 + float(grad_score) * 0.30 + var_score * 0.20
        return (px, py, best_scale, best_scale), float(max(0.0, min(1.0, conf)))

    def inpaint_residual(self, rgb, region, strength=0.85, method="ns", radius=10, padding=32):
        """对反解后的残留边缘做修复（移植自上游 inpaint_residual）。

        用 alpha 模板的梯度幅值生成稀疏掩膜，只修复星形边缘那些因插值导致
        反解数学失真的像素，按 strength 与原图混合。method: ns/telea/gaussian。
        就地修改 rgb。
        """
        x, y, rw, rh = region
        if rw < 4 or rh < 4:
            return
        strength = max(0.0, min(1.0, strength))
        if strength < 0.001:
            return

        ih, iw = rgb.shape[:2]
        px1, py1 = max(0, x - padding), max(0, y - padding)
        px2, py2 = min(iw, x + rw + padding), min(ih, y + rh + padding)
        if (px2 - px1) < 8 or (py2 - py1) < 8:
            return
        ix1, iy1 = x - px1, y - py1

        src_alpha = self._alpha_large
        interp = cv2.INTER_LINEAR if rw > src_alpha.shape[1] else cv2.INTER_AREA
        alpha_resized = cv2.resize(src_alpha, (rw, rh), interpolation=interp)

        gx = cv2.Sobel(alpha_resized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(alpha_resized, cv2.CV_32F, 0, 1, ksize=3)
        gmag = cv2.magnitude(gx, gy)
        gmin, gmax = gmag.min(), gmag.max()
        if gmax <= gmin:
            return
        grad_weight = np.sqrt((gmag - gmin) / (gmax - gmin))
        grad_weight = cv2.dilate(grad_weight, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        padded = rgb[py1:py2, px1:px2].copy()
        if method == "gaussian":
            filled = cv2.GaussianBlur(padded, (0, 0), sigmaX=2.0)
        else:
            flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
            binm = (grad_weight * 255).astype(np.uint8)
            _, binm = cv2.threshold(binm, 30, 255, cv2.THRESH_BINARY)
            mask_full = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
            mask_full[iy1:iy1 + rh, ix1:ix1 + rw] = binm
            filled = cv2.inpaint(padded, mask_full, radius, flag)

        weight_full = np.zeros((py2 - py1, px2 - px1), dtype=np.float32)
        weight_full[iy1:iy1 + rh, ix1:ix1 + rw] = grad_weight * strength
        w3 = weight_full[:, :, None]
        blended = padded.astype(np.float32) * (1 - w3) + filled.astype(np.float32) * w3
        rgb[py1:py2, px1:px2] = blended.astype(np.uint8)

    def reverse_alpha_blend(self, rgb, alpha_map, pos):
        """就地反向 alpha 混合：原图 = (含水印 - a*logo) / (1-a)。"""
        x, y = pos
        ah, aw = alpha_map.shape[:2]
        ih, iw = rgb.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + aw), min(ih, y + ah)
        if x1 >= x2 or y1 >= y2:
            return
        ax1, ay1 = x1 - x, y1 - y
        alpha = alpha_map[ay1:ay1 + (y2 - y1), ax1:ax1 + (x2 - x1)].copy()
        roi = rgb[y1:y2, x1:x2].astype(np.float32)

        mask = alpha >= 0.002
        alpha = np.clip(alpha, 0.0, 0.99)
        a3 = alpha[:, :, None]
        restored = np.clip((roi - a3 * self.logo_value) / (1.0 - a3), 0.0, 255.0)
        rgb[y1:y2, x1:x2] = np.where(mask[:, :, None], restored, roi).astype(np.uint8)


class RemoveGeminiWatermark:
    """去除 Gemini 可见水印（反向 alpha 混合）

    自动检测右下角星形水印并反解还原底层像素。检出置信度低于阈值时
    不改动该帧（避免反相伪影）。可选对残留边缘做轻度 inpaint。
    """

    DESCRIPTION = (
        "去除 Gemini/Imagen 图右下角的星标水印。先多尺度匹配定位星标，再二选一去除：\n"
        "· 直接修复(inpaint)：把星标区域整块抹掉重画，对低对比/纯色背景更稳(推荐)。\n"
        "· 反向alpha还原：用内嵌模板反解还原底层像素，检测准时质量最高、但难例易留残影。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"tooltip": "含 Gemini 右下角彩色星标的图像"}),
                "去除方式": (["直接修复(inpaint)", "反向alpha还原"], {"default": "直接修复(inpaint)",
                    "tooltip": "直接修复=抹掉星标区域重画(稳)；反向还原=反解还原像素(检测准时质量高)"}),
                "检测阈值": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "星标匹配置信度低于此值则判定未检出、原样返回。调低=更激进(可能误检)"}),
                "模板尺寸": (["自动", "小(48)", "大(96)"], {"default": "自动",
                    "tooltip": "反向还原用的模板基准尺寸。自动=按多尺度匹配结果(一般保持自动)"}),
                "修复区域扩大": ("INT", {"default": 50, "min": 0, "max": 512,
                    "tooltip": "【直接修复】检测框向外扩多少像素再抹除。星标没盖全就调大(如 70~90)"}),
                "修复方法": (["Navier-Stokes", "Telea", "Gaussian"], {"default": "Navier-Stokes",
                    "tooltip": "inpaint 算法。NS/Telea 为 OpenCV 修复(直接修复模式只用这两种)"}),
                "修复半径": ("INT", {"default": 10, "min": 1, "max": 64,
                    "tooltip": "inpaint 参考的邻域半径，越大越平滑但越慢"}),
                "残留修复": ("BOOLEAN", {"default": True,
                    "tooltip": "【反向还原】对反解后星标边缘的残留再做一遍轻度修复"}),
                "修复强度": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "【反向还原】残留修复的混合强度，0=不修、1=全用修复结果"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("图像", "遮罩", "检测信息")
    FUNCTION = "remove"
    CATEGORY = "Noctyra/水印去除"

    def remove(self, 图像, 去除方式="直接修复(inpaint)", 检测阈值=0.35, 模板尺寸="自动",
               修复区域扩大=50, 修复方法="Navier-Stokes", 修复半径=10,
               残留修复=True, 修复强度=0.85):
        if 图像 is None or len(图像) == 0:
            return (torch.zeros((0, 1, 1, 3)), torch.zeros((0, 1, 1)), "")

        engine = _GeminiEngine.get()
        _method = {"Navier-Stokes": "ns", "Telea": "telea", "Gaussian": "gaussian"}[修复方法]
        _flag = cv2.INPAINT_TELEA if 修复方法 == "Telea" else cv2.INPAINT_NS
        use_inpaint = 去除方式.startswith("直接")
        out_images, out_masks, infos = [], [], []

        for img_tensor in 图像:
            rgb = tensor_to_rgb_uint8(img_tensor)
            h, w = rgb.shape[:2]
            mask = np.zeros((h, w), dtype=np.float32)

            (rx, ry, rw, rh), conf = engine.detect(rgb)
            if conf >= 检测阈值 and rw > 0:
                if 模板尺寸 == "小(48)":
                    rw = rh = 48
                elif 模板尺寸 == "大(96)":
                    rw = rh = 96

                if use_inpaint:
                    # 直接修复：以检测中心向外扩大成方框，inpaint 抹除(对纯色背景/难例更稳)
                    cx, cy = rx + rw // 2, ry + rh // 2
                    half = max(rw, rh) // 2 + 修复区域扩大
                    x0, y0 = max(0, cx - half), max(0, cy - half)
                    x1, y1 = min(w, cx + half), min(h, cy + half)
                    m8 = np.zeros((h, w), dtype=np.uint8)
                    m8[y0:y1, x0:x1] = 255
                    rgb = cv2.inpaint(rgb, m8, 修复半径, _flag)
                    mask[y0:y1, x0:x1] = 1.0
                    infos.append(f"修复 conf={conf:.3f} 框=({x0},{y0},{x1 - x0}x{y1 - y0})")
                else:
                    # 反向 alpha 还原(检测准时质量更高，但难例可能有残影)
                    engine.reverse_alpha_blend(rgb, engine.interp_alpha(rw), (rx, ry))
                    mask[ry:min(h, ry + rh), rx:min(w, rx + rw)] = 1.0
                    if 残留修复:
                        engine.inpaint_residual(rgb, (rx, ry, rw, rh),
                                                strength=修复强度, method=_method, radius=修复半径)
                    infos.append(f"还原 conf={conf:.3f} 区域=({rx},{ry},{rw}x{rh})")
            else:
                infos.append(f"未检出 conf={conf:.3f}，原样返回")

            out_images.append(rgb_uint8_to_tensor(rgb))
            out_masks.append(torch.from_numpy(mask))

        return (torch.cat(out_images, dim=0), torch.stack(out_masks, dim=0), "; ".join(infos))


NODE_CLASS_MAPPINGS = {
    "RemoveGeminiWatermark": RemoveGeminiWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveGeminiWatermark": "去除Gemini星标水印",
}
