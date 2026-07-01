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
生成文字水印节点

把一段文字渲染成水印，输出 IMAGE(文字颜色) + MASK(文字形状的 alpha)。
两路一起接到加水印节点的『水印图像 / 水印遮罩』即可作为透明文字水印使用。
"""
import logging
import os

import numpy as np
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont

from .._utils import plugin_fonts_dir as _plugin_fonts_dir

logger = logging.getLogger("noctyra")

_FONT_EXTS = (".ttf", ".ttc", ".otf")
# 常用字体文件名(小写) -> 友好显示名(仅用于系统字体的可读化)
_FRIENDLY = {
    "sourcehansanscn-regular.otf": "思源黑体",  # 插件自带(SIL OFL 开源)
    "msyh.ttc": "微软雅黑", "msyhbd.ttc": "微软雅黑·粗", "simhei.ttf": "黑体",
    "simsun.ttc": "宋体", "simkai.ttf": "楷体", "deng.ttf": "等线",
    "stkaiti.ttf": "华文楷体", "stfangso.ttf": "华文仿宋", "arial.ttf": "Arial",
    "arialbd.ttf": "Arial·粗", "times.ttf": "Times New Roman",
}


def _fonts_models_dir():
    """可选的 ComfyUI/models/fonts(若已存在则一并扫描，不主动创建)。"""
    try:
        import folder_paths
        d = os.path.join(folder_paths.models_dir, "fonts")
        return d if os.path.isdir(d) else None
    except Exception:
        return None


def _system_font_dirs():
    if os.name == "nt":
        return [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")]
    return [d for d in (
        "/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"),
        "/Library/Fonts", os.path.expanduser("~/Library/Fonts"),
    ) if os.path.isdir(d)]


def _scan_fonts():
    """返回 {显示名: 路径}。用户目录(models/fonts、插件 fonts/)全收；系统字体只收常用。"""
    found, seen = {}, set()

    def _add(d, only_curated):
        try:
            names = sorted(os.listdir(d))
        except Exception:
            return
        for fn in names:
            if os.path.splitext(fn)[1].lower() not in _FONT_EXTS:
                continue
            base = fn.lower()
            if only_curated and base not in _FRIENDLY:
                continue
            if base in seen:
                continue
            seen.add(base)
            disp = _FRIENDLY.get(base, os.path.splitext(fn)[0])
            if disp in found:
                disp = fn  # 同名冲突则退回文件名
            found[disp] = os.path.join(d, fn)

    pd = _plugin_fonts_dir()
    if os.path.isdir(pd):
        _add(pd, False)
    md = _fonts_models_dir()
    if md and os.path.isdir(md):
        _add(md, False)
    for sd in _system_font_dirs():
        _add(sd, True)  # 系统字体只收 _FRIENDLY 里的常用项，避免下拉框上百项
    return found


def _available_fonts():
    return list(_scan_fonts().keys()) or ["默认"]


def _resolve_font(name, size):
    path = _scan_fonts().get(name)
    if path and os.path.isfile(path):
        return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


def _rgb(color, default=(255, 255, 255)):
    try:
        return ImageColor.getrgb(color.strip())
    except Exception:
        return default


class CreateTextWatermark:
    """生成文字水印（输出 图像 + 遮罩）

    渲染文字到透明画布，自动裁到文字大小。IMAGE 为文字颜色、MASK 为文字 alpha，
    一起接到加水印节点的『水印图像 / 水印遮罩』即可。
    """

    DESCRIPTION = (
        "把一段文字渲染成透明文字水印：输出 图像(文字颜色) + 遮罩(文字形状 alpha)，"
        "自动裁到文字大小。\n"
        "两路一起接到『图片/视频添加水印』节点的 水印图像 / 水印遮罩 即可作为文字水印叠加。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文本": ("STRING", {"default": "© Noctyra", "multiline": True, "tooltip": "水印文字，支持多行"}),
                "字体": (_available_fonts(), {"tooltip": "字体(插件 fonts/、models/fonts、系统字体均会扫描)。已内置思源黑体支持中文"}),
                "字号": ("INT", {"default": 72, "min": 4, "max": 1024, "tooltip": "字号(像素)"}),
                "颜色": ("STRING", {"default": "#FFFFFF", "tooltip": "文字颜色，十六进制如 #FFFFFF 或颜色名 white"}),
                "不透明度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "文字整体不透明度(写进遮罩，叠加时再乘一次)"}),
                "描边宽度": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "文字描边粗细，0=无描边。设>0 才显示『描边颜色』"}),
                "描边颜色": ("STRING", {"default": "#000000", "tooltip": "描边颜色(描边宽度>0 时生效)"}),
                "对齐": (["左", "中", "右"], {"default": "左", "tooltip": "多行文字的水平对齐"}),
                "行间距": ("INT", {"default": 8, "min": 0, "max": 256, "tooltip": "多行之间的额外像素间距"}),
                "边距": ("INT", {"default": 16, "min": 0, "max": 512, "tooltip": "文字四周留白(避免描边/斜体被裁切)"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像", "遮罩")
    FUNCTION = "create"
    CATEGORY = "Noctyra/添加水印"

    def create(self, 文本, 字体, 字号, 颜色, 不透明度, 描边宽度, 描边颜色,
               对齐, 行间距, 边距):
        text = 文本 if 文本 != "" else " "
        font = _resolve_font(字体, 字号)
        align = {"左": "left", "中": "center", "右": "right"}[对齐]
        fill = _rgb(颜色, (255, 255, 255)) + (255,)
        stroke = _rgb(描边颜色, (0, 0, 0)) + (255,)

        # 量文字包围盒
        dummy = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        bbox = dummy.multiline_textbbox(
            (0, 0), text, font=font, spacing=行间距, stroke_width=描边宽度, align=align
        )
        l, t, r, b = bbox
        tw, th = max(1, int(round(r - l))), max(1, int(round(b - t)))
        W, H = tw + 2 * 边距, th + 2 * 边距

        canvas = Image.new("RGBA", (int(W), int(H)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.multiline_text(
            (int(round(边距 - l)), int(round(边距 - t))), text, font=font, fill=fill,
            spacing=行间距, align=align, stroke_width=描边宽度, stroke_fill=stroke,
        )

        arr = np.array(canvas).astype(np.float32) / 255.0  # [H,W,4]
        rgb = arr[..., :3]
        alpha = arr[..., 3] * float(不透明度)

        image = torch.from_numpy(rgb).unsqueeze(0)          # [1,H,W,3]
        mask = torch.from_numpy(alpha).unsqueeze(0)         # [1,H,W]
        return (image, mask)


NODE_CLASS_MAPPINGS = {
    "CreateTextWatermark": CreateTextWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CreateTextWatermark": "文字水印生成",
}
