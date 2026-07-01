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
# 本节点封装 vendored MIT 项目 remove-ai-watermarks 的 identify 模块
# (Copyright (c) 2025 wiltodelta)。详见仓库 THIRD_PARTY_NOTICES.md。
# ---------------------------------------------------------------------------

"""
AI 溯源鉴定节点

聚合 C2PA 内容凭证、IPTC "Made with AI" 标签、嵌入生成参数、SynthID 元数据
代理、可见 Gemini 闪光等信号，给出一张图的来源判定与水印清单。

两种输入(二选一)：
- 接 IMAGE：只能做**像素域检测**(可见水印)。ComfyUI 的 IMAGE 是解码后的像素，
  不含元数据，C2PA/EXIF/IPTC 这些已随解码丢失。
- 填**图片路径**：读原始文件做**完整溯源**(元数据 + 像素域)。只填文件名时
  自动到 ComfyUI input/ 目录找。
"""
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .._vendor import ensure as _ensure_vendor

logger = logging.getLogger("noctyra")


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif", ".avif", ".heic")


def _resolve_path(p):
    """解析图片路径：绝对/相对存在则用；只给文件名则到 ComfyUI input/ 找。"""
    p = p.strip().strip('"')
    if not p:
        return None
    if os.path.isfile(p):
        return p
    try:
        import folder_paths
        cand = os.path.join(folder_paths.get_input_directory(), p)
        if os.path.isfile(cand):
            return cand
    except Exception:
        pass
    return None


def _trace_image_path(prompt, node_id, depth=0):
    """沿工作流反查图像来源的原始文件路径或 URL（best-effort，最多 6 跳）。"""
    if depth > 6 or prompt is None:
        return None
    node = prompt.get(str(node_id)) or prompt.get(node_id)
    if not isinstance(node, dict):
        return None
    inputs = node.get("inputs", {}) or {}
    ctype = node.get("class_type", "") or ""

    # 已知图片加载器：用 ComfyUI 的注解路径解析
    if "LoadImage" in ctype:
        fn = inputs.get("image")
        if isinstance(fn, str):
            try:
                import folder_paths
                p = folder_paths.get_annotated_filepath(fn)
                if p and os.path.isfile(p):
                    return p
            except Exception:
                pass

    # 通用：任何 URL 或本地图片路径字符串输入
    for v in inputs.values():
        if isinstance(v, str):
            vs = v.strip()
            if vs.startswith(("http://", "https://")):
                return vs
            if vs.lower().endswith(_IMG_EXTS):
                cand = _resolve_path(vs)
                if cand:
                    return cand

    # 沿连线(形如 [上游id, 槽位])递归往上游找
    for v in inputs.values():
        if isinstance(v, list) and len(v) == 2:
            r = _trace_image_path(prompt, v[0], depth + 1)
            if r:
                return r
    return None


def _auto_source_path(prompt, unique_id):
    """从本节点的『图像』输入连线，反查上游原始图片路径/URL。"""
    try:
        me = prompt.get(str(unique_id)) or prompt.get(unique_id)
        link = (me or {}).get("inputs", {}).get("图像")
        if isinstance(link, list) and link:
            return _trace_image_path(prompt, link[0], 0)
    except Exception as e:
        logger.debug(f"反查图像来源失败: {e}")
    return None


def _download_to_temp(url):
    """下载网络图片到临时文件，返回路径(失败 None)。"""
    import urllib.request
    try:
        ext = os.path.splitext(url.split("?")[0])[1].lower()
        if ext not in _IMG_EXTS:
            ext = ".img"
        tmp = os.path.join(tempfile.gettempdir(), f"noctyra_url_{int(time.time()*1000)}{ext}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        max_bytes = 50 * 1024 * 1024  # 50MB 上限，防恶意/超大 URL 撑爆内存与磁盘
        with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
            total = 0
            while True:
                chunk = r.read(1 << 20)  # 1MB
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise RuntimeError(f"下载内容超过上限 {max_bytes} 字节，已中止: {url}")
                f.write(chunk)
        return tmp
    except Exception as e:
        logger.warning(f"下载网络图片失败: {e}")
        return None


class IdentifyAIProvenance:
    """AI 溯源鉴定（C2PA / 类型识别）

    直接连『图像』即可：节点会自动反查上游(Load Image / URL 加载等)的原始文件路径，
    读原文件做完整溯源(元数据 + 像素域)。反查不到时退回像素域检测。也可手填『图片路径』。
    """

    DESCRIPTION = (
        "鉴定一张图是否 AI 生成、出自哪个平台：聚合 C2PA 内容凭证、IPTC『Made with AI』、"
        "嵌入生成参数、SynthID 元数据、可见 Gemini 闪光等信号。\n"
        "直接连『图像』即可——会自动反查上游 Load Image/URL 的原始文件读元数据；"
        "连不到时退回纯像素检测(元数据已随解码丢失，结果较弱)。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "图像": ("IMAGE", {"tooltip": "待鉴定图。会自动反查它的上游原始文件路径以读元数据。与『图片路径』二选一"}),
                "图片路径": ("STRING", {"forceInput": True, "tooltip": "原图路径/文件名字符串输入(读原文件做完整溯源)。只给文件名时到 ComfyUI input/ 找。与『图像』二选一"}),
                "检测像素域": ("BOOLEAN", {"default": True, "tooltip": "是否额外做像素级可见/隐形水印检测(关掉则只看元数据，更快)"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("报告", "是否AI生成", "平台")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Noctyra/水印去除"

    def run(self, 图像=None, 图片路径="", 检测像素域=True, unique_id=None, prompt=None):
        _ensure_vendor()
        try:
            from remove_ai_watermarks.identify import identify
        except ImportError as e:
            return (f"vendored 模块导入失败: {e}", "未知", "")

        pixel_only = False
        tmp = None
        # 1) 手填路径优先
        path = _resolve_path(图片路径)
        # 2) 自动反查上游图像来源(原始文件/URL)
        if path is None:
            src = _auto_source_path(prompt, unique_id)
            if src:
                if src.startswith(("http://", "https://")):
                    tmp = _download_to_temp(src)
                    if tmp:
                        path = tmp
                elif os.path.isfile(src):
                    path = src
        # 3) 仍无路径但有 IMAGE：仅像素域(元数据已随解码丢失)
        if path is None and 图像 is not None:
            arr = np.clip(图像[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            tmp = os.path.join(tempfile.gettempdir(), f"noctyra_idf_{int(time.time()*1000)}.png")
            Image.fromarray(arr).save(tmp)
            path = tmp
            pixel_only = True
        if path is None:
            return ("未提供有效输入：请连接『图像』或填『图片路径』。", "未知", "")

        try:
            report = identify(Path(path), check_visible=检测像素域, check_invisible=检测像素域)
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

        verdict_map = {True: "是(AI生成)", False: "否(非AI)", None: "未知"}
        verdict = verdict_map[report.is_ai_generated]
        platform = report.platform or ""

        # 标记名取简短形式(去掉冗长括号说明)
        def _short(w):
            return w.split("(")[0].strip() if "(" in w else w

        lines = [f"判定: {verdict}  平台: {platform or '未确定'}"]
        if report.watermarks:
            lines.append("标记: " + " · ".join(_short(w) for w in report.watermarks))
        elif report.is_ai_generated is None:
            tail = "(仅像素检测，元数据已丢失)" if pixel_only else "(可能被剥除，或无本地检测器)"
            lines.append("未发现可读的 AI 信号 " + tail)
        if report.integrity_clashes:
            lines.append("⚠ 溯源信号矛盾: " + "；".join(report.integrity_clashes))

        report_str = "\n".join(lines)
        from .._utils import safe_print
        src = os.path.basename(path)
        safe_print(f"[Noctyra] AI水印识别 [{src}] {report_str.replace(chr(10), '  ')}")
        return {"ui": {"text": [report_str]}, "result": (report_str, verdict, platform)}


NODE_CLASS_MAPPINGS = {
    "IdentifyAIProvenance": IdentifyAIProvenance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IdentifyAIProvenance": "AI水印识别（C2PA/类型识别）",
}
