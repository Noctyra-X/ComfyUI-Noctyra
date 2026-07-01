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
视频处理相关节点
"""
import logging
import os
import random

import folder_paths
from comfy_api.util import VideoCodec, VideoContainer

logger = logging.getLogger("noctyra")


def _video_dims(video):
    """校验 VIDEO 并返回 (w, h)。None/空/损坏视频给出清晰中文错误,不裸崩。"""
    if video is None:
        raise RuntimeError("VIDEO 输入为空(上游未产出视频)")
    try:
        w, h = video.get_dimensions()
    except Exception as e:
        raise RuntimeError(f"读取视频失败(可能是空或损坏的视频): {e}")
    if not w or not h or w <= 0 or h <= 0:
        raise RuntimeError(f"视频尺寸异常({w}x{h}),无法保存/预览")
    return w, h


class SaveVideoNoMetadata:
    """保存视频（无元数据）节点

    与 ComfyUI 自带 SaveVideo 等价，但强制不写入 prompt / workflow 元数据。
    无论是否启动 --disable-metadata，都不会向容器（mp4 udta / mkv tag）写任何 JSON。
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    DESCRIPTION = (
        "保存视频,并强制不写入 ComfyUI 的 prompt / workflow 元数据(mp4 udta / mkv tag 均不注入 JSON),"
        "无论是否以 --disable-metadata 启动均生效。注:仅阻止 ComfyUI 注入,不清除源视频本身已有的标签。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "要保存的视频"}),
                "filename_prefix": ("STRING", {"default": "video/Clean_Video", "tooltip": "输出路径/文件名前缀(可含子目录)，会自动追加序号"}),
                "format": (VideoContainer.as_input(), {"default": "auto", "tooltip": "封装容器格式，auto=按编码自动选(通常 mp4)"}),
                "codec": (VideoCodec.as_input(), {"default": "auto", "tooltip": "视频编码器，auto=自动选择"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "Noctyra/视频"

    def save_video(self, video, filename_prefix="video/Clean_Video", format="auto", codec="auto"):
        width, height = _video_dims(video)
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix,
                self.output_dir,
                width,
                height,
            )
        )

        ext = VideoContainer.get_extension(format) or "mp4"
        file = f"{filename}_{counter:05}_.{ext}"

        # metadata=None 强制跳过 prompt / extra_pnginfo 写入
        video.save_to(
            os.path.join(full_output_folder, file),
            format=VideoContainer(format),
            codec=codec,
            metadata=None,
        )

        return {
            "ui": {
                "images": [{"filename": file, "subfolder": subfolder, "type": self.type}],
                "animated": (True,),
            }
        }


class PreviewVideo:
    """预览视频节点:将输入的 VIDEO 在节点内直接播放。视频保存至临时目录(随 ComfyUI 临时
    文件清理,不写入输出目录),作用等同于 PreviewImage 之于图像。"""

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(5))

    DESCRIPTION = "预览视频。将输入的视频在节点内直接播放,文件保存至临时目录(不写入输出目录)。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO", {"tooltip": "待预览的视频"}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "Noctyra/视频"

    @classmethod
    def IS_CHANGED(cls, video):
        return float("nan")   # 预览语义:每次都重新落盘+刷新,不走缓存

    def preview(self, video):
        width, height = _video_dims(video)
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            "noctyra_preview" + self.prefix_append, self.output_dir, width, height)
        file = f"{filename}_{counter:05}_.mp4"
        # 强制 mp4(浏览器可播);存 temp 不写元数据
        video.save_to(
            os.path.join(full_output_folder, file),
            format=VideoContainer("mp4"), codec="auto", metadata=None)
        return {"ui": {"images": [{"filename": file, "subfolder": subfolder, "type": self.type}],
                       "animated": (True,)}}


NODE_CLASS_MAPPINGS = {
    "SaveVideoNoMetadata": SaveVideoNoMetadata,
    "NoctyraPreviewVideo": PreviewVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveVideoNoMetadata": "Save Video (No Metadata)",
    "NoctyraPreviewVideo": "预览视频",
}
