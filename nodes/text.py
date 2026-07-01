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
文本相关节点:保存文本到输出目录。
"""
import os

import folder_paths


class SaveText:
    """保存文本节点:把字符串写成文件(默认 .txt,UTF-8)到输出目录。"""

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    DESCRIPTION = (
        "将文本保存为文件到输出目录(默认 .txt,UTF-8)。可连接任意输出 STRING 的节点,或直接在框内输入。\n"
        "新建模式:每次按序号生成新文件;追加模式:写到固定文件名末尾(每次追加一行)。支持自定义扩展名(txt/md/json/csv…)。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "要保存的文本(可连接 STRING 输出,或直接输入)"}),
                "filename_prefix": ("STRING", {"default": "text/ComfyUI",
                                    "tooltip": "输出文件名前缀(可含子目录);新建模式自动追加序号"}),
            },
            "optional": {
                "extension": ("STRING", {"default": "txt",
                              "tooltip": "扩展名(不含点),如 txt / md / json / csv"}),
                "append": ("BOOLEAN", {"default": False, "label_on": "追加", "label_off": "新建",
                           "tooltip": "新建:每次按序号建新文件;追加:写到固定文件名(前缀.扩展名)末尾,每次追加一行"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Noctyra/文本"

    def save(self, text, filename_prefix="text/ComfyUI", extension="txt", append=False):
        text = "" if text is None else str(text)
        ext = (extension or "txt").lstrip(".").strip() or "txt"

        if append:
            # 追加模式:固定文件名(前缀.扩展名,不加序号),末尾追加 + 换行
            path = os.path.join(self.output_dir, f"{filename_prefix}.{ext}")
            os.makedirs(os.path.dirname(path) or self.output_dir, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        else:
            # 新建模式:复用 ComfyUI 命名,自动追加序号,每次新文件
            full_output_folder, filename, counter, _subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix, self.output_dir)
            os.makedirs(full_output_folder, exist_ok=True)
            path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.{ext}")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)

        return {"ui": {"text": [text]}, "result": (path,)}


NODE_CLASS_MAPPINGS = {
    "SaveText": SaveText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveText": "保存文本",
}
