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
ComfyUI-Noctyra 插件

自定义节点集合：
- 图像处理节点（无元数据保存）
- 水印节点（含网格水印，随机偏移防重叠）
- 51EasyAI API 节点（文生图、图生图）

配套的模型管理器已独立为 ComfyUI-Noctyra-Manager 插件。
"""

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__version__ = "1.0.0"

# 前端扩展目录：按下拉/开关动态显隐控制参数（见 web/noctyra_dynamic.js）
WEB_DIRECTORY = "./web"

print(f"\033[34m[Noctyra]\033[0m v{__version__} \033[92m已加载\033[0m {len(NODE_CLASS_MAPPINGS)} 个节点")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
