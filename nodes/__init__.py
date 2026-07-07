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
Noctyra 节点模块

- image / video      : 无元数据保存
- easyai_api         : AI 服务节点
- watermark_add/     : 添加水印（图像/网格/视频/全屏/文字）
- watermark_remove/  : 去除水印 + 溯源 + 模型下载
- motion/            : 视频转 3D 骨骼动作（GVHMR+HaMeR → BVH/GLB，sidecar 跑）
"""
import logging
import sys

# 统一控制台输出：所有 logging.getLogger("noctyra") 的消息自动加 [Noctyra] 前缀。
_nlog = logging.getLogger("noctyra")
if not getattr(_nlog, "_noctyra_configured", False):
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[Noctyra] %(message)s"))
    _nlog.addHandler(_h)
    _nlog.setLevel(logging.INFO)
    _nlog.propagate = False
    _nlog._noctyra_configured = True

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 各子模块独立 try：缺某个依赖(如 opencv)只禁用它自己的节点，不连累其它子模块
for _name in ("image", "video", "text", "watermark_add", "watermark_remove", "load_switch", "seamless_preview"):  # motion→Noctyra-Mocap, easyai_api→Noctyra-EasyAI
    try:
        _mod = __import__(f"{__name__}.{_name}", fromlist=[_name])
    except Exception as _e:
        _nlog.warning(f"子模块 {_name} 加载失败(其节点不可用): {_e}")
        continue
    NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
