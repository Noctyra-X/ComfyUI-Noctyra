# ComfyUI-Noctyra — 去除水印节点子包
# GPL-3.0 (见仓库 LICENSE)
"""
去除水印 / 溯源节点：
- region    : 去除可见水印（通用区域修复，cv2 / LaMa）
- gemini    : 去除 Gemini 星标水印（反向 alpha 还原）
- doubao    : 去除豆包 AIGC 文字水印
- invisible : 去除隐形水印（SDXL / CtrlRegen）
- identify  : AI 水印溯源（C2PA / 类型识别）
- downloader: 隐形水印模型下载器
"""
from . import region, gemini, doubao, invisible, identify, downloader

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _mod in (region, gemini, doubao, invisible, identify, downloader):
    NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
