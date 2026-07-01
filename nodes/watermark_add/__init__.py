# ComfyUI-Noctyra — 添加水印节点子包
# GPL-3.0 (见仓库 LICENSE)
"""
添加水印节点：
- image_wm  : 添加图像水印（定位）
- grid      : 网格水印（随机偏移防重叠）
- video_wm  : 视频加水印（定位·帧区间+淡入淡出）
- fullscreen: 全屏平铺水印（图片 / 视频）
- text      : 生成文字水印（输出 图像+遮罩）
- trim      : 水印去空白裁剪（自动包围）
"""
from . import image_wm, grid, video_wm, fullscreen, text, trim

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _mod in (image_wm, grid, video_wm, fullscreen, text, trim):
    NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))
