# ComfyUI-Noctyra

ComfyUI 自定义节点合集：水印处理、无元数据保存、带开关的加载节点。

## 节点

**加水印**
- 图片 / 视频加水印：按位置、全屏平铺、网格（行列密度可调）
- 文字水印生成、遮罩裁剪

**去水印**
- 去可见水印：豆包 AIGC 文字条、Gemini 星标、任意区域修复
- 去隐形水印：SDXL / CtrlRegen
- AI 溯源识别：C2PA / 水印类型识别

**无元数据保存**
- 保存图片 / 视频 / 文本，均不写入任何元数据

**加载开关**
- 加载图片 / 视频 / 音频，带启用开关（关闭即跳过，无需拆线）

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Noctyra-X/ComfyUI-Noctyra.git
pip install -r ComfyUI-Noctyra/requirements.txt
```

重启 ComfyUI 即可。去水印节点首次使用时按需下载模型。

> 视频转 3D 骨骼动作（Mocap）已独立为 [ComfyUI-Mocap](https://github.com/Noctyra-X/ComfyUI-Mocap)。

## 致谢

去可见水印节点部分移植自 MIT 项目 [remove-ai-watermarks](https://github.com/wiltodelta/remove-ai-watermarks)，详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 许可证

[GPL-3.0-or-later](LICENSE)
