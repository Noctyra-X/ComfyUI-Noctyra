# Third-Party Notices

ComfyUI-Noctyra 的「去除可见水印」节点（Gemini / 豆包 引擎及内嵌模板资源）
移植自以下 MIT 许可的开源项目，特此致谢并保留版权声明。

---

## remove-ai-watermarks

- 项目：https://github.com/wiltodelta/remove-ai-watermarks
- 引用范围：
  - `nodes/dewatermark_gemini.py` —— Gemini 反向 alpha 混合算法（移植自其 `gemini_engine.py`）
  - `nodes/dewatermark_doubao.py` —— 豆包 AIGC 文字条定位/掩膜/修复算法（移植自其 `doubao_engine.py`）
  - `nodes/assets/gemini_bg_48.png`、`nodes/assets/gemini_bg_96.png` —— Gemini 水印 alpha 模板（原样拷贝）

```
MIT License

Copyright (c) 2025 wiltodelta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## GeminiWatermarkTool

remove-ai-watermarks 的 Gemini 引擎本身是对以下 C++ 工具反向 alpha 混合算法的移植：

- 项目：https://github.com/allenk/GeminiWatermarkTool
- 作者：Allen Kuo (allenk)
