# ComfyUI-Noctyra
# Copyright (C) 2026 Noctyra — GPL-3.0 (见仓库 LICENSE)

"""
把 vendored 的 remove-ai-watermarks (MIT) 包挂到 sys.path，供各封装节点导入。

import 时调用 ensure() 即可；重复调用安全。详见 THIRD_PARTY_NOTICES.md。
"""
import os
import sys

_VENDOR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor")


def ensure():
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    return _VENDOR_DIR
