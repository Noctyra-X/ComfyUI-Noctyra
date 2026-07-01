# ComfyUI-Noctyra — 带开关的加载节点(图片 / 视频 / 音频)
# Copyright (C) 2026 Noctyra — GPL-3.0(见 LICENSE)
#
# 在官方「加载图片/视频/音频」基础上加一个布尔开关 enabled:
#   - 开:正常从输入目录读取素材并输出(IMAGE/MASK、VIDEO、AUDIO),连线照常工作。
#   - 关:对每个输出返回 ExecutionBlocker → 下游节点被跳过(不执行),等于「不加载」。
# 用途:在工作流里用一个开关挂/断某路素材,而不必删连线或改图。
import hashlib
import os

import folder_paths
from comfy_execution.graph_utils import ExecutionBlocker

_SWITCH = ("BOOLEAN", {
    "default": True,
    "label_on": "加载",
    "label_off": "关闭",
    "tooltip": "开启:正常读取并输出;关闭:阻断下游(该路素材不加载,依赖此输出的节点不执行)",
})


def _files(content_types):
    d = folder_paths.get_input_directory()
    os.makedirs(d, exist_ok=True)
    fs = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
    return sorted(folder_paths.filter_files_content_types(fs, content_types))


def _hash_file(name):
    try:
        p = folder_paths.get_annotated_filepath(name)
        m = hashlib.sha256()
        with open(p, "rb") as f:
            m.update(f.read())
        return m.digest().hex()
    except Exception:
        return ""


# ============================ 加载图片(开关) ============================
class NoctyraLoadImageSwitch:
    DESCRIPTION = ("加载图像,带启用开关。开关开启时正常读取并输出 IMAGE / MASK;"
                   "关闭时阻断下游,依赖此输出的节点不执行(用于按需挂接或断开该路输入)。")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "enabled": _SWITCH,
            "image": (_files(["image"]), {"image_upload": True}),
        }}

    CATEGORY = "Noctyra/加载"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "run"

    def run(self, enabled, image):
        if not enabled:
            return (ExecutionBlocker(None), ExecutionBlocker(None))
        # 复用官方 LoadImage 的加载逻辑(忠实一致)
        from nodes import LoadImage
        return LoadImage().load_image(image)

    @classmethod
    def IS_CHANGED(s, enabled, image):
        return "disabled" if not enabled else _hash_file(image)

    @classmethod
    def VALIDATE_INPUTS(s, enabled, image):
        if not enabled:
            return True   # 关闭时不校验文件(允许留空/无效选择)
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"
        return True


# ============================ 加载视频(开关) ============================
class NoctyraLoadVideoSwitch:
    DESCRIPTION = ("加载视频,带启用开关。开关开启时正常读取并输出 VIDEO;"
                   "关闭时阻断下游,依赖此输出的节点不执行(用于按需挂接或断开该路输入)。")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "enabled": _SWITCH,
            "file": (_files(["video"]), {"video_upload": True}),
        }}

    CATEGORY = "Noctyra/加载"
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "run"

    def run(self, enabled, file):
        if not enabled:
            return (ExecutionBlocker(None),)
        from comfy_api.latest import InputImpl
        path = folder_paths.get_annotated_filepath(file)
        return (InputImpl.VideoFromFile(path),)

    @classmethod
    def IS_CHANGED(s, enabled, file):
        if not enabled:
            return "disabled"
        try:
            return os.path.getmtime(folder_paths.get_annotated_filepath(file))
        except Exception:
            return ""

    @classmethod
    def VALIDATE_INPUTS(s, enabled, file):
        if not enabled:
            return True
        if not folder_paths.exists_annotated_filepath(file):
            return f"Invalid video file: {file}"
        return True


# ============================ 加载音频(开关) ============================
class NoctyraLoadAudioSwitch:
    DESCRIPTION = ("加载音频,带启用开关。开关开启时正常读取并输出 AUDIO;"
                   "关闭时阻断下游,依赖此输出的节点不执行(用于按需挂接或断开该路输入)。")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "enabled": _SWITCH,
            "audio": (_files(["audio", "video"]), {"audio_upload": True}),
        }}

    CATEGORY = "Noctyra/加载"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"

    def run(self, enabled, audio):
        if not enabled:
            return (ExecutionBlocker(None),)
        from comfy_extras.nodes_audio import load as _load_audio
        path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = _load_audio(path)
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)

    @classmethod
    def IS_CHANGED(s, enabled, audio):
        return "disabled" if not enabled else _hash_file(audio)

    @classmethod
    def VALIDATE_INPUTS(s, enabled, audio):
        if not enabled:
            return True
        if not folder_paths.exists_annotated_filepath(audio):
            return f"Invalid audio file: {audio}"
        return True


NODE_CLASS_MAPPINGS = {
    "NoctyraLoadImageSwitch": NoctyraLoadImageSwitch,
    "NoctyraLoadVideoSwitch": NoctyraLoadVideoSwitch,
    "NoctyraLoadAudioSwitch": NoctyraLoadAudioSwitch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NoctyraLoadImageSwitch": "加载图片(开关)",
    "NoctyraLoadVideoSwitch": "加载视频(开关)",
    "NoctyraLoadAudioSwitch": "加载音频(开关)",
}
