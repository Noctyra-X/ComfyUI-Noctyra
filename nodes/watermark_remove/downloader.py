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
隐形水印去除 —— 扩散模型自动下载节点（多下载源）

把上游 remove-ai-watermarks 隐形水印引擎用到的模型按管线打包下载。支持三种
下载源，由用户在节点里自行选择：
- HuggingFace        : 官方源，下载到 HF 缓存，后续 from_pretrained(repo_id) 直接命中。
- HF镜像(hf-mirror)  : 同上但走 https://hf-mirror.com，国内更快；缓存与 repo id 不变。
- ModelScope         : 走 modelscope.cn 直链（无需 SDK）下到本地目录；仅部分模型有，
                       缺失的(如 ctrlregen)自动回退 HF 镜像。

管线与模型：
- default (SDXL img2img):  stabilityai/stable-diffusion-xl-base-1.0
- ctrlregen (CtrlRegen):   SG161222/Realistic_Vision_V4.0_noVAE (基础SD)
                           yepengliu/ctrlregen                 (空间ControlNet+语义IP-Adapter)
                           facebook/dinov2-giant               (图像编码器)
                           stabilityai/sd-vae-ft-mse           (VAE)
"""
import json
import logging
import os
import urllib.parse
import urllib.request
from collections import defaultdict

logger = logging.getLogger("noctyra")

HF_MIRROR = "https://hf-mirror.com"

# 各管线所需仓库（统一用 HF repo id 作为主键）
BUNDLES = {
    "default": ["stabilityai/stable-diffusion-xl-base-1.0"],
    "ctrlregen": [
        "SG161222/Realistic_Vision_V4.0_noVAE",
        "yepengliu/ctrlregen",
        "facebook/dinov2-giant",
        "stabilityai/sd-vae-ft-mse",
    ],
}

# HF repo id -> ModelScope id（仅 ModelScope 上确实存在的；缺失者回退 HF 镜像）
MS_MAP = {
    "stabilityai/stable-diffusion-xl-base-1.0": "AI-ModelScope/stable-diffusion-xl-base-1.0",
    "SG161222/Realistic_Vision_V4.0_noVAE": "AI-ModelScope/Realistic_Vision_V4.0_noVAE",
    "facebook/dinov2-giant": "AI-ModelScope/dinov2-giant",
    "stabilityai/sd-vae-ft-mse": "stabilityai/sd-vae-ft-mse",
    # yepengliu/ctrlregen : ModelScope 无 -> 回退 HF 镜像
}


def _human(nbytes):
    g = nbytes / 1e9
    return f"{g:.2f} GB" if g >= 1.0 else f"{nbytes / 1e6:.1f} MB"


def _select_files(name_size, prefer_fp16=True):
    """从 {文件名: 字节} 精选 from_pretrained 真正需要的文件。

    - 保留所有配置/分词器/调度器等小文件；
    - 权重按文件夹分组择优：fp16 变体 > 默认 safetensors > .bin；
    - diffusers 仓库(含 model_index.json)排除根目录的单文件大 checkpoint；
    - onnx/ckpt/pth 等残渣不在 safetensors/bin 之列，自然排除。
    返回 (保留文件名集合, 总字节)。
    """
    is_diffusers = "model_index.json" in name_size
    keep = set()
    for f in name_size:
        head = f.split("/", 1)[0]
        if f.endswith((".json", ".txt", ".model")) or head in (
            "tokenizer", "tokenizer_2", "scheduler", "feature_extractor",
        ):
            keep.add(f)

    folders = defaultdict(list)
    for f in name_size:
        if f.endswith((".safetensors", ".bin")):
            folders[f.rsplit("/", 1)[0] if "/" in f else ""].append(f)

    for folder, files in folders.items():
        if is_diffusers and folder == "":
            continue
        st_fp16 = [f for f in files if f.endswith(".fp16.safetensors")]
        st = [f for f in files if f.endswith(".safetensors") and ".fp16." not in f]
        binf = [f for f in files if f.endswith(".bin")]
        if prefer_fp16 and st_fp16:
            keep.update(st_fp16)
        elif st:
            keep.update(st)
        elif st_fp16:
            keep.update(st_fp16)
        else:
            keep.update(binf)

    return keep, sum(name_size[f] for f in keep)


# ── ModelScope 直链（无需 modelscope SDK） ─────────────────────────────

def _make_opener(no_proxy):
    """返回 urllib opener；no_proxy=True 时强制直连(忽略环境/注册表代理)。"""
    if no_proxy:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _ms_list_files(ms_id, opener):
    """列 ModelScope 仓库文件，返回 {路径: 字节}。"""
    url = f"https://modelscope.cn/api/v1/models/{ms_id}/repo/files?Revision=master&Recursive=true"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with opener.open(req, timeout=60) as r:
        d = json.load(r)
    files = (d.get("Data") or {}).get("Files") or []
    return {f["Path"]: (f.get("Size") or 0) for f in files if f.get("Type") == "blob"}


def _ms_download_file(ms_id, rel_path, dest_path, opener):
    """从 ModelScope 直链下载单个文件到 dest_path（已存在且大小一致则跳过）。"""
    quoted = urllib.parse.quote(rel_path, safe="/")
    url = f"https://modelscope.cn/api/v1/models/{ms_id}/repo?Revision=master&FilePath={quoted}"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    tmp = dest_path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with opener.open(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest_path)


def _models_base():
    """本地落地根目录：插件自带 models/(去水印扩散模型放这里)。"""
    from .._utils import plugin_models_dir
    return plugin_models_dir()


def _dest_root(hf_repo, base_override=""):
    """统一落地目录：<根>/<HF仓库id>，与下载渠道无关。

    例: stabilityai/stable-diffusion-xl-base-1.0
        -> .../models/stabilityai__stable-diffusion-xl-base-1.0
    """
    base = base_override.strip() or _models_base()
    return os.path.join(base, hf_repo.replace("/", "__"))


class InvisibleWatermarkModelDownloader:
    """隐形水印模型下载器（HF / HF镜像 / ModelScope）

    按所选管线与下载源把模型下到本地。建议先勾选『仅检查不下载』查看各源
    大小，确认后再取消勾选执行下载。
    """

    DESCRIPTION = (
        "把『去除隐形水印』节点所需的扩散模型下到本插件 models/ 目录。\n"
        "建议先勾『仅检查不下载』看各源体积，确认后再取消勾选执行。国内推荐 ModelScope 源 + 『不走代理』。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "管线": (["default (SDXL)", "ctrlregen", "全部"], {"default": "default (SDXL)",
                    "tooltip": "下哪套模型，需与『去除隐形水印』节点的管线一致。default≈7GB；ctrlregen 另需若干模型；全部=都下"}),
                "下载源": (["HuggingFace", "HF镜像(hf-mirror)", "ModelScope"], {"default": "ModelScope",
                    "tooltip": "下载站点。国内首选 ModelScope 或 HF镜像；HuggingFace 官方需能直连"}),
                "不走代理": ("BOOLEAN", {"default": True, "tooltip": "下载时绕过系统/环境代理，省流量(国内大文件建议开)"}),
                "仅检查不下载": ("BOOLEAN", {"default": True, "tooltip": "只列出将下载的文件与总体积，不实际下载。确认无误后关掉再运行"}),
                "精度": (["fp16优先(省空间)", "完整精度"], {"default": "fp16优先(省空间)",
                    "tooltip": "fp16优先=只取半精度权重(体积约一半，去水印够用)；完整精度=连 fp32 一起下"}),
            },
            "optional": {
                "hf_token": ("STRING", {"default": "", "tooltip": "HuggingFace 访问令牌，仅下载 gated/私有库时才需要"}),
                "目标目录": ("STRING", {"default": "", "tooltip": "自定义保存目录，留空=插件 models/ 目录(推荐)"}),
                "自定义SDXL模型": ("STRING", {"default": "", "tooltip": "用别的 SDXL 仓库替换默认底模(填 repo_id)，留空=用默认。仅含 SDXL 的管线生效"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("报告",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Noctyra/水印去除"

    def _repos(self, 管线, 自定义SDXL模型):
        sdxl = 自定义SDXL模型.strip() or BUNDLES["default"][0]
        if 管线.startswith("default"):
            return [sdxl]
        if 管线 == "ctrlregen":
            return list(BUNDLES["ctrlregen"])
        return [sdxl] + BUNDLES["ctrlregen"]

    # ── 单仓库处理（HF / 镜像） ────────────────────────────────────────
    def _do_hf(self, repo, endpoint, prefer_fp16, dry, token, base):
        import huggingface_hub
        from huggingface_hub import HfApi, snapshot_download

        # 镜像：仅靠 snapshot_download 的 endpoint= 参数不会透传到底层文件下载，
        # 须同时改全局 constants.ENDPOINT + 环境变量，否则会去连 huggingface.co。
        _saved_ep = None
        if endpoint:
            _saved_ep = (huggingface_hub.constants.ENDPOINT, os.environ.get("HF_ENDPOINT"))
            huggingface_hub.constants.ENDPOINT = endpoint
            os.environ["HF_ENDPOINT"] = endpoint
        try:
            api = HfApi(endpoint=endpoint) if endpoint else HfApi()
            info = api.repo_info(repo, files_metadata=True, token=token)
            name_size = {s.rfilename: (s.size or 0) for s in info.siblings}
            keep, size = _select_files(name_size, prefer_fp16)
            if dry:
                return size, f"- {repo}  {len(keep)} 文件  {_human(size)}"
            dest_root = _dest_root(repo, base)
            kw = {"allow_patterns": sorted(keep), "token": token, "local_dir": dest_root}
            if endpoint:
                kw["endpoint"] = endpoint
            snapshot_download(repo, **kw)
            return size, f"[OK] {repo}  {_human(size)}\n    -> {dest_root}"
        finally:
            if _saved_ep is not None:
                huggingface_hub.constants.ENDPOINT = _saved_ep[0]
                if _saved_ep[1] is None:
                    os.environ.pop("HF_ENDPOINT", None)
                else:
                    os.environ["HF_ENDPOINT"] = _saved_ep[1]

    # ── 单仓库处理（ModelScope，无 SDK 直链） ──────────────────────────
    def _do_ms(self, repo, prefer_fp16, dry, opener, base):
        ms_id = MS_MAP.get(repo)
        if ms_id is None:
            return None  # 该仓库 ModelScope 无，调用方回退 HF 镜像
        name_size = _ms_list_files(ms_id, opener)
        keep, size = _select_files(name_size, prefer_fp16)
        if dry:
            return size, f"- {repo}  [MS:{ms_id}]  {len(keep)} 文件  {_human(size)}"
        dest_root = _dest_root(repo, base)  # 用 HF 仓库 id 命名，与渠道无关
        root_abs = os.path.abspath(dest_root)
        for i, rel in enumerate(sorted(keep), 1):
            dest = os.path.join(dest_root, rel.replace("/", os.sep))
            # 防目录穿越：确保落点仍在 dest_root 内
            if os.path.commonpath([root_abs, os.path.abspath(dest)]) != root_abs:
                logger.warning(f"跳过异常路径(疑似穿越): {rel}")
                continue
            if os.path.exists(dest) and os.path.getsize(dest) == name_size[rel]:
                continue
            logger.info(f"ModelScope 下载 {ms_id} :: {rel} ({i}/{len(keep)})")
            _ms_download_file(ms_id, rel, dest, opener)
        return size, f"[OK] {repo}  [MS:{ms_id}]  {_human(size)}\n    -> {dest_root}"

    def run(self, 管线, 下载源, 不走代理, 仅检查不下载, 精度, hf_token="", 目标目录="", 自定义SDXL模型=""):
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            return ("未安装 huggingface_hub，请先 pip install huggingface_hub",)

        token = hf_token.strip() or None
        base = 目标目录.strip()
        prefer_fp16 = 精度.startswith("fp16")
        dry = 仅检查不下载
        repos = self._repos(管线, 自定义SDXL模型)

        is_ms = 下载源 == "ModelScope"
        endpoint = HF_MIRROR if 下载源.startswith("HF镜像") else None
        opener = _make_opener(不走代理)

        # 不走代理：临时清掉代理环境变量，使 huggingface_hub(requests) 也直连
        _saved_proxy = {}
        if 不走代理:
            for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
                if k in os.environ:
                    _saved_proxy[k] = os.environ.pop(k)

        lines = [f"管线={管线} 源={下载源} 代理={'否' if 不走代理 else '是'} 精度={精度} {'[仅检查]' if dry else '[下载]'}"]
        lines.append(f"落地根目录={base or _models_base()}")
        lines.append("-" * 44)

        total = 0
        try:
            for repo in repos:
                try:
                    if is_ms:
                        res = self._do_ms(repo, prefer_fp16, dry, opener, base)
                        if res is None:
                            # ModelScope 无此仓库 -> 回退 HF 镜像
                            size, line = self._do_hf(repo, HF_MIRROR, prefer_fp16, dry, token, base)
                            line += "  (ModelScope无, 回退HF镜像)"
                        else:
                            size, line = res
                    else:
                        size, line = self._do_hf(repo, endpoint, prefer_fp16, dry, token, base)
                    total += size
                    lines.append(line)
                except Exception as e:
                    logger.exception(f"模型下载失败: {repo}")
                    lines.append(f"[X] {repo}  失败: {repr(e)[:120]}")
        finally:
            os.environ.update(_saved_proxy)  # 恢复代理环境变量

        lines.append("-" * 44)
        lines.append(f"{'预计下载' if dry else '已处理'}合计 约 {_human(total)}")
        if dry:
            lines.append("确认后取消勾选『仅检查不下载』再次执行即开始下载。")

        report = "\n".join(lines)
        from .._utils import safe_print
        safe_print("[Noctyra] 模型下载\n" + report)
        return (report,)


NODE_CLASS_MAPPINGS = {
    "InvisibleWatermarkModelDownloader": InvisibleWatermarkModelDownloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InvisibleWatermarkModelDownloader": "隐形水印模型下载器",
}
