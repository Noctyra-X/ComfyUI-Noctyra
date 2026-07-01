"""
水印处理相关节点
"""
import numpy as np
import torch
from PIL import Image, ImageDraw


class AddImageWatermark:
    """添加图像水印节点（支持遮罩）

    在 ZML_添加图像水印的基础上新增 MASK 输入。
    当提供水印图像的遮罩时，遮罩会作为水印的 alpha 通道，
    使得水印图像的背景区域真正透明，而不是直接整张贴到背景图上。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "水印大小比例": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 2.0, "step": 0.01}),
                "不透明度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "位置": (["左上", "中上", "右上", "左中", "居中", "右中", "左下", "中下", "右下", "全屏"],),
                "水平边距": ("INT", {"default": 30, "min": 0, "max": 4096}),
                "垂直边距": ("INT", {"default": 30, "min": 0, "max": 4096}),
                "全屏水印旋转角度": ("INT", {"default": -30, "min": -360, "max": 360}),
                "全屏水印密度": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1}),
                "全屏水印间距": ("INT", {"default": 10, "min": 0, "max": 200}),
                "反转遮罩": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "水印图像": ("IMAGE",),
                "水印遮罩": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "add_watermark"
    CATEGORY = "Noctyra/图片"

    @staticmethod
    def _tensor_to_pil(tensor):
        arr = np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _pil_to_tensor(pil_image):
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def _mask_to_pil_l(mask_tensor, target_size):
        """把 MASK 张量（[H,W] 或 [1,H,W]）转换成与水印同尺寸的 L 模式 PIL 图像。
        ComfyUI 约定 MASK 中 1.0 表示选中区域；这里把它当作水印的不透明度。"""
        m = mask_tensor
        if m.dim() == 3:
            m = m[0]
        arr = np.clip(m.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        mask_pil = Image.fromarray(arr, mode="L")
        if mask_pil.size != target_size:
            mask_pil = mask_pil.resize(target_size, Image.Resampling.LANCZOS)
        return mask_pil

    def add_watermark(
        self,
        图像,
        水印大小比例=0.25,
        不透明度=1.0,
        位置="右下",
        水平边距=30,
        垂直边距=30,
        全屏水印旋转角度=-30,
        全屏水印密度=1.0,
        全屏水印间距=10,
        反转遮罩=False,
        水印图像=None,
        水印遮罩=None,
    ):
        # 准备水印 PIL 图像（RGBA），并把 MASK 应用到 alpha 通道
        if 水印图像 is not None:
            try:
                watermark_pil = self._tensor_to_pil(水印图像[0]).convert("RGBA")

                if 水印遮罩 is not None:
                    # 取第一张 mask 与水印图像匹配
                    mask_t = 水印遮罩[0] if 水印遮罩.dim() == 3 else 水印遮罩
                    mask_l = self._mask_to_pil_l(mask_t, watermark_pil.size)
                    if 反转遮罩:
                        mask_l = mask_l.point(lambda p: 255 - p)
                    r, g, b, _ = watermark_pil.split()
                    watermark_pil = Image.merge("RGBA", (r, g, b, mask_l))
            except Exception as e:
                print(f"[Noctyra] 水印图像处理失败: {e}")
                watermark_pil = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
                draw = ImageDraw.Draw(watermark_pil)
                draw.rectangle([0, 0, 100, 100], fill=(0, 0, 0, 100))
                draw.text((20, 40), "Watermark", fill=(255, 255, 255, 200))
        else:
            watermark_pil = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
            draw = ImageDraw.Draw(watermark_pil)
            draw.rectangle([0, 0, 100, 100], fill=(0, 0, 0, 100))
            draw.text((20, 40), "Watermark", fill=(255, 255, 255, 200))

        processed_images = []
        for img_tensor in 图像:
            pil_image = self._tensor_to_pil(img_tensor).convert("RGBA")
            img_width, img_height = pil_image.size

            # 等比缩放水印
            base_size = min(img_width, img_height)
            target_size = int(base_size * 水印大小比例)
            wm_width, wm_height = watermark_pil.size
            ratio = min(target_size / wm_width, target_size / wm_height)
            new_width = max(1, int(wm_width * ratio))
            new_height = max(1, int(wm_height * ratio))
            resized_watermark = watermark_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            wm_width, wm_height = resized_watermark.size

            # 调整不透明度（在已应用 mask 的 alpha 上再乘一遍）
            if 不透明度 < 1.0:
                r, g, b, a = resized_watermark.split()
                a = a.point(lambda p: int(p * 不透明度))
                resized_watermark = Image.merge("RGBA", (r, g, b, a))

            if 位置 == "全屏":
                try:
                    rot_img = resized_watermark.rotate(
                        全屏水印旋转角度, expand=True, resample=Image.Resampling.BICUBIC
                    )
                    r_width, r_height = rot_img.size
                    sx = max(1, int((r_width + 水平边距) / 全屏水印密度))
                    sy = max(1, int((r_height + 垂直边距) / 全屏水印密度))
                    offset, row_idx = sx // 2, 0
                    for y in range(-r_height, img_height, sy):
                        start_x = -r_width + (offset if (row_idx % 2) != 0 else 0)
                        for x in range(start_x, img_width, sx):
                            if 0 <= x + r_width and 0 <= y + r_height:
                                pil_image.paste(rot_img, (x, y), rot_img)
                        row_idx += 1
                except Exception as e:
                    print(f"[Noctyra] 全屏水印处理失败: {e}")
            else:
                x_pos = (
                    水平边距 if "左" in 位置
                    else (img_width - wm_width - 水平边距 if "右" in 位置
                          else (img_width - wm_width) // 2)
                )
                y_pos = (
                    垂直边距 if "上" in 位置
                    else (img_height - wm_height - 垂直边距 if "下" in 位置
                          else (img_height - wm_height) // 2)
                )
                x_pos = max(0, min(x_pos, img_width - 1))
                y_pos = max(0, min(y_pos, img_height - 1))
                pil_image.paste(resized_watermark, (x_pos, y_pos), resized_watermark)

            # 转回 RGB，避免下游不支持 RGBA
            pil_image = pil_image.convert("RGB")
            processed_images.append(self._pil_to_tensor(pil_image))

        return (torch.cat(processed_images, dim=0),)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "AddImageWatermark": AddImageWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddImageWatermark": "添加图像水印（支持遮罩）",
}
