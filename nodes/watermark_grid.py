"""
全屏网格水印节点 - 独立控制行列密度
"""
import numpy as np
import torch
import random
from PIL import Image, ImageDraw


class AddGridWatermark:
    """添加全屏网格水印（独立控制行列密度）

    分别控制水平方向和垂直方向的密度，可以独立调整行列数量。
    包围盒基于水印实际尺寸计算，水平和垂直互不干扰。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "水印大小比例": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 2.0, "step": 0.01}),
                "不透明度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "水平密度": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 5.0, "step": 0.1, "description": "每行水印数量系数"}),
                "垂直密度": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 5.0, "step": 0.1, "description": "每列水印数量系数"}),
                "包围盒倍数": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1, "description": "1.0=紧贴，1.5=半间距，2.0=宽松"}),
                "最小包围盒比例": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01, "description": "相对于图片短边的最小包围盒大小"}),
                "最大随机偏移": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "description": "相对于剩余空间的偏移比例，1.0=可触及包围盒边缘"}),
                "旋转角度": ("INT", {"default": -30, "min": -360, "max": 360}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
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
        水印大小比例=0.12,
        不透明度=0.5,
        水平密度=1.0,
        垂直密度=1.0,
        包围盒倍数=1.5,
        最小包围盒比例=0.0,
        最大随机偏移=0.8,
        旋转角度=-30,
        随机种子=0,
        反转遮罩=False,
        水印图像=None,
        水印遮罩=None,
    ):
        # 准备水印 PIL 图像（RGBA）
        if 水印图像 is not None:
            try:
                watermark_pil = self._tensor_to_pil(水印图像[0]).convert("RGBA")

                if 水印遮罩 is not None:
                    mask_t = 水印遮罩[0] if 水印遮罩.dim() == 3 else 水印遮罩
                    mask_l = self._mask_to_pil_l(mask_t, watermark_pil.size)
                    if 反转遮罩:
                        mask_l = mask_l.point(lambda p: 255 - p)
                    r, g, b, _ = watermark_pil.split()
                    watermark_pil = Image.merge("RGBA", (r, g, b, mask_l))
            except Exception as e:
                print(f"[Noctyra] 水印图像处理失败: {e}")
                watermark_pil = self._create_default_watermark()
        else:
            watermark_pil = self._create_default_watermark()

        # 按分辨率分组，同分辨率同位置
        resolution_groups = {}
        for img_idx, img_tensor in enumerate(图像):
            h, w = img_tensor.shape[0], img_tensor.shape[1]
            key = (w, h)
            if key not in resolution_groups:
                resolution_groups[key] = []
            resolution_groups[key].append((img_idx, img_tensor))

        processed_images = [None] * len(图像)

        for (img_width, img_height), group in resolution_groups.items():
            # 设置随机种子
            if 随机种子 > 0:
                seed = 随机种子
            else:
                seed = (img_width * 73856093) ^ (img_height * 19349663)
                seed = seed & 0xffffffffffffffff
            rng = random.Random(seed)

            # 处理主图
            base_pil = self._tensor_to_pil(group[0][1]).convert("RGBA")

            # 缩放水印
            base_size = min(img_width, img_height)
            target_size = int(base_size * 水印大小比例)
            wm_width, wm_height = watermark_pil.size
            ratio = min(target_size / wm_width, target_size / wm_height)
            new_width = max(1, int(wm_width * ratio))
            new_height = max(1, int(wm_height * ratio))
            resized_watermark = watermark_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 调整不透明度
            if 不透明度 < 1.0:
                r, g, b, a = resized_watermark.split()
                a = a.point(lambda p: int(p * 不透明度))
                resized_watermark = Image.merge("RGBA", (r, g, b, a))

            # 旋转水印
            if 旋转角度 != 0:
                rotated_watermark = resized_watermark.rotate(
                    旋转角度, expand=True, resample=Image.Resampling.BICUBIC
                )
            else:
                rotated_watermark = resized_watermark

            rot_width, rot_height = rotated_watermark.size

            # 计算最小包围盒（基于图片短边）
            min_box_size = int(base_size * 最小包围盒比例)

            # 计算水平和垂直的包围盒（长方形，分别计算）
            box_width = max(int(rot_width * 包围盒倍数), min_box_size)
            box_height = max(int(rot_height * 包围盒倍数), min_box_size)

            # 分别计算行列数（独立密度控制）
            base_cols = max(3, int(3 * 水平密度))
            base_rows = max(3, int(3 * 垂直密度))
            
            # 根据图片比例调整，确保不会太稀疏或太密集
            cols = max(2, int(base_cols * img_width / max(img_width, img_height)))
            rows = max(2, int(base_rows * img_height / max(img_width, img_height)))

            # 计算网格起始位置（居中布局）
            total_width = cols * box_width
            total_height = rows * box_height
            start_x = (img_width - total_width) // 2
            start_y = (img_height - total_height) // 2

            # 预计算所有水印位置
            positions = []
            for row in range(rows):
                for col in range(cols):
                    # 包围盒左上角
                    box_x = start_x + col * box_width
                    box_y = start_y + row * box_height

                    # 包围盒中心
                    center_x = box_x + box_width / 2
                    center_y = box_y + box_height / 2

                    # 计算剩余空间（包围盒减去水印尺寸的一半）
                    spare_x = (box_width - rot_width) / 2
                    spare_y = (box_height - rot_height) / 2

                    # 在剩余空间内随机偏移（限制最大偏移比例）
                    max_offset_x = spare_x * min(最大随机偏移, 1.0)
                    max_offset_y = spare_y * min(最大随机偏移, 1.0)

                    offset_x = rng.uniform(-max_offset_x, max_offset_x) if max_offset_x > 0 else 0
                    offset_y = rng.uniform(-max_offset_y, max_offset_y) if max_offset_y > 0 else 0

                    # 最终位置（水印左上角）
                    final_x = int(center_x - rot_width / 2 + offset_x)
                    final_y = int(center_y - rot_height / 2 + offset_y)

                    positions.append((final_x, final_y))

            # 应用水印到该分辨率的所有图片
            for img_idx, img_tensor in group:
                pil_image = base_pil.copy() if img_idx == group[0][0] else self._tensor_to_pil(img_tensor).convert("RGBA")

                for final_x, final_y in positions:
                    # 只粘贴可见区域
                    if final_x + rot_width > 0 and final_x < img_width and \
                       final_y + rot_height > 0 and final_y < img_height:
                        pil_image.paste(rotated_watermark, (final_x, final_y), rotated_watermark)

                pil_image = pil_image.convert("RGB")
                processed_images[img_idx] = self._pil_to_tensor(pil_image)

        return (torch.cat(processed_images, dim=0),)

    def _create_default_watermark(self):
        """创建默认水印图像"""
        watermark_pil = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark_pil)
        draw.rectangle([0, 0, 100, 100], fill=(0, 0, 0, 100))
        draw.text((20, 40), "Watermark", fill=(255, 255, 255, 200))
        return watermark_pil


# 节点映射
NODE_CLASS_MAPPINGS = {
    "AddGridWatermark": AddGridWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddGridWatermark": "添加网格水印（独立行列密度）",
}
