"""
SAM3 Semantic Segmentor
=======================

基于 SAM3 模型的语义分割类，提供与 SemanticSegmentor 一致的接口，
支持多图片、多提示词的批处理推理。

用法示例:
    from sam3_segmentor import SAM3Segmentor, SegmentQuery

    segmentor = SAM3Segmentor()
    query = SegmentQuery(
        image_paths=["image1.png", "image2.png"],
        prompts=["cup", "keyboard"]
    )
    response = segmentor.predict(query)
    response.visualize(output_dir="./vis_results")
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SegmentQuery:
    """
    语义分割查询对象，封装图片路径和文本提示词。
    """

    def __init__(
        self,
        image_paths: Union[str, List[Union[str, np.ndarray, Image.Image]]],
        prompts: Union[str, List[str]],
    ):
        """
        :param image_paths: 图片文件路径列表，也可以传入 numpy 数组或 PIL Image。
                            单个路径/图片也可直接传入，会自动转为列表。
        :param prompts:     文本描述列表（每个描述对应一个目标类别）。
                            单个字符串也可直接传入，会自动转为列表。
        """
        if isinstance(image_paths, (str, np.ndarray, Image.Image)):
            self.image_paths = [image_paths]
        else:
            self.image_paths = list(image_paths)

        if isinstance(prompts, str):
            self.prompts = [prompts]
        else:
            self.prompts = list(prompts)


class SegmentResult:
    """
    单次检测的结果，包含 scores, boxes, masks。
    """

    def __init__(self, scores: np.ndarray, boxes: np.ndarray, masks: np.ndarray):
        """
        :param scores: (N,) 置信度分数
        :param boxes:  (N, 4) 边界框 [x1, y1, x2, y2]
        :param masks:  (N, H, W) 二值掩码
        """
        self.scores = scores
        self.boxes = boxes
        self.masks = masks

    @property
    def count(self) -> int:
        return len(self.boxes)


class SegmentResponse:
    """
    语义分割响应结果。

    内部结构:
        results[image_key][label] = SegmentResult
    """

    def __init__(self):
        self.results: Dict[str, Dict[str, SegmentResult]] = {}

    def update(
        self,
        image_key: str,
        label: str,
        scores: np.ndarray,
        boxes: np.ndarray,
        masks: np.ndarray,
    ):
        """添加/更新某张图片某个标签的分割结果。"""
        if image_key not in self.results:
            self.results[image_key] = {}
        self.results[image_key][label] = SegmentResult(scores, boxes, masks)

    def dict_results(self) -> Dict[str, Dict[str, dict]]:
        """将结果转为纯字典格式，方便序列化。"""
        out = {}
        for key, labels_data in self.results.items():
            out[key] = {}
            for label, seg in labels_data.items():
                out[key][label] = {
                    "scores": seg.scores,
                    "boxes": seg.boxes,
                    "masks": seg.masks,
                }
        return out

    def visualize(self, output_dir: Optional[str] = None):
        """
        可视化分割结果（边界框 + 掩码叠加）。

        :param output_dir: 若提供，将结果保存到该目录；否则调用 PIL show() 显示。
        """
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for image_key, labels_data in self.results.items():
            # 尝试打开图片
            try:
                image = Image.open(image_key).convert("RGBA")
            except Exception:
                print(f"[visualize] Cannot open '{image_key}', skipping.")
                continue

            mask_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)

            for label, seg in labels_data.items():
                color = tuple(np.random.randint(0, 256, 3).tolist())

                for i in range(seg.count):
                    box = seg.boxes[i]
                    mask = seg.masks[i]
                    score = seg.scores[i]

                    # 画边界框
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                    # 画标签文字
                    text = f"{label}: {score:.2f}"
                    draw.text((x1, max(0, y1 - 15)), text, fill=color)

                    # 画掩码
                    if mask.ndim == 2:
                        solid = Image.new("RGBA", image.size, color + (100,))
                        mask_pil = Image.fromarray(
                            (mask * 255).astype(np.uint8), mode="L"
                        )
                        mask_layer.paste(solid, (0, 0), mask_pil)

            result_image = Image.alpha_composite(image, mask_layer).convert("RGB")

            if output_dir:
                filename = os.path.basename(image_key)
                save_path = os.path.join(output_dir, f"vis_{filename}")
                result_image.save(save_path)
                print(f"Saved visualization to {save_path}")
            else:
                result_image.show()


class SAM3Segmentor:
    """
    基于 SAM3 的语义分割器。

    功能:
        - 支持多张图片 + 多个文本提示词的批处理推理
        - 输入支持文件路径 / numpy 数组 / PIL Image
        - 自动将 SAM3 输出转为统一的 SegmentResponse 格式
        - 内置结果可视化

    参数 (options):
        - confidence_threshold (float): 置信度阈值，默认 0.5
        - device (str): 推理设备，默认 "cuda"
        - resolution (int): 输入分辨率，默认 1008
    """

    def __init__(self, options: Optional[dict] = None):
        self.options = options if options is not None else {}

        device = self.options.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        resolution = self.options.get("resolution", 1008)
        confidence_threshold = self.options.get("confidence_threshold", 0.5)

        print(f"[SAM3Segmentor] Loading model (device={device}, resolution={resolution})...")
        self.model = build_sam3_image_model(device=device)
        self.processor = Sam3Processor(
            self.model,
            resolution=resolution,
            device=device,
            confidence_threshold=confidence_threshold,
        )
        self.device = device
        print("[SAM3Segmentor] Model loaded.")

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(inp, idx: int):
        """
        将输入统一转换为 (PIL.Image, image_key) 的形式。

        :return: (image, key) 或 (None, None)
        """
        if isinstance(inp, str):
            try:
                return Image.open(inp).convert("RGB"), inp
            except Exception as e:
                print(f"[SAM3Segmentor] Error opening '{inp}': {e}")
                return None, None
        elif isinstance(inp, np.ndarray):
            try:
                return Image.fromarray(inp).convert("RGB"), f"image_{idx}"
            except Exception as e:
                print(f"[SAM3Segmentor] Error converting ndarray: {e}")
                return None, None
        elif isinstance(inp, Image.Image):
            return inp.convert("RGB"), f"image_{idx}"
        else:
            print(f"[SAM3Segmentor] Unsupported input type: {type(inp)}")
            return None, None

    @staticmethod
    def _postprocess_output(output: dict):
        """
        将 SAM3 的原始 tensor 输出转为 numpy 数组，
        并做 squeeze / threshold 等后处理。

        :return: (masks_np, boxes_np, scores_np) 或 None
        """
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")

        if masks is None or boxes is None or len(boxes) == 0:
            return None

        # float32 转换（避免 BFloat16 不支持 numpy 的问题）
        masks_np = masks.detach().float().cpu().numpy()
        boxes_np = boxes.detach().float().cpu().numpy()
        scores_np = scores.detach().float().cpu().numpy()

        # [N, 1, H, W] -> [N, H, W]
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np.squeeze(1)

        # 二值化
        masks_np = (masks_np > 0).astype(np.uint8)

        # 修复 0-d 数组
        if np.ndim(scores_np) == 0:
            scores_np = np.array([scores_np])

        return masks_np, boxes_np, scores_np

    # ------------------------------------------------------------------
    # 核心推理
    # ------------------------------------------------------------------

    def predict(self, query: SegmentQuery) -> SegmentResponse:
        """
        对查询中的每张图片，依次用每个文本提示词进行推理。

        SAM3 的 `set_image` 会对图片做一次 backbone 前向传播并缓存特征，
        然后可多次调用 `set_text_prompt` 对不同 prompt 复用特征，
        因此每张图只做一次图像编码，多个 prompt 共享 backbone 输出。

        :param query: SegmentQuery 对象
        :return: SegmentResponse 对象
        """
        response = SegmentResponse()

        prompts = [p.strip().rstrip(".") for p in query.prompts]
        total = len(query.image_paths)

        for idx, inp in enumerate(query.image_paths):
            image, key = self._load_image(inp, idx)
            if image is None:
                continue

            print(f"[SAM3Segmentor] Processing image {idx + 1}/{total}: {key}")

            # 一次图像编码
            inference_state = self.processor.set_image(image)

            # 对每个 prompt 复用 backbone 特征
            for prompt in prompts:
                output = self.processor.set_text_prompt(
                    prompt=prompt, state=inference_state
                )
                result = self._postprocess_output(output)
                if result is None:
                    continue
                masks_np, boxes_np, scores_np = result
                response.update(
                    image_key=key,
                    label=prompt,
                    scores=scores_np,
                    boxes=boxes_np,
                    masks=masks_np,
                )

        return response

    def predict_single(
        self,
        image: Union[str, np.ndarray, Image.Image],
        prompt: str,
    ) -> Optional[SegmentResult]:
        """
        便捷方法：对单张图片、单个 prompt 进行推理。

        :return: SegmentResult 或 None（无检测结果时）
        """
        query = SegmentQuery(image, prompt)
        resp = self.predict(query)

        if not resp.results:
            return None

        first_key = next(iter(resp.results))
        label_data = resp.results[first_key]
        if prompt in label_data:
            return label_data[prompt]
        return None


# ======================================================================
# 主入口示例
# ======================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAM3 Semantic Segmentation Demo")
    parser.add_argument(
        "--images",
        nargs="+",
        default=["../../asserts/111.png", "../../asserts/222.png"],
        help="Image file paths",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["vase", "chair"],
        help="Text prompts for detection",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vis_results",
        help="Directory to save visualization results",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold",
    )
    args = parser.parse_args()

    # 初始化
    segmentor = SAM3Segmentor(options={"confidence_threshold": args.confidence})

    # 构造查询
    query = SegmentQuery(image_paths=args.images, prompts=args.prompts)

    # 推理
    response = segmentor.predict(query)

    # 可视化
    response.visualize(output_dir=args.output_dir)

    # 打印结果摘要
    results = response.dict_results()
    for path, labels_data in results.items():
        print(f"\nResults for {path}:")
        for label, data in labels_data.items():
            n = len(data["boxes"])
            print(f"  Label '{label}': {n} detection(s), scores={data['scores']}")
