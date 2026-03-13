"""
模块名称: SAM3 语义分割测试脚本
功能描述:
    1. 读取 ./benchmark/example/discription.json 中的内容
    2. 选择指定图像 (3, 5, 8, 10, 11)
    3. 提取这些图像中所有物体的 short_en 和 detailed_en 描述
    4. 调用 SAM3 接口对相应的图像进行语言描述的语义分割
    5. 保存检测结果并绘制 Bounding Box 与 Mask
    6. 将结果图片保存到 ./sam3_result 目录
    7. 记录所有推理请求和结果到日志文件
"""

import json
import os
import re
import sys
import shutil
import time
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw

# 添加 sam3 模块到路径
sys.path.insert(0, os.path.dirname(__file__))
from sam3.sam3_segmentor import SAM3Segmentor, SegmentQuery, SegmentResult


# ================================================================
# 推理日志记录配置
# ================================================================

class SAM3Logger:
    """SAM3 推理日志记录器"""

    def __init__(self, log_file):
        """
        初始化日志记录器

        参数:
            log_file (str): 日志文件路径
        """
        self.log_file = log_file
        self.logs = []
        self.detection_results = []  # 用于统计检测结果

    def _parse_image_info(self, image_filename):
        """
        从图像文件名解析图片序号、物体序号和类型

        返回:
            tuple: (image_num, object_num, desc_type)
        """
        # 尝试提取单物体模式: 3.jpg_single_obj_1_short/detailed
        match = re.search(r'(\d+)\.jpg_single_obj_(\d+)_(short|detailed)', image_filename)
        if match:
            image_num = match.group(1)
            object_num = match.group(2)
            desc_type = "short_en" if match.group(3) == "short" else "detailed_en"
            return image_num, object_num, desc_type

        # 尝试提取: 3_obj1_short/detailed
        match = re.search(r'(\d+)_obj(\d+)_(short|detailed)', image_filename)
        if match:
            image_num = match.group(1)
            object_num = match.group(2)
            desc_type = "short_en" if match.group(3) == "short" else "detailed_en"
            return image_num, object_num, desc_type

        # 尝试提取多物体模式: 3.jpg_short_en 或 3.jpg_detailed_en
        match = re.search(r'(\d+)\.jpg_(short|detailed)_en', image_filename)
        if match:
            image_num = match.group(1)
            desc_type = "short_en" if match.group(2) == "short" else "detailed_en"
            return image_num, None, desc_type

        return None, None, None

    def _generate_entry_name(self, log_entry):
        """
        生成日志条目的名称

        返回:
            str: 日志条目名称
        """
        if log_entry['type'] in ['INFO', 'ERROR']:
            message = log_entry.get('message', '').strip()
            if message:
                name_part = message.replace('===', '').strip()[:20]
                return f"{log_entry['type']}_{name_part}" if name_part else log_entry['type']
            return log_entry['type']

        image_file = log_entry.get('image_file')

        if not image_file:
            current_index = self.logs.index(log_entry)
            for prev_log in reversed(self.logs[:current_index]):
                if prev_log.get('type') == 'INFERENCE' and 'image_file' in prev_log:
                    image_file = prev_log.get('image_file', '')
                    break

        if not image_file:
            return f"{log_entry['type']}_{log_entry.get('prompt', 'N/A')[:15]}"

        image_num, object_num, desc_type = self._parse_image_info(image_file)

        if image_num is None or desc_type is None:
            return f"{log_entry['type']}_{log_entry.get('prompt', 'N/A')[:15]}"

        desc_type_short = desc_type.replace('_en', '')
        if object_num is not None:
            return f"{image_num}_{object_num}_{desc_type_short}"
        else:
            return f"{image_num}_{desc_type_short}"

    def log_inference(self, image_path, prompt, image_filename, duration_sec, result_count):
        """
        记录推理请求

        参数:
            image_path (str): 图像路径
            prompt (str): 检测提示词
            image_filename (str): 图像文件标识
            duration_sec (float): 推理耗时（秒）
            result_count (int): 检测到的物体数
        """
        timestamp = datetime.now().isoformat()

        log_entry = {
            "timestamp": timestamp,
            "type": "INFERENCE",
            "image_path": image_path,
            "prompt": prompt,
            "image_file": image_filename,
            "duration_sec": round(duration_sec, 3),
            "result_count": result_count,
        }

        self.logs.append(log_entry)
        print(f"[LOG] {timestamp} - INFERENCE: prompt='{prompt}', results={result_count}, time={duration_sec:.3f}s")

    def log_result_detail(self, image_filename, prompt, scores, boxes):
        """
        记录检测结果详情

        参数:
            image_filename (str): 图像文件标识
            prompt (str): 提示词
            scores (np.ndarray): 置信度
            boxes (np.ndarray): 边界框
        """
        timestamp = datetime.now().isoformat()

        log_entry = {
            "timestamp": timestamp,
            "type": "RESULT_DETAIL",
            "image_file": image_filename,
            "prompt": prompt,
            "scores": scores.tolist() if isinstance(scores, np.ndarray) else scores,
            "boxes": boxes.tolist() if isinstance(boxes, np.ndarray) else boxes,
        }

        self.logs.append(log_entry)

    def log_error(self, error_msg):
        """记录错误信息"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "type": "ERROR",
            "message": error_msg,
        }
        self.logs.append(log_entry)
        print(f"[LOG] {timestamp} - ERROR: {error_msg}")

    def log_info(self, info_msg):
        """记录信息"""
        timestamp = datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "type": "INFO",
            "message": info_msg,
        }
        self.logs.append(log_entry)
        print(f"[LOG] {timestamp} - INFO: {info_msg}")

    def save_to_file(self):
        """将所有日志保存到文件"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SAM3 语义分割 推理日志记录\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().isoformat()}\n")
            f.write(f"总条目数: {len(self.logs)}\n")
            f.write("=" * 80 + "\n\n")

            for log in self.logs:
                entry_name = self._generate_entry_name(log)
                f.write(f"\n【{entry_name}】\n")
                f.write(f"时间: {log['timestamp']}\n")
                f.write(f"类型: {log['type']}\n")

                if log['type'] == 'INFERENCE':
                    f.write(f"图像路径: {log['image_path']}\n")
                    f.write(f"提示词: {log['prompt']}\n")
                    f.write(f"图像标识: {log['image_file']}\n")
                    f.write(f"推理耗时: {log['duration_sec']}s\n")
                    f.write(f"检测数量: {log['result_count']}\n")

                elif log['type'] == 'RESULT_DETAIL':
                    f.write(f"图像标识: {log['image_file']}\n")
                    f.write(f"提示词: {log['prompt']}\n")
                    f.write(f"置信度: {log['scores']}\n")
                    f.write(f"边界框: {log['boxes']}\n")

                elif log['type'] == 'ERROR':
                    f.write(f"错误: {log['message']}\n")

                elif log['type'] == 'INFO':
                    f.write(f"信息: {log['message']}\n")

                f.write("\n" + "-" * 80 + "\n")

        print(f"\n[+] 推理日志已保存到: {self.log_file}")

    def add_detection_result(self, image_num, object_id, desc_type, prompt,
                             detected_count, scores, boxes):
        """
        添加检测结果到统计列表

        参数:
            image_num (int): 图像编号
            object_id (int): 物体编号
            desc_type (str): 描述类型 (short_en / detailed_en)
            prompt (str): 检测提示词
            detected_count (int): 检测到的数量
            scores (list): 置信度列表
            boxes (list): 边界框列表
        """
        self.detection_results.append({
            'image_num': image_num,
            'object_id': object_id,
            'description_type': desc_type,
            'prompt': prompt,
            'detected_count': detected_count,
            'scores': scores,
            'boxes': boxes,
            'status': 'success' if detected_count > 0 else 'empty',
            'timestamp': datetime.now().isoformat(),
        })

    def save_detection_summary(self):
        """保存检测结果统计到 JSON 文件"""
        summary_file = self.log_file.replace('.txt', '_summary.json')

        total_detections = len(self.detection_results)
        successful_detections = sum(1 for r in self.detection_results if r['status'] == 'success')
        empty_detections = sum(1 for r in self.detection_results if r['status'] == 'empty')

        summary_data = {
            'generated_time': datetime.now().isoformat(),
            'statistics': {
                'total_detections': total_detections,
                'successful_detections': successful_detections,
                'empty_detections': empty_detections,
                'success_rate': f"{(successful_detections / total_detections * 100):.2f}%" if total_detections > 0 else '0%',
            },
            'detection_results': self.detection_results,
        }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print(f"[+] 检测结果统计已保存到: {summary_file}")
        print(f"    - 总检测次数: {total_detections}")
        print(f"    - 成功检测: {successful_detections}")
        print(f"    - 未检测到: {empty_detections}")
        print(f"    - 成功率: {summary_data['statistics']['success_rate']}")


# 全局日志记录器
sam3_logger: SAM3Logger | None = None

# 全局 SAM3 分割器（避免重复加载模型）
sam3_segmentor: SAM3Segmentor | None = None


# ================================================================
# 工具函数
# ================================================================

def load_description_data(json_path):
    """
    读取 description.json 文件

    参数:
        json_path (str): JSON 文件的路径
    返回:
        list: 解析后的 JSON 数据
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到描述文件: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def draw_sam3_results_on_image(image_path, seg_result: SegmentResult, label: str, suffix: str):
    """
    在图像上绘制 SAM3 检测结果（bbox + mask）并保存

    参数:
        image_path (str): 原图路径
        seg_result (SegmentResult): 分割结果
        label (str): 标签文字
        suffix (str): 保存文件名后缀
    """
    try:
        image = Image.open(image_path).convert("RGBA")
    except Exception as e:
        print(f"[!] 无法打开图像 {image_path}: {e}")
        return

    mask_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    color = tuple(np.random.randint(50, 230, 3).tolist())

    for i in range(seg_result.count):
        box = seg_result.boxes[i]
        score = seg_result.scores[i]
        mask = seg_result.masks[i]

        # 画边界框
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 画标签
        text = f"{label}: {score:.2f}"
        draw.text((x1, max(0, y1 - 15)), text, fill=color)

        # 画掩码
        if mask.ndim == 2:
            solid = Image.new("RGBA", image.size, color + (80,))
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            mask_layer.paste(solid, (0, 0), mask_pil)

    result_image = Image.alpha_composite(image, mask_layer).convert("RGB")

    # 生成保存路径
    base, ext = os.path.splitext(image_path)
    save_path = f"{base}{suffix}{ext}"
    result_image.save(save_path)
    print(f"    [+] 结果图片已保存: {save_path}")


def run_sam3_inference(segmentor: SAM3Segmentor, image_path: str, prompt: str):
    """
    运行一次 SAM3 推理并返回结果和耗时

    参数:
        segmentor (SAM3Segmentor): 分割器实例
        image_path (str): 图像路径
        prompt (str): 文本提示词

    返回:
        tuple: (SegmentResult or None, duration_sec)
    """
    start_time = time.time()
    result = segmentor.predict_single(image_path, prompt)
    duration = time.time() - start_time
    return result, duration


# ================================================================
# 图像处理
# ================================================================

def process_image(image_path, image_info, output_dir, negative_prompt_info=None):
    """
    处理单张图像的语义分割（对每个物体分别使用 short_en 和 detailed_en）

    参数:
        image_path (str): 图像文件路径
        image_info (dict): 从 JSON 中读取的图像信息
        output_dir (str): 输出目录路径
        negative_prompt_info (dict | None): 负样本提示词来源（通常为下一张图的描述）
    """
    global sam3_logger, sam3_segmentor

    if not os.path.exists(image_path):
        print(f"[!] 警告: 图像文件不存在: {image_path}")
        return False

    image_filename = image_info['image_filename']
    print(f"\n{'='*70}")
    print(f"处理图像: {image_filename}")
    print(f"{'='*70}")

    try:
        objects = image_info['object_descriptions']

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 复制原图到输出目录
        base_filename = os.path.basename(image_path)
        output_image_path = os.path.join(output_dir, base_filename)
        if not os.path.exists(output_image_path):
            shutil.copy(image_path, output_image_path)

        # 从文件名提取图像编号
        match = re.search(r'(\d+)\.jpg', base_filename)
        image_num = int(match.group(1)) if match else 0

        print(f"\n[*] 图像中共有 {len(objects)} 个物体")
        print(f"[*] 将对每个物体分别进行 short_en 和 detailed_en 检测，总共 {len(objects) * 2} 轮检测")

        # 对每个物体分别进行检测
        for obj_idx, obj in enumerate(objects, 1):
            print(f"\n{'='*70}")
            print(f"[物体 {obj_idx}/{len(objects)}] {obj['short_en']}")
            print(f"{'='*70}")

            print(f"  - 物体ID: {obj.get('object_id', 'N/A')}")
            print(f"  - 难度: {obj.get('difficulty', 'N/A')}")
            print(f"  - 简短描述: {obj['short_en']}")
            print(f"  - 详细描述: {obj['detailed_en']}")

            # ====== 使用 short_en 检测 ======
            print(f"\n  [检测 {obj_idx * 2 - 1}/{len(objects) * 2}] 使用 short_en 进行检测")
            prompt_short = obj['short_en']
            print(f"    提示词: {prompt_short}")

            try:
                result, duration = run_sam3_inference(sam3_segmentor, output_image_path, prompt_short)

                detected_count = result.count if result else 0

                # 记录推理日志
                if sam3_logger:
                    sam3_logger.log_inference(
                        image_path=output_image_path,
                        prompt=prompt_short,
                        image_filename=f"{base_filename}_obj{obj_idx}_short",
                        duration_sec=duration,
                        result_count=detected_count,
                    )

                if result and result.count > 0:
                    print(f"    [+] 检测成功: 检测到 {result.count} 个物体 (耗时 {duration:.3f}s)")

                    # 记录结果详情
                    if sam3_logger:
                        sam3_logger.log_result_detail(
                            image_filename=f"{base_filename}_obj{obj_idx}_short",
                            prompt=prompt_short,
                            scores=result.scores,
                            boxes=result.boxes,
                        )
                        sam3_logger.add_detection_result(
                            image_num=image_num,
                            object_id=obj_idx,
                            desc_type='short_en',
                            prompt=prompt_short,
                            detected_count=result.count,
                            scores=result.scores.tolist(),
                            boxes=result.boxes.tolist(),
                        )

                    # 绘制结果
                    draw_sam3_results_on_image(
                        output_image_path,
                        result,
                        prompt_short,
                        f"_obj{obj_idx}_short",
                    )
                else:
                    print(f"    [!] 未检测到任何物体 (耗时 {duration:.3f}s)")
                    if sam3_logger:
                        sam3_logger.add_detection_result(
                            image_num=image_num,
                            object_id=obj_idx,
                            desc_type='short_en',
                            prompt=prompt_short,
                            detected_count=0,
                            scores=[],
                            boxes=[],
                        )

            except Exception as e:
                print(f"    [X] 检测出错: {e}")
                if sam3_logger:
                    sam3_logger.log_error(f"物体 {obj_idx} short_en 检测出错: {e}")

            # ====== 使用 detailed_en 检测 ======
            print(f"\n  [检测 {obj_idx * 2}/{len(objects) * 2}] 使用 detailed_en 进行检测")
            prompt_detailed = obj['detailed_en']
            print(f"    提示词: {prompt_detailed}")

            try:
                result, duration = run_sam3_inference(sam3_segmentor, output_image_path, prompt_detailed)

                detected_count = result.count if result else 0

                if sam3_logger:
                    sam3_logger.log_inference(
                        image_path=output_image_path,
                        prompt=prompt_detailed,
                        image_filename=f"{base_filename}_obj{obj_idx}_detailed",
                        duration_sec=duration,
                        result_count=detected_count,
                    )

                if result and result.count > 0:
                    print(f"    [+] 检测成功: 检测到 {result.count} 个物体 (耗时 {duration:.3f}s)")

                    if sam3_logger:
                        sam3_logger.log_result_detail(
                            image_filename=f"{base_filename}_obj{obj_idx}_detailed",
                            prompt=prompt_detailed,
                            scores=result.scores,
                            boxes=result.boxes,
                        )
                        sam3_logger.add_detection_result(
                            image_num=image_num,
                            object_id=obj_idx,
                            desc_type='detailed_en',
                            prompt=prompt_detailed,
                            detected_count=result.count,
                            scores=result.scores.tolist(),
                            boxes=result.boxes.tolist(),
                        )

                    draw_sam3_results_on_image(
                        output_image_path,
                        result,
                        prompt_detailed[:30],
                        f"_obj{obj_idx}_detailed",
                    )
                else:
                    print(f"    [!] 未检测到任何物体 (耗时 {duration:.3f}s)")
                    if sam3_logger:
                        sam3_logger.add_detection_result(
                            image_num=image_num,
                            object_id=obj_idx,
                            desc_type='detailed_en',
                            prompt=prompt_detailed,
                            detected_count=0,
                            scores=[],
                            boxes=[],
                        )

            except Exception as e:
                print(f"    [X] 检测出错: {e}")
                if sam3_logger:
                    sam3_logger.log_error(f"物体 {obj_idx} detailed_en 检测出错: {e}")

        # ====== 误检测测试：使用下一张图的提示词 ======
        if negative_prompt_info is not None:
            next_image_name = negative_prompt_info.get('image_filename', 'N/A')
            negative_objects = negative_prompt_info.get('object_descriptions', [])

            print(f"\n{'='*70}")
            print("[误检测测试] 当前图像使用下一张图的提示词")
            print(f"  - 当前图像: {image_filename}")
            print(f"  - 提示词来源: {next_image_name}")
            print(f"  - 负样本提示词数量: {len(negative_objects) * 2}")
            print(f"{'='*70}")

            for neg_idx, neg_obj in enumerate(negative_objects, 1):
                # short_en 负样本
                neg_short = neg_obj.get('short_en', '').strip()
                if neg_short:
                    print(f"\n  [误检 short {neg_idx}/{len(negative_objects)}] 提示词: {neg_short}")
                    try:
                        result, duration = run_sam3_inference(sam3_segmentor, output_image_path, neg_short)
                        detected_count = result.count if result else 0

                        if sam3_logger:
                            sam3_logger.log_inference(
                                image_path=output_image_path,
                                prompt=neg_short,
                                image_filename=f"{base_filename}_mis_next_obj{neg_idx}_short",
                                duration_sec=duration,
                                result_count=detected_count,
                            )
                            sam3_logger.add_detection_result(
                                image_num=image_num,
                                object_id=neg_idx,
                                desc_type='misdetect_short_en',
                                prompt=neg_short,
                                detected_count=detected_count,
                                scores=result.scores.tolist() if result else [],
                                boxes=result.boxes.tolist() if result else [],
                            )

                        if result and result.count > 0:
                            print(f"    [!] 误检测命中: {result.count} 个 (耗时 {duration:.3f}s)")
                            if sam3_logger:
                                sam3_logger.log_result_detail(
                                    image_filename=f"{base_filename}_mis_next_obj{neg_idx}_short",
                                    prompt=neg_short,
                                    scores=result.scores,
                                    boxes=result.boxes,
                                )
                            draw_sam3_results_on_image(
                                output_image_path,
                                result,
                                f"MIS-{neg_short}",
                                f"_mis_next_obj{neg_idx}_short",
                            )
                        else:
                            print(f"    [+] 符合预期: 未检测到 (耗时 {duration:.3f}s)")
                    except Exception as e:
                        print(f"    [X] 误检 short 测试出错: {e}")
                        if sam3_logger:
                            sam3_logger.log_error(f"误检 short 测试出错(当前图 {image_filename}, 词来源 {next_image_name}, idx={neg_idx}): {e}")

                # detailed_en 负样本
                neg_detailed = neg_obj.get('detailed_en', '').strip()
                if neg_detailed:
                    print(f"\n  [误检 detailed {neg_idx}/{len(negative_objects)}] 提示词: {neg_detailed}")
                    try:
                        result, duration = run_sam3_inference(sam3_segmentor, output_image_path, neg_detailed)
                        detected_count = result.count if result else 0

                        if sam3_logger:
                            sam3_logger.log_inference(
                                image_path=output_image_path,
                                prompt=neg_detailed,
                                image_filename=f"{base_filename}_mis_next_obj{neg_idx}_detailed",
                                duration_sec=duration,
                                result_count=detected_count,
                            )
                            sam3_logger.add_detection_result(
                                image_num=image_num,
                                object_id=neg_idx,
                                desc_type='misdetect_detailed_en',
                                prompt=neg_detailed,
                                detected_count=detected_count,
                                scores=result.scores.tolist() if result else [],
                                boxes=result.boxes.tolist() if result else [],
                            )

                        if result and result.count > 0:
                            print(f"    [!] 误检测命中: {result.count} 个 (耗时 {duration:.3f}s)")
                            if sam3_logger:
                                sam3_logger.log_result_detail(
                                    image_filename=f"{base_filename}_mis_next_obj{neg_idx}_detailed",
                                    prompt=neg_detailed,
                                    scores=result.scores,
                                    boxes=result.boxes,
                                )
                            draw_sam3_results_on_image(
                                output_image_path,
                                result,
                                "MIS-detailed",
                                f"_mis_next_obj{neg_idx}_detailed",
                            )
                        else:
                            print(f"    [+] 符合预期: 未检测到 (耗时 {duration:.3f}s)")
                    except Exception as e:
                        print(f"    [X] 误检 detailed 测试出错: {e}")
                        if sam3_logger:
                            sam3_logger.log_error(f"误检 detailed 测试出错(当前图 {image_filename}, 词来源 {next_image_name}, idx={neg_idx}): {e}")

        print(f"\n{'='*70}")
        print(f"[+] 图像 {image_filename} 处理完成！")
        print(f"{'='*70}")
        return True

    except Exception as e:
        print(f"[X] 处理图像时出错: {e}")
        if sam3_logger:
            sam3_logger.log_error(f"处理图像 {image_filename} 时出错: {e}")
        traceback.print_exc()
        return False


def process_single_object_detection(image_path, image_info, object_index, output_dir):
    """
    处理单张图像中单个物体的语义分割

    参数:
        image_path (str): 图像文件路径
        image_info (dict): 从 JSON 中读取的图像信息
        object_index (int): 要检测的物体索引 (0-based)
        output_dir (str): 输出目录路径
    """
    global sam3_logger, sam3_segmentor

    if not os.path.exists(image_path):
        print(f"[!] 警告: 图像文件不存在: {image_path}")
        return False

    image_filename = image_info['image_filename']
    print(f"\n{'='*70}")
    print(f"处理图像: {image_filename}")
    print(f"{'='*70}")

    try:
        objects = image_info['object_descriptions']
        if object_index >= len(objects) or object_index < 0:
            print(f"[!] 物体索引 {object_index} 超出范围 (共有 {len(objects)} 个物体)")
            return False

        target_object = objects[object_index]

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 复制原图到输出目录
        base_filename = os.path.basename(image_path)
        base_without_ext = os.path.splitext(base_filename)[0]
        ext = os.path.splitext(base_filename)[1]
        output_image_path = os.path.join(output_dir, f"{base_without_ext}_obj{object_index+1}{ext}")
        shutil.copy(image_path, output_image_path)

        # 从文件名提取图像编号
        match = re.search(r'(\d+)\.jpg', base_filename)
        image_num = int(match.group(1)) if match else 0

        print(f"\n[*] 目标物体信息:")
        print(f"    - 物体ID: {target_object.get('object_id', 'N/A')}")
        print(f"    - 难度: {target_object.get('difficulty', 'N/A')}")
        print(f"    - 简短描述 (short_en): {target_object.get('short_en', 'N/A')}")
        print(f"    - 详细描述 (detailed_en): {target_object.get('detailed_en', 'N/A')}")

        # ============================================================
        # 单物体检测 - short_en
        # ============================================================
        print(f"\n{'='*70}")
        print(f"[单物体检测 - short_en] 使用简短描述")
        print("-" * 70)

        prompt_short = target_object['short_en']
        print(f"[*] 检测提示词: {prompt_short}")

        try:
            result, duration = run_sam3_inference(sam3_segmentor, output_image_path, prompt_short)
            detected_count = result.count if result else 0

            if sam3_logger:
                sam3_logger.log_inference(
                    image_path=output_image_path,
                    prompt=prompt_short,
                    image_filename=f"{base_filename}_single_obj_{object_index+1}_short",
                    duration_sec=duration,
                    result_count=detected_count,
                )
                sam3_logger.add_detection_result(
                    image_num=image_num,
                    object_id=object_index + 1,
                    desc_type='short_en',
                    prompt=prompt_short,
                    detected_count=detected_count,
                    scores=result.scores.tolist() if result else [],
                    boxes=result.boxes.tolist() if result else [],
                )

            if result and result.count > 0:
                print(f"[+] 成功检测到 {result.count} 个物体 (耗时 {duration:.3f}s)")
                if sam3_logger:
                    sam3_logger.log_result_detail(
                        image_filename=f"{base_filename}_single_obj_{object_index+1}_short",
                        prompt=prompt_short,
                        scores=result.scores,
                        boxes=result.boxes,
                    )
                draw_sam3_results_on_image(
                    output_image_path,
                    result,
                    prompt_short,
                    "_short_detected",
                )
            else:
                print(f"[!] 未检测到任何物体 (耗时 {duration:.3f}s)")

        except Exception as e:
            print(f"[X] 短描述检测出错: {e}")
            if sam3_logger:
                sam3_logger.log_error(f"单物体短描述检测出错: {e}")

        # ============================================================
        # 单物体检测 - detailed_en
        # ============================================================
        print(f"\n{'='*70}")
        print(f"[单物体检测 - detailed_en] 使用详细描述")
        print("-" * 70)

        prompt_detailed = target_object['detailed_en']
        print(f"[*] 检测提示词: {prompt_detailed}")

        try:
            result, duration = run_sam3_inference(sam3_segmentor, output_image_path, prompt_detailed)
            detected_count = result.count if result else 0

            if sam3_logger:
                sam3_logger.log_inference(
                    image_path=output_image_path,
                    prompt=prompt_detailed,
                    image_filename=f"{base_filename}_single_obj_{object_index+1}_detailed",
                    duration_sec=duration,
                    result_count=detected_count,
                )
                sam3_logger.add_detection_result(
                    image_num=image_num,
                    object_id=object_index + 1,
                    desc_type='detailed_en',
                    prompt=prompt_detailed,
                    detected_count=detected_count,
                    scores=result.scores.tolist() if result else [],
                    boxes=result.boxes.tolist() if result else [],
                )

            if result and result.count > 0:
                print(f"[+] 成功检测到 {result.count} 个物体 (耗时 {duration:.3f}s)")
                if sam3_logger:
                    sam3_logger.log_result_detail(
                        image_filename=f"{base_filename}_single_obj_{object_index+1}_detailed",
                        prompt=prompt_detailed,
                        scores=result.scores,
                        boxes=result.boxes,
                    )
                draw_sam3_results_on_image(
                    output_image_path,
                    result,
                    prompt_detailed[:30],
                    "_detailed_detected",
                )
            else:
                print(f"[!] 未检测到任何物体 (耗时 {duration:.3f}s)")

        except Exception as e:
            print(f"[X] 详细描述检测出错: {e}")
            if sam3_logger:
                sam3_logger.log_error(f"单物体详细描述检测出错: {e}")

        return True

    except Exception as e:
        print(f"[X] 处理单物体检测时出错: {e}")
        if sam3_logger:
            sam3_logger.log_error(f"处理单物体检测时出错: {e}")
        traceback.print_exc()
        return False


# ================================================================
# 批量处理入口
# ================================================================

def process_selected_images(json_path, image_dir, selected_image_numbers, output_dir, log_file,
                            confidence_threshold=0.3, enable_misdetect_test=True):
    """
    处理选定的图像集合

    参数:
        json_path (str): 描述文件路径
        image_dir (str): 图像目录路径
        selected_image_numbers (list): 要处理的图像编号列表 (1-based)
        output_dir (str): 输出目录路径
        log_file (str): 日志文件路径
        confidence_threshold (float): 置信度阈值
        enable_misdetect_test (bool): 是否启用误检测测试（下一张图提示词测当前图）
    """
    global sam3_logger, sam3_segmentor

    # 初始化日志记录器
    sam3_logger = SAM3Logger(log_file)
    sam3_logger.log_info("=== 开始 SAM3 批量语义分割 ===")

    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"[+] 创建输出目录: {output_dir}")

        # 初始化 SAM3 模型 (只加载一次)
        if sam3_segmentor is None:
            print("[*] 正在加载 SAM3 模型...")
            sam3_segmentor = SAM3Segmentor(options={"confidence_threshold": confidence_threshold})
            sam3_logger.log_info(f"SAM3 模型加载完成 (confidence_threshold={confidence_threshold})")

        # 读取 JSON 数据
        print("[*] 读取描述文件...")
        data = load_description_data(json_path)
        print(f"[+] 成功读取 {len(data)} 张图的描述信息")

        # 统计成功和失败
        success_count = 0
        failed_count = 0

        for selected_idx, image_num in enumerate(selected_image_numbers):
            idx = image_num - 1  # 转换为 0-based 索引

            if idx >= len(data) or idx < 0:
                print(f"[!] 图像编号 {image_num} 超出范围 (共有 {len(data)} 张图)")
                failed_count += 1
                continue

            image_info = data[idx]
            image_filename = image_info['image_filename']
            image_path = os.path.join(image_dir, image_filename)

            negative_prompt_info = None
            if enable_misdetect_test and len(selected_image_numbers) > 0:
                next_image_num = selected_image_numbers[(selected_idx + 1) % len(selected_image_numbers)]
                next_idx = next_image_num - 1
                if 0 <= next_idx < len(data):
                    negative_prompt_info = data[next_idx]

            if process_image(image_path, image_info, output_dir, negative_prompt_info=negative_prompt_info):
                success_count += 1
            else:
                failed_count += 1

        # 输出处理统计
        print(f"\n{'='*70}")
        print(f"处理完成统计:")
        print(f"  - 成功: {success_count}/{len(selected_image_numbers)}")
        print(f"  - 失败: {failed_count}/{len(selected_image_numbers)}")
        print(f"  - 结果目录: {output_dir}")
        print(f"  - 日志文件: {log_file}")
        print(f"{'='*70}")

    except Exception as e:
        print(f"[X] 处理过程中发生错误: {e}")
        if sam3_logger:
            sam3_logger.log_error(f"处理过程中发生错误: {e}")
        traceback.print_exc()

    finally:
        if sam3_logger:
            sam3_logger.save_to_file()
            sam3_logger.save_detection_summary()


# ================================================================
# 主函数
# ================================================================

def main_single():
    """主函数 - 单物体测试模式 (仅测试一张图片中的一个物体)"""
    print("\n" + "=" * 70)
    print("SAM3 语义分割测试 (单物体模式)")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    json_path = os.path.join(script_dir, 'benchmark', 'example', 'discription.json')
    image_dir = os.path.join(script_dir, 'benchmark', 'example')
    output_dir = os.path.join(script_dir, 'sam3_result_single')
    log_file = os.path.join(output_dir, 'sam3_inference_log_single.txt')

    # 配置: 测试第 3 张图的第 1 个物体
    target_image_number = 3   # 1-based
    target_object_index = 0   # 0-based (第 1 个物体)
    confidence_threshold = 0.3
    enable_misdetect_test = True

    print(f"[*] 配置信息:")
    print(f"    - 脚本目录: {script_dir}")
    print(f"    - JSON 文件: {json_path}")
    print(f"    - 图像目录: {image_dir}")
    print(f"    - 输出目录: {output_dir}")
    print(f"    - 日志文件: {log_file}")
    print(f"    - 目标图像: 第 {target_image_number} 张")
    print(f"    - 目标物体: 第 {target_object_index + 1} 个")
    print(f"    - 置信度阈值: {confidence_threshold}")
    print(f"    - 模式: 单物体，分别用 short_en 和 detailed_en 检测")

    global sam3_logger, sam3_segmentor

    try:
        sam3_logger = SAM3Logger(log_file)
        sam3_logger.log_info("=== 开始单物体检测 ===")

        os.makedirs(output_dir, exist_ok=True)
        print(f"[+] 创建输出目录: {output_dir}")

        # 初始化 SAM3 模型
        if sam3_segmentor is None:
            print("[*] 正在加载 SAM3 模型...")
            sam3_segmentor = SAM3Segmentor(options={"confidence_threshold": confidence_threshold})
            sam3_logger.log_info(f"SAM3 模型加载完成 (confidence_threshold={confidence_threshold})")

        # 读取 JSON 数据
        print("[*] 读取描述文件...")
        data = load_description_data(json_path)
        print(f"[+] 成功读取 {len(data)} 张图的描述信息")

        # 验证图像编号
        idx = target_image_number - 1
        if idx >= len(data) or idx < 0:
            print(f"[!] 图像编号 {target_image_number} 超出范围 (共有 {len(data)} 张图)")
            return

        image_info = data[idx]
        image_filename = image_info['image_filename']
        image_path = os.path.join(image_dir, image_filename)

        if process_single_object_detection(image_path, image_info, target_object_index, output_dir):
            print(f"\n{'='*70}")
            print(f"[+] 单物体检测成功完成！")
            print(f"    - 图像: {image_filename}")
            print(f"    - 物体: {image_info['object_descriptions'][target_object_index]['short_en']}")
            print(f"    - 结果目录: {output_dir}")
            print(f"    - 日志文件: {log_file}")
            print(f"{'='*70}")

    except Exception as e:
        print(f"[X] 单物体检测过程中发生错误: {e}")
        if sam3_logger:
            sam3_logger.log_error(f"单物体检测过程中发生错误: {e}")
        traceback.print_exc()

    finally:
        if sam3_logger:
            sam3_logger.save_to_file()
            sam3_logger.save_detection_summary()


def main():
    """主函数 - 处理多张图片"""
    print("\n" + "=" * 70)
    print("SAM3 语义分割测试 (多张图片模式)")
    print("=" * 70)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    json_path = os.path.join(script_dir, 'benchmark', 'example', 'discription.json')
    image_dir = os.path.join(script_dir, 'benchmark', 'example')
    output_dir = os.path.join(script_dir, 'sam3_result')
    log_file = os.path.join(output_dir, 'sam3_inference_log.txt')

    # 要处理的图像编号 (1-based)
    selected_images = [3, 5, 8, 10, 11]
    confidence_threshold = 0.3
    enable_misdetect_test = True

    print(f"[*] 配置信息:")
    print(f"    - 脚本目录: {script_dir}")
    print(f"    - JSON 文件: {json_path}")
    print(f"    - 图像目录: {image_dir}")
    print(f"    - 输出目录: {output_dir}")
    print(f"    - 日志文件: {log_file}")
    print(f"    - 选定图像: {selected_images}")
    print(f"    - 置信度阈值: {confidence_threshold}")
    print(f"    - 误检测测试: {'开启' if enable_misdetect_test else '关闭'} (下一张图提示词测当前图)")
    print(f"    - 模式: 多张图片，每张图片分别用 short_en 和 detailed_en 检测")

    process_selected_images(json_path, image_dir, selected_images, output_dir, log_file,
                            confidence_threshold=confidence_threshold,
                            enable_misdetect_test=enable_misdetect_test)

    print(f"\n[+] 所有处理完成！")


if __name__ == "__main__":
    # 支持命令行参数选择模式
    if len(sys.argv) > 1 and sys.argv[1] == 'single':
        main_single()
    else:
        main()
