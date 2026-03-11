"""
模块名称: DINOX 目标检测测试脚本
功能描述:
    1. 读取 ./benchmark/example/discription.json 中的内容
    2. 选择指定图像 (3, 5, 8, 10, 11)
    3. 提取这些图像中所有物体的 short_en 和 detailed_en 描述
    4. 调用 DINOX 接口对相应的图像进行语言描述的目标检测
    5. 保存检测结果并绘制 Bounding Box
    6. 将结果图片保存到 ./dinox_result 目录
    7. 记录所有 API 请求和响应到日志文件（去除敏感信息）
"""

import json
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# 添加 dinox 模块到路径
sys.path.insert(0, os.path.dirname(__file__))
from dinox.dinox import create_task as dinox_create_task
from dinox.dinox import poll_and_parse_result as dinox_poll_and_parse_result
from dinox.dinox import draw_bboxes_on_image_opencv


# ================================================================
# API 日志记录配置
# ================================================================

class APILogger:
    """API 请求/响应日志记录器，自动过滤敏感信息"""
    
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
        
        格式示例:
        - "3.jpg_short_en" -> (3, None, "short_en")
        - "3_obj1_short" -> (3, 1, "short_en")
        - "3_obj1_detailed" -> (3, 1, "detailed_en")
        - "3.jpg_single_obj_1_short" -> (3, 1, "short_en")
        - "3.jpg_single_obj_1_detailed" -> (3, 1, "detailed_en")
        
        返回:
            tuple: (image_num, object_num, desc_type)
        """
        import re
        
        # 尝试提取带.jpg的单物体模式: 3.jpg_single_obj_1_short/detailed
        match = re.search(r'(\d+)\.jpg_single_obj_(\d+)_(short|detailed)', image_filename)
        if match:
            image_num = match.group(1)
            object_num = match.group(2)
            desc_type = "short_en" if match.group(3) == "short" else "detailed_en"
            return image_num, object_num, desc_type
        
        # 尝试提取单物体模式: 3_obj1_short/detailed
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
        
        # 默认返回
        return None, None, None
        
    def _generate_entry_name(self, log_entry):
        """
        生成日志条目的名称
        
        返回:
            str: 日志条目名称，格式如: "3_1_short_1" 或 "3_detailed_2"
        """
        # 对于 INFO 和 ERROR 日志，使用 message 内容作为名称
        if log_entry['type'] in ['INFO', 'ERROR']:
            message = log_entry.get('message', '').strip()
            if message:
                # 提取前20个字符，去除===符号
                name_part = message.replace('===', '').strip()[:20]
                return f"{log_entry['type']}_{name_part}" if name_part else log_entry['type']
            return log_entry['type']
        
        # 获取 image_file
        image_file = log_entry.get('image_file')
        
        # 如果当前日志没有 image_file，向前查找
        if not image_file:
            # 对于 RESPONSE 或轮询类请求，从前一个 REQUEST 获取 image_file
            current_index = self.logs.index(log_entry)
            for prev_log in reversed(self.logs[:current_index]):
                if prev_log['type'] == 'REQUEST' and 'image_file' in prev_log:
                    prev_image_file = prev_log.get('image_file', '')
                    # 优先使用非poll的请求，但如果找不到则也接受poll
                    if 'poll_#' not in prev_image_file:
                        image_file = prev_image_file
                        break
                    # 如果是poll请求，继续向前找POST请求
                    elif not image_file:  # 只有当还没找到时才用poll的
                        # 从这个poll请求再向前找POST请求
                        for prev_prev_log in reversed(self.logs[:self.logs.index(prev_log)]):
                            if prev_prev_log['type'] == 'REQUEST' and prev_prev_log.get('method') == 'POST' and 'image_file' in prev_prev_log:
                                image_file = prev_prev_log['image_file']
                                break
                        if image_file:
                            break
        
        # 如果还是找不到有效的 image_file，使用默认命名
        if not image_file:
            return f"{log_entry['type']}_{log_entry.get('method', 'N/A')}"
        
        image_num, object_num, desc_type = self._parse_image_info(image_file)
        
        # 如果无法解析文件名，使用默认命名
        if image_num is None or desc_type is None:
            return f"{log_entry['type']}_{log_entry.get('method', 'N/A')}"
        
        # 构建条目名称
        desc_type_short = desc_type.replace('_en', '')
        if object_num is not None:
            return f"{image_num}_{object_num}_{desc_type_short}"
        else:
            return f"{image_num}_{desc_type_short}"
        
    def log_request(self, url, method, headers, payload, image_filename):
        """
        记录 API 请求（过滤敏感信息）
        
        参数:
            url (str): 请求 URL
            method (str): HTTP 方法
            headers (dict): 请求头
            payload (dict): 请求体
            image_filename (str): 图像文件名
        """
        timestamp = datetime.now().isoformat()
        
        # 复制并清理数据以隐藏敏感信息
        safe_headers = headers.copy()
        if "Token" in safe_headers:
            safe_headers["Token"] = "[REDACTED_TOKEN]"
        
        safe_payload = payload.copy()
        if "image" in safe_payload:
            safe_payload["image"] = f"[BASE64_IMAGE_{image_filename}]"
        
        log_entry = {
            "timestamp": timestamp,
            "type": "REQUEST",
            "method": method,
            "url": url,
            "headers": safe_headers,
            "payload": safe_payload,
            "image_file": image_filename
        }
        
        self.logs.append(log_entry)
        print(f"[API] {timestamp} - REQUEST: {method} {url}")
        
    def log_response(self, status_code, response_data):
        """
        记录 API 响应
        
        参数:
            status_code (int): HTTP 状态码
            response_data (dict): 响应数据
        """
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "type": "RESPONSE",
            "status_code": status_code,
            "data": response_data
        }
        
        self.logs.append(log_entry)
        print(f"[API] {timestamp} - RESPONSE: Status {status_code}")
        
    def log_error(self, error_msg):
        """
        记录错误信息
        
        参数:
            error_msg (str): 错误消息
        """
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "type": "ERROR",
            "message": error_msg
        }
        
        self.logs.append(log_entry)
        print(f"[API] {timestamp} - ERROR: {error_msg}")
    
    def log_info(self, info_msg):
        """
        记录信息
        
        参数:
            info_msg (str): 信息内容
        """
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "type": "INFO",
            "message": info_msg
        }
        
        self.logs.append(log_entry)
        print(f"[API] {timestamp} - INFO: {info_msg}")
        
    def save_to_file(self):
        """将所有日志保存到文件"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DINOX API 请求/响应日志记录\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().isoformat()}\n")
            f.write(f"总条目数: {len(self.logs)}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, log in enumerate(self.logs, 1):
                # 生成条目名称
                entry_name = self._generate_entry_name(log)
                f.write(f"\n【{entry_name}】\n")
                f.write(f"时间: {log['timestamp']}\n")
                f.write(f"类型: {log['type']}\n")
                
                if log['type'] == 'REQUEST':
                    f.write(f"方法: {log['method']}\n")
                    f.write(f"URL: {log['url']}\n")
                    f.write(f"请求头:\n")
                    for key, value in log['headers'].items():
                        f.write(f"    {key}: {value}\n")
                    
                    # 仅对 POST 请求显示请求体和图像文件
                    if log['method'] == 'POST':
                        f.write(f"请求体:\n")
                        f.write(json.dumps(log['payload'], indent=2, ensure_ascii=False))
                        f.write(f"\n图像文件: {log['image_file']}\n")
                    
                elif log['type'] == 'RESPONSE':
                    f.write(f"状态码: {log['status_code']}\n")
                    f.write(f"响应数据:\n")
                    f.write(json.dumps(log['data'], indent=2, ensure_ascii=False))
                    f.write("\n")
                    
                    # 检查是否检测结果为空
                    data = log.get('data', {})
                    if data.get('code') == 0:
                        result = data.get('data', {}).get('result', {})
                        if isinstance(result, dict):
                            objects = result.get('objects', [])
                            if isinstance(objects, list) and len(objects) == 0:
                                f.write("\n⚠️  检测结果为空：未检测到任何物体\n")
                    
                elif log['type'] == 'ERROR':
                    f.write(f"错误: {log['message']}\n")                    
                elif log['type'] == 'INFO':
                    f.write(f"信息: {log['message']}\n")                
                f.write("\n" + "-" * 80 + "\n")
        
        print(f"\n[+] API 日志已保存到: {self.log_file}")
    
    def add_detection_result(self, image_num, object_id, desc_type, prompt, bbox_result):
        """
        添加检测结果到统计列表
        
        参数:
            image_num (int): 图像编号
            object_id (int): 物体编号
            desc_type (str): 描述类型 (short_en/detailed_en)
            prompt (str): 检测提示词
            bbox_result (list): 检测到的bbox列表
        """
        self.detection_results.append({
            'image_num': image_num,
            'object_id': object_id,
            'description_type': desc_type,
            'prompt': prompt,
            'detected_count': len(bbox_result) if bbox_result else 0,
            'bboxes': bbox_result if bbox_result else [],
            'status': 'success' if bbox_result and len(bbox_result) > 0 else 'empty',
            'timestamp': datetime.now().isoformat()
        })
    
    def save_detection_summary(self):
        """保存检测结果统计到JSON文件"""
        summary_file = self.log_file.replace('.txt', '_summary.json')
        
        # 统计信息
        total_detections = len(self.detection_results)
        successful_detections = sum(1 for r in self.detection_results if r['status'] == 'success')
        empty_detections = sum(1 for r in self.detection_results if r['status'] == 'empty')
        
        summary_data = {
            'generated_time': datetime.now().isoformat(),
            'statistics': {
                'total_detections': total_detections,
                'successful_detections': successful_detections,
                'empty_detections': empty_detections,
                'success_rate': f"{(successful_detections/total_detections*100):.2f}%" if total_detections > 0 else '0%'
            },
            'detection_results': self.detection_results
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"[+] 检测结果统计已保存到: {summary_file}")
        print(f"    - 总检测次数: {total_detections}")
        print(f"    - 成功检测: {successful_detections}")
        print(f"    - 未检测到: {empty_detections}")
        print(f"    - 成功率: {summary_data['statistics']['success_rate']}")


# 全局 API 日志记录器
api_logger = None


def create_task_with_logging(image_path, prompt_text, image_filename):
    """
    包装的 create_task 函数，自动记录 API 请求/响应
    
    参数:
        image_path (str): 图像路径
        prompt_text (str): 检测提示词
        image_filename (str): 图像文件名
    返回:
        str: 任务 UUID
    """
    global api_logger
    
    try:
        # 记录请求前的准备
        from dinox.dinox import CREATE_TASK_URL, image_to_base64
        import requests
        import os
        
        api_token = os.getenv("DINO_API_TOKEN")
        headers = {
            "Token": api_token,
            "Content-Type": "application/json"
        }
        
        base64_image = image_to_base64(image_path)
        
        payload = {
            "model": "DINO-XSeek-1.0",
            "image": base64_image,
            "prompt": {
                "type": "text",
                "text": prompt_text
            },
            "targets": ["bbox"]
        }
        
        # 记录请求
        if api_logger:
            api_logger.log_request(CREATE_TASK_URL, "POST", headers, payload, image_filename)
        
        # 发送请求
        response = requests.post(CREATE_TASK_URL, headers=headers, json=payload)
        response.raise_for_status()
        resp_data = response.json()
        
        # 记录响应
        if api_logger:
            api_logger.log_response(response.status_code, resp_data)
        
        if resp_data.get("code") == 0:
            task_uuid = resp_data.get("data", {}).get("task_uuid") or resp_data.get("task_uuid")
            if task_uuid:
                return task_uuid
            else:
                raise ValueError(f"API 返回成功但未找到 task_uuid。完整返回: {resp_data}")
        else:
            raise RuntimeError(f"API 逻辑错误: {resp_data.get('msg')}")
            
    except Exception as e:
        if api_logger:
            api_logger.log_error(str(e))
        raise


def poll_and_parse_result_with_logging(task_uuid, timeout=120, poll_interval=3):
    """
    包装的 poll_and_parse_result 函数，自动记录 API 请求/响应
    
    参数:
        task_uuid (str): 任务 UUID
        timeout (int): 超时时间
        poll_interval (int): 轮询间隔
    返回:
        list: 检测对象列表
    """
    global api_logger
    
    try:
        from dinox.dinox import QUERY_TASK_URL_TEMPLATE
        import requests
        import os
        import time
        
        api_token = os.getenv("DINO_API_TOKEN")
        headers = {"Token": api_token}
        query_url = QUERY_TASK_URL_TEMPLATE.format(task_uuid=task_uuid)
        
        start_time = time.time()
        poll_count = 0
        
        while time.time() - start_time < timeout:
            poll_count += 1
            
            # 记录轮询请求
            if api_logger:
                api_logger.log_request(query_url, "GET", headers, {"task_uuid": task_uuid}, f"poll_#{poll_count}")
            
            response = requests.get(query_url, headers=headers)
            response.raise_for_status()
            resp_data = response.json()
            
            # 记录响应
            if api_logger:
                api_logger.log_response(response.status_code, resp_data)
            
            if resp_data.get("code") == 0:
                data = resp_data.get("data", {})
                status = data.get("status")
                
                if status == "success":
                    objects = data.get("result", {}).get("objects", [])
                    return objects
                    
                elif status in ["failed", "error"]:
                    error_msg = data.get("error", "未知错误")
                    raise RuntimeError(f"服务器处理任务失败: {error_msg}")
                else:
                    time.sleep(poll_interval)
            else:
                raise RuntimeError(f"查询异常: {resp_data}")
                
        raise TimeoutError(f"任务在 {timeout} 秒内未完成。")
        
    except Exception as e:
        if api_logger:
            api_logger.log_error(str(e))
        raise


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


def build_detection_prompt(scene_description, objects):
    """
    构建检测提示词（使用 short_en 和 detailed_en）
    
    参数:
        objects (list): 物体列表，包含 short_en 和 detailed_en
    返回:
        str: 构建好的提示词
    """
    object_descriptions = []
    for obj in objects:
        short_en = obj.get('short_en', '')
        detailed_en = obj.get('detailed_en', '')
        object_descriptions.append(f"- {short_en}: {detailed_en}")
    
    prompt = "\n".join(object_descriptions)
    return prompt


def build_detection_prompt_short_only(scene_description, objects):
    """
    构建检测提示词（仅使用 short_en）
    
    参数:
        objects (list): 物体列表
    返回:
        str: 构建好的提示词
    """
    object_descriptions = []
    for obj in objects:
        short_en = obj.get('short_en', '')
        object_descriptions.append(f"- {short_en}")
    
    prompt = "\n".join(object_descriptions)
    return prompt


def build_detection_prompt_detailed_only(scene_description, objects):
    """
    构建检测提示词（仅使用 detailed_en）
    
    参数:
        objects (list): 物体列表
    返回:
        str: 构建好的提示词
    """
    object_descriptions = []
    for obj in objects:
        detailed_en = obj.get('detailed_en', '')
        object_descriptions.append(f"- {detailed_en}")
    
    prompt = "\n".join(object_descriptions)
    return prompt


def process_image(image_path, image_info, output_dir):
    """
    处理单张图像的目标检测（对每个物体分别使用 short_en 和 detailed_en）
    
    参数:
        image_path (str): 图像文件路径
        image_info (dict): 从 JSON 中读取的图像信息
        output_dir (str): 输出目录路径
    """
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
        
        # 修改图像路径以保存到输出目录
        base_filename = os.path.basename(image_path)
        output_image_path = os.path.join(output_dir, base_filename)
        
        # 复制原图到输出目录
        import shutil
        if not os.path.exists(output_image_path):
            shutil.copy(image_path, output_image_path)
        
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
            prompt_short = f"- {obj['short_en']}"
            print(f"    提示词: {prompt_short}")
            
            try:
                task_id = create_task_with_logging(
                    output_image_path, 
                    prompt_short, 
                    f"{base_filename}_obj{obj_idx}_short"
                )
                detected_objects = poll_and_parse_result_with_logging(task_id, timeout=300, poll_interval=5)
                
                # 记录检测结果到统计
                if api_logger:
                    # 从文件名提取图像编号
                    import re
                    match = re.search(r'(\d+)\.jpg', base_filename)
                    image_num = int(match.group(1)) if match else 0
                    api_logger.add_detection_result(
                        image_num=image_num,
                        object_id=obj_idx,
                        desc_type='short_en',
                        prompt=prompt_short,
                        bbox_result=detected_objects
                    )
                
                if detected_objects:
                    print(f"    [+] 检测成功: 检测到 {len(detected_objects)} 个物体")
                    # 在图像上绘制 bbox
                    draw_bboxes_on_image_opencv(
                        output_image_path, 
                        detected_objects, 
                        f"_obj{obj_idx}_short"
                    )
                else:
                    print(f"    [!] 未检测到任何物体")
            except Exception as e:
                print(f"    [X] 检测出错: {e}")
                if api_logger:
                    api_logger.log_error(f"物体 {obj_idx} short_en 检测出错: {e}")
            
            # ====== 使用 detailed_en 检测 ======
            print(f"\n  [检测 {obj_idx * 2}/{len(objects) * 2}] 使用 detailed_en 进行检测")
            prompt_detailed = f"- {obj['detailed_en']}"
            print(f"    提示词: {prompt_detailed}")
            
            try:
                task_id = create_task_with_logging(
                    output_image_path, 
                    prompt_detailed, 
                    f"{base_filename}_obj{obj_idx}_detailed"
                )
                detected_objects = poll_and_parse_result_with_logging(task_id, timeout=300, poll_interval=5)
                
                # 记录检测结果到统计
                if api_logger:
                    # 从文件名提取图像编号
                    import re
                    match = re.search(r'(\d+)\.jpg', base_filename)
                    image_num = int(match.group(1)) if match else 0
                    api_logger.add_detection_result(
                        image_num=image_num,
                        object_id=obj_idx,
                        desc_type='detailed_en',
                        prompt=prompt_detailed,
                        bbox_result=detected_objects
                    )
                
                if detected_objects:
                    print(f"    [+] 检测成功: 检测到 {len(detected_objects)} 个物体")
                    # 在图像上绘制 bbox
                    draw_bboxes_on_image_opencv(
                        output_image_path, 
                        detected_objects, 
                        f"_obj{obj_idx}_detailed"
                    )
                else:
                    print(f"    [!] 未检测到任何物体")
            except Exception as e:
                print(f"    [X] 检测出错: {e}")
                if api_logger:
                    api_logger.log_error(f"物体 {obj_idx} detailed_en 检测出错: {e}")
        
        print(f"\n{'='*70}")
        print(f"[+] 图像 {image_filename} 处理完成！")
        print(f"{'='*70}")
        return True
            
    except Exception as e:
        print(f"[X] 处理图像时出错: {e}")
        if api_logger:
            api_logger.log_error(f"处理图像 {image_filename} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_single_object_detection(image_path, image_info, object_index, output_dir):
    """
    处理单张图像中单个物体的目标检测
    
    参数:
        image_path (str): 图像文件路径
        image_info (dict): 从 JSON 中读取的图像信息
        object_index (int): 要检测的物体索引 (0-based)
        output_dir (str): 输出目录路径
    """
    if not os.path.exists(image_path):
        print(f"[!] 警告: 图像文件不存在: {image_path}")
        return False
    
    image_filename = image_info['image_filename']
    print(f"\n{'='*70}")
    print(f"处理图像: {image_filename}")
    print(f"{'='*70}")
    
    try:
        # 获取选定的物体
        objects = image_info['object_descriptions']
        if object_index >= len(objects) or object_index < 0:
            print(f"[!] 物体索引 {object_index} 超出范围 (共有 {len(objects)} 个物体)")
            return False
        
        target_object = objects[object_index]
        scene_desc = image_info['scene_understanding'].get('description_en', '')
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 修改图像路径以保存到输出目录
        base_filename = os.path.basename(image_path)
        base_without_ext = os.path.splitext(base_filename)[0]
        ext = os.path.splitext(base_filename)[1]
        output_image_path = os.path.join(output_dir, f"{base_without_ext}_obj{object_index+1}{ext}")
        
        # 复制原图到输出目录
        import shutil
        shutil.copy(image_path, output_image_path)
        
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
        
        prompt_short = f"Find this object: {target_object['short_en']}"
        print(f"[*] 检测提示词: {prompt_short}")
        
        try:
            task_id = create_task_with_logging(
                output_image_path, 
                prompt_short, 
                f"{base_filename}_single_obj_{object_index+1}_short"
            )
            detected_objects = poll_and_parse_result_with_logging(task_id, timeout=300, poll_interval=5)
            
            # 记录检测结果到统计
            if api_logger:
                # 从文件名提取图像编号
                import re
                match = re.search(r'(\d+)\.jpg', base_filename)
                image_num = int(match.group(1)) if match else 0
                api_logger.add_detection_result(
                    image_num=image_num,
                    object_id=object_index + 1,
                    desc_type='short_en',
                    prompt=prompt_short,
                    bbox_result=detected_objects
                )
            
            if detected_objects:
                print(f"[+] 成功检测到 {len(detected_objects)} 个物体")
                draw_bboxes_on_image_opencv(
                    output_image_path, 
                    detected_objects, 
                    "_short_detected"
                )
            else:
                print(f"[!] 未检测到任何物体")
        except Exception as e:
            print(f"[X] 短描述检测出错: {e}")
            if api_logger:
                api_logger.log_error(f"单物体短描述检测出错: {e}")
        
        # ============================================================
        # 单物体检测 - detailed_en
        # ============================================================
        print(f"\n{'='*70}")
        print(f"[单物体检测 - detailed_en] 使用详细描述")
        print("-" * 70)
        
        prompt_detailed = f"Find this object: {target_object['detailed_en']}"
        print(f"[*] 检测提示词: {prompt_detailed}")
        
        try:
            task_id = create_task_with_logging(
                output_image_path, 
                prompt_detailed, 
                f"{base_filename}_single_obj_{object_index+1}_detailed"
            )
            detected_objects = poll_and_parse_result_with_logging(task_id, timeout=300, poll_interval=5)
            
            # 记录检测结果到统计
            if api_logger:
                # 从文件名提取图像编号
                import re
                match = re.search(r'(\d+)\.jpg', base_filename)
                image_num = int(match.group(1)) if match else 0
                api_logger.add_detection_result(
                    image_num=image_num,
                    object_id=object_index + 1,
                    desc_type='detailed_en',
                    prompt=prompt_detailed,
                    bbox_result=detected_objects
                )
            
            if detected_objects:
                print(f"[+] 成功检测到 {len(detected_objects)} 个物体")
                draw_bboxes_on_image_opencv(
                    output_image_path, 
                    detected_objects, 
                    "_detailed_detected"
                )
            else:
                print(f"[!] 未检测到任何物体")
        except Exception as e:
            print(f"[X] 详细描述检测出错: {e}")
            if api_logger:
                api_logger.log_error(f"单物体详细描述检测出错: {e}")
        
        return True
        
    except Exception as e:
        print(f"[X] 处理单物体检测时出错: {e}")
        if api_logger:
            api_logger.log_error(f"处理单物体检测时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_selected_images(json_path, image_dir, selected_image_numbers, output_dir, log_file):
    """
    处理选定的图像集合
    
    参数:
        json_path (str): 描述文件路径
        image_dir (str): 图像目录路径
        selected_image_numbers (list): 要处理的图像编号列表 (1-based, 如 [3, 5, 8, 10, 11])
        output_dir (str): 输出目录路径
        log_file (str): 日志文件路径
    """
    global api_logger
    
    # 初始化 API 日志记录器
    api_logger = APILogger(log_file)
    
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"[+] 创建输出目录: {output_dir}")
        
        # 读取 JSON 数据
        print("[*] 读取描述文件...")
        data = load_description_data(json_path)
        print(f"[+] 成功读取 {len(data)} 张图的描述信息")
        
        # 统计成功和失败
        success_count = 0
        failed_count = 0
        
        # 转换为 0-based 索引并处理
        for image_num in selected_image_numbers:
            idx = image_num - 1  # 转换为 0-based 索引
            
            if idx >= len(data) or idx < 0:
                print(f"[!] 图像编号 {image_num} 超出范围 (共有 {len(data)} 张图)")
                failed_count += 1
                continue
            
            image_info = data[idx]
            image_filename = image_info['image_filename']
            image_path = os.path.join(image_dir, image_filename)
            
            # 处理图像
            if process_image(image_path, image_info, output_dir):
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
        if api_logger:
            api_logger.log_error(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 保存日志和统计结果
        if api_logger:
            api_logger.save_to_file()
            api_logger.save_detection_summary()

def main_single():
    """主函数 - 单物体测试模式 (仅测试一张图片中的一个物体)"""
    print("\n" + "="*70)
    print("DINOX 目标检测测试 (单物体模式)")
    print("="*70)
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # JSON 文件路径
    json_path = os.path.join(script_dir, 'benchmark', 'example', 'discription.json')
    
    # 图像目录路径
    image_dir = os.path.join(script_dir, 'benchmark', 'example')
    
    # 输出目录路径（检测结果图片）
    output_dir = os.path.join(script_dir, 'dinox_result_single')
    
    # 日志文件路径
    log_file = os.path.join(output_dir, 'api_requests_log_single.txt')
    
    # 配置: 测试第 3 张图的第 1 个物体
    target_image_number = 3  # 1-based
    target_object_index = 0  # 0-based (第 1 个物体)
    
    print(f"[*] 配置信息:")
    print(f"    - 脚本目录: {script_dir}")
    print(f"    - JSON 文件: {json_path}")
    print(f"    - 图像目录: {image_dir}")
    print(f"    - 输出目录: {output_dir}")
    print(f"    - 日志文件: {log_file}")
    print(f"    - 目标图像: 第 {target_image_number} 张")
    print(f"    - 目标物体: 第 {target_object_index + 1} 个")
    print(f"    - 模式: 单物体，分别用 short_en 和 detailed_en 检测")
    
    try:
        # 初始化 API 日志记录器
        global api_logger
        api_logger = APILogger(log_file)
        api_logger.log_info("=== 开始单物体检测 ===")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"[+] 创建输出目录: {output_dir}")
        
        # 读取 JSON 数据
        print("[*] 读取描述文件...")
        data = load_description_data(json_path)
        print(f"[+] 成功读取 {len(data)} 张图的描述信息")
        
        # 验证图像编号和物体索引
        idx = target_image_number - 1
        if idx >= len(data) or idx < 0:
            print(f"[!] 图像编号 {target_image_number} 超出范围 (共有 {len(data)} 张图)")
            return
        
        image_info = data[idx]
        image_filename = image_info['image_filename']
        image_path = os.path.join(image_dir, image_filename)
        
        # 处理单个物体
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
        if api_logger:
            api_logger.log_error(f"单物体检测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 保存日志和统计结果
        if api_logger:
            api_logger.save_to_file()
            api_logger.save_detection_summary()


def main():
    """主函数 - 处理多张图片"""
    print("\n" + "="*70)
    print("DINOX 目标检测测试 (多张图片模式)")
    print("="*70)
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # JSON 文件路径
    json_path = os.path.join(script_dir, 'benchmark', 'example', 'discription.json')
    
    # 图像目录路径
    image_dir = os.path.join(script_dir, 'benchmark', 'example')
    
    # 输出目录路径（检测结果图片）
    output_dir = os.path.join(script_dir, 'dinox_result')
    
    # 日志文件路径
    log_file = os.path.join(output_dir, 'api_requests_log.txt')
    
    # 要处理的图像编号 (1-based)
    # selected_images = [3, 5, 8, 10, 11]
    selected_images = [8, 10, 11]
    
    print(f"[*] 配置信息:")
    print(f"    - 脚本目录: {script_dir}")
    print(f"    - JSON 文件: {json_path}")
    print(f"    - 图像目录: {image_dir}")
    print(f"    - 输出目录: {output_dir}")
    print(f"    - 日志文件: {log_file}")
    print(f"    - 选定图像: {selected_images}")
    print(f"    - 模式: 多张图片，每张图片分别用 short_en 和 detailed_en 检测")
    
    # 处理选定的图像
    process_selected_images(json_path, image_dir, selected_images, output_dir, log_file)
    
    print(f"\n[+] 所有处理完成！")


if __name__ == "__main__":
    import sys
    
    # 支持命令行参数选择模式
    if len(sys.argv) > 1 and sys.argv[1] == 'single':
        # 单物体测试模式
        main_single()
    else:
        # 默认: 多张图片模式
        main()
