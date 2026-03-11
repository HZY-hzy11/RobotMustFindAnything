"""
模块名称: DINO-XSeek 目标检测异步调用与可视化工具
功能描述:
    1. 自动读取本地图片并编码为 Base64 格式。
    2. 调用 DINO-XSeek-1.0 API 创建异步目标检测任务。
    3. 自动轮询 API 获取任务状态，直至检测完成。
    4. 解析返回的 Bounding Box (BBox) 坐标。
    5. 使用 OpenCV 在原图上绘制检测框及标签，并将结果保存至本地。
使用前置条件:
    - 已安装依赖: pip install requests opencv-python numpy
"""

import os
import sys
import time
import base64
import requests
import cv2  
import numpy as np

# =================================================================
# 接口地址常量 (可供模块内所有函数使用)
# =================================================================
CREATE_TASK_URL = "https://api.deepdataspace.com/v2/task/dino_xseek/detection"
QUERY_TASK_URL_TEMPLATE = "https://api.deepdataspace.com/v2/task_status/{task_uuid}"

# =================================================================
# 核心功能接口 (支持被其他文件 Import 调用)
# =================================================================

def image_to_base64(file_path):
    """
    将本地图片文件转换为 Base64 编码字符串。
    
    参数:
        file_path (str): 本地图片文件的路径。
    返回:
        str: 包含 Data URI 前缀的 Base64 字符串。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到指定的图片文件: {file_path}")
        
    # 获取文件后缀并标准化处理
    ext = os.path.splitext(file_path)[-1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
        
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
    # API 要求格式: data:image/<type>;base64,<data>
    return f"data:image/{ext};base64,{encoded_string}"


def create_task(image_path, prompt_text):
    """
    提交检测请求到服务器，创建异步处理任务。自动从环境变量读取 Token。
    
    参数:
        image_path (str): 待检测的本地图片路径。
        prompt_text (str): 检测目标的文本描述 (Prompt)。
    返回:
        str: 服务器分配的任务唯一标识符 (task_uuid)。
    """
    # 自动获取并校验 API Token
    api_token = os.getenv("DINO_API_TOKEN")
    if not api_token:
        print("-" * 50)
        print("错误: 未检测到环境变量 'DINO_API_TOKEN'")
        print("-" * 50)
        print("请先设置您的 API Token：\n")
        print('export DINO_API_TOKEN="您的Token"')
        print("-" * 50)
        sys.exit(1)

    headers = {
        "Token": api_token,
        "Content-Type": "application/json"
    }
    
    print(f"[*] 步骤 1: 正在处理图片并读取 Base64 ({image_path})...")
    base64_image = image_to_base64(image_path)
    
    # 构造请求体
    payload = {
        "model": "DINO-XSeek-1.0",
        "image": base64_image,
        "prompt": {
            "type": "text",
            "text": prompt_text
        },
        "targets": ["bbox"]
    }
    
    print(f"[*] 步骤 2: 正在向服务器提交任务, Prompt: '{prompt_text}'...")
    response = requests.post(CREATE_TASK_URL, headers=headers, json=payload)
    response.raise_for_status() # 若响应状态码不是 200，抛出异常
    
    resp_data = response.json()
    
    if resp_data.get("code") == 0:
        # API 响应结构可能在 data 内或顶层包含 uuid
        task_uuid = resp_data.get("data", {}).get("task_uuid") or resp_data.get("task_uuid")
            
        if task_uuid:
            print(f"[+] 任务提交成功! UUID: {task_uuid}")
            return task_uuid
        else:
            raise ValueError(f"API 返回成功但未找到 task_uuid。完整返回: {resp_data}")
    else:
        raise RuntimeError(f"API 逻辑错误: {resp_data.get('msg')}")


def poll_and_parse_result(task_uuid, timeout=120, poll_interval=3):
    """
    循环查询任务状态，直到任务完成、失败或超时。自动从环境变量读取 Token。
    
    参数:
        task_uuid (str): 任务 ID。
        timeout (int): 最大等待时间（秒）。
        poll_interval (int): 每次查询之间的间隔。
    返回:
        list: 检测到的对象列表，包含 bbox 等信息。
    """
    # 自动获取并校验 API Token
    api_token = os.getenv("DINO_API_TOKEN")
    if not api_token:
        print("-" * 50)
        print("错误: 调用查询接口时未检测到环境变量 'DINO_API_TOKEN'")
        sys.exit(1)

    headers = {"Token": api_token}
    query_url = QUERY_TASK_URL_TEMPLATE.format(task_uuid=task_uuid)
    
    start_time = time.time()
    print(f"[*] 步骤 3: 正在轮询任务结果 (最长等待 {timeout} 秒)...")
    
    while time.time() - start_time < timeout:
        response = requests.get(query_url, headers=headers)
        response.raise_for_status()
        resp_data = response.json()
        
        if resp_data.get("code") == 0:
            data = resp_data.get("data", {})
            status = data.get("status")
            
            if status == "success":
                print("[+] 任务已完成处理！")
                objects = data.get("result", {}).get("objects", [])
                print(f"[i] 共检测到 {len(objects)} 个目标。")
                return objects
                
            elif status in ["failed", "error"]:
                error_msg = data.get("error", "未知错误")
                raise RuntimeError(f"服务器处理任务失败: {error_msg}")
            else:
                # 状态通常为 pending(排队中) 或 running(运行中)
                print(f"    - 状态: {status}... 等待 {poll_interval}s")
                time.sleep(poll_interval)
        else:
            raise RuntimeError(f"查询异常: {resp_data}")
            
    raise TimeoutError(f"任务在 {timeout} 秒内未完成。")


def draw_bboxes_on_image_opencv(image_path, objects, output_prefix="_detected"):
    """
    使用 OpenCV 在图像上绘制矩形框和文字标签。
    
    参数:
        image_path (str): 原始图片路径。
        objects (list): 包含 bbox 坐标的列表。
        output_prefix (str): 保存文件时添加的文件名后缀。
    """
    # 加载原图
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法打开图片进行绘图: {image_path}")

    # 绘图配置
    color = (0, 255, 0)       # 边框颜色: 绿色 (BGR格式)
    thickness = 2             # 边框粗细
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for idx, obj in enumerate(objects, start=1):
        bbox = obj.get("bbox") # 格式通常为 [xmin, ymin, xmax, ymax]
        if bbox and len(bbox) == 4:
            # 坐标转换与取整 (OpenCV 绘图要求整数坐标)
            xmin, ymin, xmax, ymax = map(int, bbox)
            
            # 画矩形框
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)

            # 绘制标签背景与文字
            label = f"#{idx}"
            cv2.putText(image, label, (xmin, max(ymin - 10, 20)), font, 0.6, color, 2)

    # 构造输出路径，例: test.jpg -> test_detected.jpg
    base, ext = os.path.splitext(image_path)
    output_path = f"{base}{output_prefix}{ext}"
    
    cv2.imwrite(output_path, image)
    print(f"[+] 可视化结果已保存至: {output_path}")

# =================================================================
# 本地直接运行时的测试入口
# =================================================================
if __name__ == "__main__":
    # 定义测试用的本地变量
    TEST_IMAGE_PATH = "2_resized.jpg"  
    TEST_PROMPT_TEXT = "平铺散落在床边木质柜子平整台面上的几张淡黄色的方形薄纸片"  

    try:
        # 提交任务
        task_id = create_task(TEST_IMAGE_PATH, TEST_PROMPT_TEXT)
        # 轮询获取结果
        detected_objects = poll_and_parse_result(task_id)

        if detected_objects:
            print(f"[*] 正在在原图上绘制 {len(detected_objects)} 个 BBox...")
            draw_bboxes_on_image_opencv(TEST_IMAGE_PATH, detected_objects)
        else:
            print("[!] 未发现匹配目标的 BBox 结果。")

    except Exception as e:
        print(f"\n[X] 脚本运行中发生错误: {e}")