# DINO-X 目标检测模块

## 概述

本模块基于 [DINO-XSeek-1.0](https://cloud.deepdataspace.com/zh/playground/dino-x) API 实现了一个**开放词汇目标检测工具**，支持通过自然语言文本描述在图像中检测目标物体并返回边界框 (BBox) 坐标。

### 特性

- **开放词汇检测**：通过任意文本 prompt 描述目标物体，无需预定义类别
- **异步任务模式**：提交检测任务后自动轮询获取结果，支持超时控制
- **结果可视化**：使用 OpenCV 在原图上绘制检测框并保存
- **模块化设计**：核心函数支持被其他文件 import 调用

---

## 目录结构

```
rofa/dinox/
└── dinox.py          # DINO-X 目标检测主模块
rofa/
├── test_dinox.py     # 批量检测测试脚本（含日志记录）
└── dinox_result/     # 检测结果输出目录
```

---

## 快速开始

### 安装依赖

```bash
pip install requests opencv-python numpy
```

### 配置 API Key

```bash
export DINO_API_TOKEN="your_api_key_here"
```

> API Key 可从 [DINO-X 开放平台](https://cloud.deepdataspace.com/zh/playground/dino-x) 注册获取。

### 基本用法

```python
from dinox.dinox import create_task, poll_and_parse_result, draw_bboxes_on_image_opencv

# 1. 提交检测任务
task_id = create_task("image.jpg", "red cup on the table")

# 2. 轮询获取结果
detected_objects = poll_and_parse_result(task_id, timeout=120, poll_interval=0.3)

# 3. 可视化结果
if detected_objects:
    draw_bboxes_on_image_opencv("image.jpg", detected_objects)
    # 结果保存为 image_detected.jpg
```

### 命令行运行

直接运行模块进行测试：

```bash
python dinox.py
```

默认使用 `../benchmark/example/2.jpg` 作为测试图片。

---

## API 参考

### `image_to_base64(file_path)`

将本地图片转换为 Base64 编码字符串（含 Data URI 前缀）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `file_path` | `str` | 本地图片文件路径 |

**返回值**：`str` — `data:image/<type>;base64,<data>` 格式字符串

---

### `create_task(image_path, prompt_text)`

提交检测请求，创建异步处理任务。自动从环境变量 `DINO_API_TOKEN` 读取 Token。

| 参数 | 类型 | 说明 |
|------|------|------|
| `image_path` | `str` | 待检测的本地图片路径 |
| `prompt_text` | `str` | 检测目标的文本描述 |

**返回值**：`str` — 任务唯一标识符 (`task_uuid`)

---

### `poll_and_parse_result(task_uuid, timeout=120, poll_interval=0.3)`

循环查询任务状态，直到完成、失败或超时。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `task_uuid` | `str` | — | 任务 ID |
| `timeout` | `int` | `120` | 最大等待时间（秒） |
| `poll_interval` | `float` | `0.3` | 轮询间隔（秒） |

**返回值**：`list` — 检测到的对象列表，每个对象包含 `bbox` 字段 `[xmin, ymin, xmax, ymax]`

---

### `draw_bboxes_on_image_opencv(image_path, objects, output_prefix="_detected")`

在图像上绘制矩形框和编号标签，保存结果图片。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image_path` | `str` | — | 原始图片路径 |
| `objects` | `list` | — | 含 `bbox` 坐标的检测结果列表 |
| `output_prefix` | `str` | `"_detected"` | 输出文件名后缀 |

**输出**：保存为 `{原文件名}{output_prefix}.{扩展名}`

---

## 检测流程

```
输入: 本地图片路径 + 文本 Prompt
        │
        ▼
┌──────────────────────────────────┐
│  1. 图片 → Base64 编码           │  ← image_to_base64()
├──────────────────────────────────┤
│  2. POST 创建异步检测任务        │  ← DINO-XSeek-1.0 API
│     返回 task_uuid               │
├──────────────────────────────────┤
│  3. 轮询 GET 查询任务状态        │  ← pending → running → success
│     直到完成/失败/超时            │
├──────────────────────────────────┤
│  4. 解析返回的 objects 列表      │  ← 提取 bbox 坐标
├──────────────────────────────────┤
│  5. OpenCV 绘制检测框并保存      │  ← 绿色矩形框 + 编号标签
└──────────────────────────────────┘
        │
        ▼
  结果图片 + 检测对象列表
```

---

## 批量测试脚本 (`test_dinox.py`)

### 功能

- 从 `benchmark/example/discription.json` 读取场景和物体描述
- 选择指定图像（如 3, 5, 8, 10, 11），逐个物体进行检测
- 分别使用 `short_en`（简短描述）和 `detailed_en`（详细描述）作为 prompt
- 保存检测结果图片和完整的 API 日志（自动过滤 Token 等敏感信息）
- 输出检测成功率统计

### 运行方式

```bash
# 多图模式（默认）
python test_dinox.py

# 单物体测试模式
python test_dinox.py single
```

### 输出结构

```
rofa/dinox_result/
├── 3.jpg                          # 原图
├── 3_obj1_short.jpg               # 第1个物体 short_en 检测结果
├── 3_obj1_detailed.jpg            # 第1个物体 detailed_en 检测结果
├── ...
├── api_requests_log.txt           # API 请求/响应日志
└── api_requests_log_summary.json  # 检测结果统计
```

### 统计输出示例

```json
{
    "statistics": {
        "total_detections": 50,
        "successful_detections": 42,
        "empty_detections": 8,
        "success_rate": "84.00%"
    }
}
```

---

## API 接口地址

| 接口 | URL | 方法 |
|------|-----|------|
| 创建任务 | `https://api.deepdataspace.com/v2/task/dino_xseek/detection` | POST |
| 查询状态 | `https://api.deepdataspace.com/v2/task_status/{task_uuid}` | GET |

---

## 环境变量

| 变量名 | 必填 | 说明 |
|--------|------|------|
| `DINO_API_TOKEN` | 是 | deepdataspace DINO-X 开放平台 API Token |
