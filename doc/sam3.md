# SAM3 语义分割模块

## 概述

本模块基于 [SAM3](https://github.com/facebookresearch/sam3)（Segment Anything Model 3）实现了一个**开放词汇语义分割器**，支持通过自然语言文本描述定位并分割图像中的目标物体。

### 特性

- **开放词汇检测**：通过任意文本 prompt 检测对应目标，无需预定义类别
- **多图片批处理**：一次调用传入多张图片，逐张编码并推理
- **多 prompt 复用特征**：同一张图片只做一次 backbone 编码，多个 prompt 共享特征，提升效率
- **多种输入格式**：支持文件路径（`str`）、`numpy.ndarray`、`PIL.Image` 三种输入
- **统一的结果接口**：`SegmentResponse` 提供结构化的结果访问与可视化

---

## 目录结构

```
rofa/sam3/
├── sam3_segmentor.py    # SAM3 分割器主模块（本文件描述的内容）
├── sam3_demo.py         # 简单 demo 脚本
└── sam3_lib/            # SAM3 模型库（第三方子模块）
    └── sam3/
        └── sam3/
            ├── model_builder.py           # 模型构建入口
            └── model/
                └── sam3_image_processor.py # 图像处理器
```

---

## 快速开始

### 安装依赖

确保已安装 SAM3 模型库（`sam3_lib/sam3` 目录下的包）：

```bash
cd rofa/sam3/sam3_lib/sam3
pip install -e .
```

### 基本用法

```python
from sam3_segmentor import SAM3Segmentor, SegmentQuery

# 1. 初始化分割器
segmentor = SAM3Segmentor(options={
    "confidence_threshold": 0.5,  # 置信度阈值（默认 0.5）
    "device": "cuda",             # 推理设备（默认 cuda）
    "resolution": 1008,           # 输入分辨率（默认 1008）
})

# 2. 构造查询
query = SegmentQuery(
    image_paths=["image1.png", "image2.png"],
    prompts=["cup", "keyboard"],
)

# 3. 推理
response = segmentor.predict(query)

# 4. 获取结果
results = response.dict_results()
for path, labels in results.items():
    for label, data in labels.items():
        print(f"{path} - {label}: {len(data['boxes'])} detections")
        # data["scores"]: (N,)   置信度
        # data["boxes"]:  (N, 4) 边界框 [x1, y1, x2, y2]
        # data["masks"]:  (N, H, W) 二值掩码

# 5. 可视化并保存
response.visualize(output_dir="./vis_results")
```

### 单图推理快捷方法

```python
result = segmentor.predict_single("image.png", "cup")
if result is not None:
    print(f"Found {result.count} instances, scores: {result.scores}")
```

### 命令行运行

```bash
python sam3_segmentor.py \
    --images image1.png image2.png \
    --prompts cup keyboard \
    --output_dir ./vis_results \
    --confidence 0.5
```

---

## API 参考

### `SegmentQuery`

查询对象，封装输入图片和文本提示词。

```python
SegmentQuery(image_paths, prompts)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `image_paths` | `str \| list` | 图片路径列表，或单个路径/ndarray/PIL.Image |
| `prompts` | `str \| list` | 文本提示词列表，或单个字符串 |

---

### `SegmentResult`

单个标签的检测结果。

| 属性 | 类型 | 说明 |
|------|------|------|
| `scores` | `np.ndarray (N,)` | 每个检测的置信度分数 |
| `boxes` | `np.ndarray (N, 4)` | 边界框 `[x1, y1, x2, y2]`（像素坐标） |
| `masks` | `np.ndarray (N, H, W)` | 二值分割掩码（0/1） |
| `count` | `int` | 检测到的实例数量 |

---

### `SegmentResponse`

完整的分割响应，包含所有图片、所有标签的结果。

| 方法 | 说明 |
|------|------|
| `dict_results()` | 将结果转为嵌套字典 `{image_key: {label: {scores, boxes, masks}}}` |
| `visualize(output_dir=None)` | 可视化结果。传入目录则保存图片，否则调用系统查看器显示 |

---

### `SAM3Segmentor`

分割器主类。

#### 构造函数

```python
SAM3Segmentor(options=None)
```

| options 参数 | 类型 | 默认值 | 说明 |
|-------------|------|--------|------|
| `confidence_threshold` | `float` | `0.5` | 检测置信度阈值 |
| `device` | `str` | `"cuda"` | 推理设备 |
| `resolution` | `int` | `1008` | 模型输入分辨率 |

#### 方法

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `predict(query)` | `SegmentQuery` | `SegmentResponse` | 批量推理，支持多图多 prompt |
| `predict_single(image, prompt)` | 图片, 文本 | `SegmentResult \| None` | 单图单 prompt 便捷推理 |

---

## 推理流程

```
输入图片列表 + prompt 列表
        │
        ▼
┌──────────────────────────┐
│  遍历每张图片              │
│  ┌────────────────────┐  │
│  │ set_image()        │  │  ← 一次 backbone 编码
│  │ (图像特征缓存)      │  │
│  └────────┬───────────┘  │
│           │               │
│  ┌────────▼───────────┐  │
│  │ 遍历每个 prompt     │  │
│  │ set_text_prompt()  │  │  ← 复用图像特征，只做文本编码+解码
│  │ → masks, boxes,    │  │
│  │   scores           │  │
│  └────────────────────┘  │
└──────────────────────────┘
        │
        ▼
  SegmentResponse (统一结果)
```
