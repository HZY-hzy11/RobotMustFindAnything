# RealSense 相机模块

本文档描述 `rofa/realsense/` 模块的功能和使用方法。环境安装请参考项目根目录 [README.md](../README.md)。

## 支持型号

- **Intel RealSense D435** — RGB-D 深度相机

## 代码结构

| 文件 | 说明 |
|------|------|
| `realsense_base.py` | 基类，负责设备检测与序列号管理 |
| `d435.py` | D435 实现，RGB/Depth 流管理与深度对齐 |

## 使用方法

### 1. 自动连接（默认设备）

未指定序列号时，自动连接第一个可用的 RealSense 设备：

```python
from rofa.realsense.d435 import D435

# Initialize (auto-selects first device if serial is not provided)
cam = D435()
cam.start()

try:
    # 获取帧
    rgb, depth = cam.get_frame(aligned=True)
    
    if rgb is not None and depth is not None:
        pass  # 在此处理图像
finally:
    # 停止相机，释放资源
    cam.stop()
```

### 2. 指定序列号连接

连接多台相机时，通过序列号指定目标设备：

```python
# 替换为实际的相机序列号
cam = D435(serial_number="817412071234")
cam.start()
```

### 3. 获取帧 & 深度对齐

`get_frame` 方法返回 numpy 数组格式的彩色图和深度图。

- **函数签名**：`get_frame(aligned=True)`
- **返回值**：`(color_image, depth_image)`，失败时返回 `(None, None)`

**参数说明：**
- `aligned=True`（默认）：深度图对齐到彩色相机视角，实现像素级对应（RGB-D）。
- `aligned=False`：返回各传感器原始视角的图像（深度与彩色视角不同）。

```python
# 获取对齐后的 RGB-D 帧
color_img, depth_img = cam.get_frame(aligned=True)
# color_img: (480, 640, 3) BGR 格式（适用于 OpenCV）
# depth_img: (480, 640) 深度值
```

### 4. 列出已连接设备

通过基类的静态方法查看所有已连接设备及其序列号：

```python
from rofa.realsense.realsense_base import RealsenseBase

# 静态方法，无需实例化
devices = RealsenseBase.list_devices()

for dev in devices:
    print(f"Name: {dev['name']}, Serial: {dev['serial_number']}")
```

## 常见问题

- **未找到设备**：确认使用 USB 3.0 线缆并连接到 USB 3.0 端口。
- **权限不足（Linux）**：可能需要安装 RealSense 的 `udev` 规则，或以 `sudo` 运行（不推荐用于生产环境）。
- **导入错误**：确认已在 `rofa` 环境中安装依赖（`pip install pyrealsense2 numpy`）。
