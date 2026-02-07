# ArUco 标记检测模块

本文档描述 `rofa/aruco/` 模块的功能和使用方法。环境安装请参考项目根目录 [README.md](../README.md)。

## 当前配置

| 参数 | 值 |
|------|------|
| ArUco 字典 | DICT_4X4_50 |
| Marker 边长 | 50mm (0.05m) |
| 目标 ID | 0 |

## 代码结构

| 文件 | 说明 |
|------|------|
| `aruco_detector.py` | ArUco 检测器，支持 6D 位姿估计与坐标系转换 |

## 坐标系定义

模块同时输出两种坐标系下的 Marker 6D 位姿：

| 坐标系 | X | Y | Z | 适用场景 |
|--------|---|---|---|----------|
| 光学坐标系 (Optical) | 右 | 下 | 前 | 视觉算法、OpenCV 默认 |
| ROS 坐标系 | 前 | 左 | 上 | ROS 导航、机器人控制 |

转换矩阵 $T_{\text{ROS} \leftarrow \text{optical}}$：

$$
T = \begin{bmatrix} 0 & 0 & 1 & 0 \\ -1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
$$

## 使用方法

### 1. 配合 D435 相机使用

```python
from rofa.realsense.d435 import D435
from rofa.aruco.aruco_detector import ArucoDetector
from cv2 import aruco

# 初始化相机
cam = D435()
cam.start()

# 初始化检测器 (Dict4x4, 50mm, ID=0)
detector = ArucoDetector(marker_size=0.05, target_id=0, dictionary=aruco.DICT_4X4_50)

# 从 D435 自动获取内参
detector.set_camera_intrinsics_from_d435(cam)

try:
    rgb, depth = cam.get_frame(aligned=True)

    result = detector.estimate_pose(rgb)
    if result is not None:
        # 光学坐标系下 6D 位姿
        print("Optical position:", result["optical"]["position"])       # [x, y, z] 米
        print("Optical quaternion:", result["optical"]["quaternion"])    # [x, y, z, w]

        # ROS 坐标系下 6D 位姿
        print("ROS position:", result["ros"]["position"])               # [x, y, z] 米
        print("ROS quaternion:", result["ros"]["quaternion"])            # [x, y, z, w]
finally:
    cam.stop()
```

### 2. 手动设置内参

如果不使用 D435 模块，可以手动提供相机内参：

```python
detector = ArucoDetector(marker_size=0.05, target_id=0)
detector.set_camera_intrinsics(fx=615.0, fy=615.0, cx=320.0, cy=240.0)
```

### 3. 可视化检测结果

```python
import cv2

vis = detector.draw_detections(rgb, result)
cv2.imshow("ArUco", vis)
cv2.waitKey(0)
```

`draw_detections` 会在图像上绘制 Marker 边框、坐标轴以及光学/ROS 坐标系下的位置信息。

## API 参考

### `ArucoDetector(marker_size, target_id, dictionary)`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `marker_size` | float | 0.05 | Marker 边长 (米) |
| `target_id` | int | 0 | 目标 Marker ID |
| `dictionary` | int | DICT_4X4_50 | ArUco 字典 |

### `estimate_pose(color_image) -> dict | None`

返回值结构：

```python
{
    "optical": {
        "rvec":            np.array,  # 旋转向量 (Rodrigues)
        "tvec":            np.array,  # 平移向量 [x, y, z] 米
        "rotation_matrix": np.array,  # 3x3 旋转矩阵
        "position":        np.array,  # 位置 [x, y, z] 米
        "quaternion":      np.array,  # 四元数 [x, y, z, w]
    },
    "ros": {
        "rotation_matrix": np.array,
        "position":        np.array,
        "quaternion":      np.array,  # [x, y, z, w]
    },
    "corners": np.array,  # 目标 Marker 的图像角点
}
```
