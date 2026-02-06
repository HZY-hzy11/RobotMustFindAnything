# ROS2 工作空间 — Livox 雷达 & FAST-LIO2

本文档描述 `rofa/ros_ws/` 工作空间的功能模块和使用方法。环境安装请参考项目根目录 [README.md](../README.md)。

## 模块组成

| 包名 | 路径 | 说明 |
|------|------|------|
| Livox-SDK2 | `src/Livox-SDK2/` | Livox 雷达底层 SDK |
| livox_ros_driver2 | `src/livox_ros_driver2/` | Livox 雷达 ROS2 驱动，发布点云话题 |
| FAST_LIO_ROS2 | `src/FAST_LIO_ROS2/` | FAST-LIO2 算法包，用于实时激光惯性里程计与建图 |

## 使用方法

> 以下命令均在 `rofa/ros_ws/` 目录下执行，且假设已完成编译（`colcon build`）。

### 1. 测试雷达数据获取

单独启动 Livox 驱动并通过 RViz 查看点云：

```bash
source install/setup.sh
ros2 launch livox_ros_driver2 rviz_MID360_launch.py
```

### 2. 运行 FAST-LIO2 建图

需要开启**两个终端**：

**终端 1** — 启动 Livox 驱动：

```bash
source install/setup.sh
ros2 launch livox_ros_driver2 msg_MID360_launch.py
```

**终端 2** — 启动 FAST-LIO2：

```bash
source install/setup.sh
ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml
```

### 3. 配置文件说明

#### Livox 驱动配置

配置文件位于 `src/livox_ros_driver2/config/MID360_config.json`：

- `host_net_info`：主机 IP 配置（4 个 IP 均填写主机静态 IP）
- `lidar_configs.ip`：雷达 IP（末两位对应雷达 SN 码）

#### FAST-LIO2 配置

配置文件位于 `src/FAST_LIO_ROS2/config/`，根据雷达型号选择对应的 YAML：

| 文件 | 适用雷达 |
|------|----------|
| `mid360.yaml` | Livox MID-360 |
| `avia.yaml` | Livox Avia |
| `horizon.yaml` | Livox Horizon |
| `velodyne.yaml` | Velodyne |
| `ouster64.yaml` | Ouster OS1-64 |

### 4. 建图结果

FAST-LIO2 的点云地图（PCD 格式）保存在 `src/FAST_LIO_ROS2/PCD/` 目录下。

