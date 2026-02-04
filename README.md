# RobotMustFindAnything

## 环境要求
- **操作系统**：Ubuntu 22.04
- **ROS 版本**：ROS 2 Humble

## 安装步骤

### 1. 安装 ROS 2 Humble
参考教程：[ROS 2 Humble 安装教程](https://comate.baidu.com/zh/page/w4rrweyik8h)

安装完成后，运行以下命令验证安装是否成功：
```bash
echo $ROS_DISTRO
```
如果输出 `humble`，则安装成功。

#### 注意事项
- 后续编译需要安装 `empy` 库（版本需小于 4）。
  - 如果使用 Conda 的 `base` 环境：
    ```bash
    pip install empy==3.3.4
    ```
  - 如果使用系统级 Python 或 ROS2 自带 Python 环境，需要先切换到相应python环境，再：
    ```bash
    pip install empy==3.3.4
    ```
    同时需要指定 ROS 编译使用的 Python 解释器：
    ```bash
    colcon build --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3
    ```

### 2. 克隆仓库
运行以下命令克隆仓库并进入目录：
```bash
git clone https://github.com/zjy109/RobotMustFindAnything.git
cd RobotMustFindAnything
```

### 3. 配置雷达网络
将雷达的网口接入电脑，并调整雷达有线连接的静态 IP 地址：
- **IP 地址**：`192.168.1.50`
- **子网掩码**：`255.255.255.0`
- **网关**：`192.168.1.255`

### 4. 安装 Livox SDK
运行以下命令安装 Livox SDK：
```bash
cd ./rofa/ros_ws/src/Livox-SDK2
mkdir build
cd build
cmake .. && make -j
sudo make install
```

### 5. 安装 livox_ros_driver2
运行以下命令安装 livox_ros_driver2：
```bash
cd ../livox_ros_driver2
./build.sh humble
```

### 6. 修改 MID360 配置文件
编辑配置文件：
```bash
gedit ./config/MID360_config.json
```
修改以下内容：
- **`"host_net_info"`** 下的 4 个 IP 均改为 `"192.168.1.50"`
- **`"lidar_configs"`** 下的 `"ip"` 改为 `"192.168.1.172"`（注意：IP 的最后两位需对应雷达的 SN 码）

### 7. 测试雷达数据获取
运行以下命令测试雷达数据获取：
```bash
cd ~/RobotMustFindAnything/rofa/ros_ws
colcon build
source install/setup.sh
ros2 launch livox_ros_driver2 rviz_MID360_launch.py
```

### 8. 安装 FASTLIO2 所需依赖
运行以下命令安装依赖：
```bash
rosdep install --from-paths src --ignore-src -y
```
如果安装失败，可能需要先运行以下命令：
```bash
sudo rosdep init
rosdep update
```

### 9. 编译 FASTLIO2
运行以下命令编译 FASTLIO2：
```bash
colcon build --symlink-install
```

### 10. 运行 FASTLIO2
1. 打开一个终端，运行 livox_ros_driver2：
   ```bash
   source install/setup.sh
   ros2 launch livox_ros_driver2 msg_MID360_launch.py
   ```
2. 打开另一个终端，运行 FASTLIO2：
   ```bash
   source install/setup.sh
   ros2 launch fast_lio mapping.launch.py config_file:=mid360.yaml
   ```

## 参考教程
- [CSDN 教程 1](https://blog.csdn.net/2401_83327355/article/details/157468981?spm=1001.2014.3001.5506)
- [CSDN 教程 2](https://blog.csdn.net/2401_83327355/article/details/157511186?spm=1001.2014.3001.5506)
- [51CTO 教程](https://blog.51cto.com/u_12968/14213494)

