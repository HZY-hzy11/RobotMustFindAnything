import pyrealsense2 as rs
import numpy as np
try:
    from .realsense_base import RealsenseBase
except ImportError:
    # 允许直接运行此文件进行测试 (Fallback provided for running as script)
    from realsense_base import RealsenseBase

class D435(RealsenseBase):
    def __init__(self, serial_number=None, width=640, height=480, fps=30):
        """
        初始化 D435 相机
        :param serial_number: 指定序列号，若为 None 则自动寻找第一个可用设备
        :param width: 图像宽度
        :param height: 图像高度
        :param fps: 帧率
        """
        super().__init__(serial_number)
        
        self.width = width
        self.height = height
        self.fps = fps
        self.is_running = False
        
        # 1. 如果初始化时没有提供序列号，尝试使用基类方法获取默认设备的序列号
        if not self.serial_number:
            self._use_default_device()

        # 2. 配置 D435 视频流
        self._config_streams()
        
        # 3. 创建对齐对象 (深度 -> 彩色)
        self.align = rs.align(rs.stream.color)

    def _use_default_device(self):
        """如果没有传入序列号，则查找并使用第一个设备"""
        found_serial = self.get_serial_number()
        if found_serial:
            print(f"[D435] Serial not provided. Using default device: {found_serial}")
        else:
            raise RuntimeError("No RealSense device found. Please connect a D435 camera.")

    def _config_streams(self):
        """配置彩色和深度流"""
        # 启用深度流
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        # 启用彩色流
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

    def start(self):
        """启动相机管线"""
        try:
            self.pipeline_profile = self.pipeline.start(self.config)
            self.is_running = True
            print(f"[D435] Camera {self.serial_number} started.")
        except Exception as e:
            print(f"[D435] Failed to start camera: {e}")
            self.is_running = False

    def stop(self):
        """停止相机管线"""
        if self.is_running:
            self.pipeline.stop()
            self.is_running = False
            print(f"[D435] Camera {self.serial_number} stopped.")

    def get_frame(self, aligned=True):
        """
        获取一帧图像数据
        :param aligned: True=返回深度对齐到彩色后的图像; False=返回原始图像
        :return: (color_image, depth_image)，格式为 numpy array. 失败返回 (None, None)
        """
        if not self.is_running:
            print("[D435] Camera is not running. Call start() first.")
            return None, None

        try:
            # 等待一帧数据，超时时间设为 5000ms
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)

            # 是否进行深度对齐
            if aligned:
                frames = self.align.process(frames)

            # 获取帧
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                return None, None

            # 转换为 numpy 数组
            # color_image: (480, 640, 3)
            # depth_image: (480, 640) - 单位通常是毫米，或者是根据 scale 确定的
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            return color_image, depth_image

        except Exception as e:
            print(f"[D435] Error getting frames: {e}")
            return None, None

if __name__ == "__main__":
    # 简单的测试代码
    import cv2
    
    try:
        # 初始化 D435，不传序列号将自动查找
        cam = D435("017322071981")
        cam.start()

        print("Press 'q' to quit...")
        while True:
            rgb, depth = cam.get_frame(aligned=True)
            
            if rgb is not None and depth is not None:
                # 为了可视化，将深度图归一化并转为伪彩色
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
                
                # 水平拼接显示
                images = np.hstack((rgb, depth_colormap))
                
                cv2.imshow('D435 RealSense', images)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'cam' in locals():
            cam.stop()
        cv2.destroyAllWindows()
