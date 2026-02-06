import pyrealsense2 as rs
import time

class RealsenseBase:
    def __init__(self, serial_number=None):
        """
        初始化 RealsenseBase 类
        :param serial_number: (Optional) 指定相机的序列号，如果为 None 则自动选择第一个可用设备
        """
        self.serial_number = serial_number
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 如果指定了序列号，则配置只启用该设备
        if self.serial_number:
            self.config.enable_device(self.serial_number)

    @staticmethod
    def list_devices():
        """
        检测并列出所有连接的 RealSense 设备
        :return: 包含设备名称和序列号的列表
        """
        ctx = rs.context()
        devices_list = []
        
        try:
            for dev in ctx.query_devices():
                try:
                    name = dev.get_info(rs.camera_info.name)
                    serial = dev.get_info(rs.camera_info.serial_number)
                    usb_type = dev.get_info(rs.camera_info.usb_type_descriptor)
                    
                    device_info = {
                        "name": name,
                        "serial_number": serial,
                        "usb_type": usb_type
                    }
                    devices_list.append(device_info)
                    print(f"Found Device: {name} | Serial: {serial} | USB: {usb_type}")
                except RuntimeError as e:
                    print(f"Error reading device info: {e}")
                    
        except Exception as e:
            print(f"Error querying devices: {e}")
            
        return devices_list

    def get_serial_number(self):
        """
        获取当前实例绑定的序列号。如果初始化时未指定，尝试获取第一个连接设备的序列号。
        """
        if self.serial_number:
            return self.serial_number
        
        # 如果未指定，查找第一个设备
        devices = self.list_devices()
        if devices:
            self.serial_number = devices[0]["serial_number"]
            self.config.enable_device(self.serial_number)
            return self.serial_number
        else:
            print("No device found.")
            return None

if __name__ == "__main__":
    print("Searching for RealSense devices...")
    devices = RealsenseBase.list_devices()
    
    if devices:
        print(f"\nTotal devices found: {len(devices)}")
    else:
        print("\nNo devices found.")
