import argparse
import time
from typing import Optional

try:
    from .zdt_ttl import ZDTTTL
except ImportError:
    from zdt_ttl import ZDTTTL


class Motor42:
    """42 闭环步进电机高层封装。"""

    def __init__(
        self,
        addr: int = 1,
        port: Optional[str] = None,
        baudrate: int = 115200,
        zero_delay: float = 0.5,
    ):
        self.addr = addr
        self.zero_delay = zero_delay
        self.driver = ZDTTTL(port=port, baudrate=baudrate)

    def power_on_set_zero(self):
        """上电后使能电机，并把当前机械位置记为 0 位。"""
        self.driver.enable(self.addr, True)
        time.sleep(0.2)
        self.driver.reset_cur_pos_to_zero(self.addr)
        time.sleep(self.zero_delay)

    def release(self):
        """失能电机，让电机泄力，便于手动转动。"""
        self.driver.enable(self.addr, False)
        time.sleep(0.1)

    def hold(self):
        """重新使能电机。"""
        self.driver.enable(self.addr, True)
        time.sleep(0.1)

    def read_angle(self) -> float:
        """读取当前角度，单位为度。"""
        return self.driver.read_position(self.addr)

    def print_angle(self, prefix: str = "当前角度"):
        angle = self.read_angle()
        print(f"{prefix}: {angle:.2f}°")

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def main():
    parser = argparse.ArgumentParser(description="42 电机上电清零、泄力并读取角度")
    parser.add_argument("--port", type=str, default=None, help="串口设备路径，默认自动检测 CH340")
    parser.add_argument("--addr", type=int, default=1, help="电机地址")
    parser.add_argument("--baudrate", type=int, default=115200, help="串口波特率")
    parser.add_argument("--interval", type=float, default=0.5, help="角度打印周期，单位秒")
    args = parser.parse_args()

    with Motor42(addr=args.addr, port=args.port, baudrate=args.baudrate) as motor:
        try:
            print("1. 电机上电并将当前位置设为 0 位")
            motor.power_on_set_zero()
            motor.print_angle("零位角度")

            print("2. 电机泄力，可以手动转动电机")
            motor.release()

            print("3. 开始读取角度，按 Ctrl+C 退出")
            while True:
                motor.print_angle()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n停止读取角度")


if __name__ == "__main__":
    main()
