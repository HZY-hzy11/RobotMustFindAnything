"""
ZDT TTL 步进电机串口通讯控制类
适用于 x86_64 Ubuntu 22.04 Linux，通过 USB 转 TTL 串口控制步进电机
基于 Emm_V5 协议，使用 pyserial 库
安装依赖: pip install pyserial
"""

import time
import struct
import serial
import serial.tools.list_ports
from typing import Optional, Tuple, List


class ZDTTTL:
    """ZDT 步进电机 TTL 串口通讯控制类"""

    # 系统参数读取功能码对照表
    FUNC_CODES = {
        'S_VER':   0x1F,   # 读取固件版本和对应的硬件版本
        'S_RL':    0x20,   # 读取相电阻和相电感
        'S_PID':   0x21,   # 读取PID参数
        'S_VBUS':  0x24,   # 读取总线电压
        'S_CPHA':  0x27,   # 读取相电流
        'S_ENCL':  0x31,   # 读取经过线性化校准后的编码器值
        'S_TPOS':  0x33,   # 读取电机目标位置角度
        'S_VEL':   0x35,   # 读取电机实时转速
        'S_CPOS':  0x36,   # 读取电机实时位置角度
        'S_PERR':  0x37,   # 读取电机位置误差角度
        'S_FLAG':  0x3A,   # 读取使能/到位/堵转状态标志位
        'S_ORG':   0x3B,   # 读取正在回零/回零失败状态标志位
        'S_Conf':  0x42,   # 读取驱动参数 (功能码后需加辅助码0x6C)
        'S_State': 0x43,   # 读取系统状态参数 (功能码后需加辅助码0x7A)
    }

    # 方向常量
    CW = 0    # 顺时针
    CCW = 1   # 逆时针

    # CH340 USB转TTL芯片的 VID:PID
    CH340_VID = 0x1A86
    CH340_PID = 0x7523

    @staticmethod
    def find_ch340_ports() -> List[str]:
        """
        自动检测所有 CH340 串口设备 (VID=1A86, PID=7523)

        Returns:
            匹配的串口设备路径列表, 如 ['/dev/ttyUSB0']
        """
        ports = []
        for port_info in serial.tools.list_ports.comports():
            if port_info.vid == ZDTTTL.CH340_VID and port_info.pid == ZDTTTL.CH340_PID:
                ports.append(port_info.device)
        return ports

    @staticmethod
    def auto_detect_port() -> str:
        """
        自动检测 CH340 串口设备并返回第一个匹配的端口路径

        Returns:
            串口设备路径

        Raises:
            RuntimeError: 未找到 CH340 设备
        """
        ports = ZDTTTL.find_ch340_ports()
        if not ports:
            # 列出所有可用串口供排查
            all_ports = list(serial.tools.list_ports.comports())
            avail = ', '.join(f'{p.device} (VID={p.vid:#06x}, PID={p.pid:#06x}, {p.description})'
                              for p in all_ports if p.vid is not None) or '无'
            raise RuntimeError(
                f'未找到 CH340 设备 (VID={ZDTTTL.CH340_VID:#06x}, PID={ZDTTTL.CH340_PID:#06x})。'
                f'当前可用USB串口: {avail}'
            )
        if len(ports) > 1:
            print(f'[ZDTTTL] 检测到多个CH340设备: {ports}, 使用第一个: {ports[0]}')
        return ports[0]

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200, timeout: float = 0.2):
        """
        初始化串口连接

        Args:
            port: 串口设备路径, 如 '/dev/ttyUSB0'。为 None 时自动检测 CH340 设备
            baudrate: 波特率, 默认 115200
            timeout: 读取超时时间(秒), 默认 0.2s
        """
        if port is None:
            port = self.auto_detect_port()
            print(f'[ZDTTTL] 自动检测到 CH340 串口: {port}')
        self.port = port
        self.baudrate = baudrate
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        )
        print(f'[ZDTTTL] 串口已打开: {self.ser.name} (波特率: {baudrate})')

    def close(self):
        """关闭串口"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print('[ZDTTTL] 串口已关闭')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ======================== 底层通信 ========================

    def _send(self, cmd: bytearray):
        """发送命令字节"""
        self.ser.write(cmd)

    def _receive(self, recv_timeout: float = 0.1) -> Tuple[str, int]:
        """
        接收串口返回数据

        Args:
            recv_timeout: 数据间隔超时(秒), 超过此时间无新数据则认为接收完成

        Returns:
            (hex_data, count): 十六进制字符串和数据字节长度
        """
        rx_data = bytearray()
        last_recv_time = time.time()

        while True:
            n = self.ser.in_waiting
            if n > 0:
                rx_data += self.ser.read(n)
                last_recv_time = time.time()
            else:
                if time.time() - last_recv_time > recv_timeout:
                    break
            time.sleep(0.001)

        if len(rx_data) == 0:
            return '', 0

        hex_data = ' '.join([f'{b:02x}' for b in rx_data])
        return hex_data, len(rx_data)

    def _flush_input(self):
        """清空接收缓冲区"""
        self.ser.reset_input_buffer()

    # ======================== 系统参数读取 ========================

    def read_sys_params(self, addr: int, param: str):
        """
        读取驱动板参数

        Args:
            addr: 电机地址 (1-247)
            param: 参数名称, 如 'S_VER', 'S_CPOS', 'S_FLAG' 等
        """
        cmd = bytearray()
        cmd.append(addr)
        if param in self.FUNC_CODES:
            cmd.append(self.FUNC_CODES[param])
        else:
            raise ValueError(f'未知参数: {param}')
        cmd.append(0x6B)
        self._send(cmd)

    # ======================== 电机控制命令 ========================

    def reset_cur_pos_to_zero(self, addr: int):
        """
        将当前位置清零

        Args:
            addr: 电机地址
        """
        cmd = bytearray([addr, 0x0A, 0x6D, 0x6B])
        self._send(cmd)

    def reset_clog_pro(self, addr: int):
        """
        解除堵转保护

        Args:
            addr: 电机地址
        """
        cmd = bytearray([addr, 0x0E, 0x52, 0x6B])
        self._send(cmd)

    def modify_ctrl_mode(self, addr: int, save: bool, ctrl_mode: int):
        """
        修改控制模式

        Args:
            addr: 电机地址
            save: 是否存储设置
            ctrl_mode: 控制模式
        """
        cmd = bytearray([
            addr,
            0x46,
            0x69,
            0x01 if save else 0x00,
            ctrl_mode,
            0x6B,
        ])
        self._send(cmd)

    def enable(self, addr: int, state: bool, sync: bool = False):
        """
        电机使能控制

        Args:
            addr: 电机地址
            state: True=使能, False=失能
            sync: 多机同步运动标志
        """
        cmd = bytearray([
            addr,
            0xF3,
            0xAB,
            0x01 if state else 0x00,
            0x01 if sync else 0x00,
            0x6B,
        ])
        self._send(cmd)

    def vel_control(self, addr: int, direction: int, vel: int, acc: int, sync: bool = False):
        """
        速度模式控制

        Args:
            addr: 电机地址
            direction: 方向, 0=CW(顺时针), 1=CCW(逆时针)
            vel: 速度(RPM)
            acc: 加速度, 0为直接启动
            sync: 多机同步运动标志
        """
        cmd = bytearray([
            addr,
            0xF6,
            direction,
            (vel >> 8) & 0xFF,
            vel & 0xFF,
            acc,
            0x01 if sync else 0x00,
            0x6B,
        ])
        self._send(cmd)

    def pos_control(self, addr: int, direction: int, vel: int, acc: int,
                    pulses: int, relative: bool = False, sync: bool = False):
        """
        位置模式控制

        Args:
            addr: 电机地址
            direction: 方向, 0=CW(顺时针), 1=CCW(逆时针)
            vel: 速度(RPM)
            acc: 加速度, 0为直接启动
            pulses: 脉冲数
            relative: True=相对运动, False=绝对运动
            sync: 多机同步运动标志
        """
        cmd = bytearray([
            addr,
            0xFD,
            direction,
            (vel >> 8) & 0xFF,
            vel & 0xFF,
            acc,
            (pulses >> 24) & 0xFF,
            (pulses >> 16) & 0xFF,
            (pulses >> 8) & 0xFF,
            pulses & 0xFF,
            0x01 if relative else 0x00,
            0x01 if sync else 0x00,
            0x6B,
        ])
        self._send(cmd)

    def stop(self, addr: int, sync: bool = False):
        """
        立即停止电机

        Args:
            addr: 电机地址
            sync: 多机同步运动标志
        """
        cmd = bytearray([
            addr,
            0xFE,
            0x98,
            0x01 if sync else 0x00,
            0x6B,
        ])
        self._send(cmd)

    def sync_motion(self, addr: int):
        """
        执行多机同步运动命令

        Args:
            addr: 电机地址 (通常用广播地址 0x00)
        """
        cmd = bytearray([addr, 0xFF, 0x66, 0x6B])
        self._send(cmd)

    # ======================== 回零相关 ========================

    def origin_set_zero(self, addr: int, save: bool = True):
        """
        设置单圈回零零点位置

        Args:
            addr: 电机地址
            save: 是否存储设置
        """
        cmd = bytearray([
            addr,
            0x93,
            0x88,
            0x01 if save else 0x00,
            0x6B,
        ])
        self._send(cmd)

    def origin_modify_params(self, addr: int, save: bool, o_mode: int, o_dir: int,
                             o_vel: int, o_timeout: int, sl_vel: int, sl_ma: int,
                             sl_ms: int, auto_home: bool):
        """
        修改回零参数

        Args:
            addr: 电机地址
            save: 是否存储设置
            o_mode: 回零模式
            o_dir: 回零方向
            o_vel: 回零速度(RPM)
            o_timeout: 回零超时时间(ms)
            sl_vel: 无限位碰撞检测转速
            sl_ma: 无限位碰撞检测电流
            sl_ms: 无限位碰撞检测时间
            auto_home: 上电自动触发回零
        """
        cmd = bytearray([
            addr,
            0x4C,
            0xAE,
            0x01 if save else 0x00,
            o_mode,
            o_dir,
            (o_vel >> 8) & 0xFF,
            o_vel & 0xFF,
            (o_timeout >> 24) & 0xFF,
            (o_timeout >> 16) & 0xFF,
            (o_timeout >> 8) & 0xFF,
            o_timeout & 0xFF,
            (sl_vel >> 8) & 0xFF,
            sl_vel & 0xFF,
            (sl_ma >> 8) & 0xFF,
            sl_ma & 0xFF,
            (sl_ms >> 8) & 0xFF,
            sl_ms & 0xFF,
            0x01 if auto_home else 0x00,
            0x6B,
        ])
        self._send(cmd)

    def origin_trigger(self, addr: int, o_mode: int = 0, sync: bool = False):
        """
        触发回零

        Args:
            addr: 电机地址
            o_mode: 回零模式
            sync: 多机同步运动标志
        """
        cmd = bytearray([
            addr,
            0x9A,
            o_mode,
            0x01 if sync else 0x00,
            0x6B,
        ])
        self._send(cmd)

    def origin_interrupt(self, addr: int):
        """
        强制中断退出回零

        Args:
            addr: 电机地址
        """
        cmd = bytearray([addr, 0x9C, 0x48, 0x6B])
        self._send(cmd)

    # ======================== 高级功能 ========================

    def read_position(self, addr: int) -> float:
        """
        读取电机实时位置角度

        Args:
            addr: 电机地址

        Returns:
            当前位置角度(度), 读取失败返回 0.0
        """
        self._flush_input()
        self.read_sys_params(addr, 'S_CPOS')
        data, count = self._receive()

        if count == 0 or not data:
            print(f'[ZDTTTL] 电机{addr}: 无数据返回')
            return 0.0

        data_hex = data.split()
        if count >= 7 and int(data_hex[0], 16) == addr and int(data_hex[1], 16) == 0x36:
            pos_raw = struct.unpack('>I', bytes.fromhex(''.join(data_hex[3:7])))[0]
            angle = float(pos_raw) * 360.0 / 65536.0
            if int(data_hex[2], 16):  # 符号位
                angle = -angle
            return angle
        else:
            print(f'[ZDTTTL] 电机{addr}: 数据格式错误 -> {data}')
            return 0.0

    def read_velocity(self, addr: int) -> float:
        """
        读取电机实时转速

        Args:
            addr: 电机地址

        Returns:
            当前转速(RPM), 读取失败返回 0.0
        """
        self._flush_input()
        self.read_sys_params(addr, 'S_VEL')
        data, count = self._receive()

        if count == 0 or not data:
            print(f'[ZDTTTL] 电机{addr}: 无数据返回')
            return 0.0

        data_hex = data.split()
        if count >= 4 and int(data_hex[0], 16) == addr and int(data_hex[1], 16) == 0x35:
            vel_raw = struct.unpack('>H', bytes.fromhex(''.join(data_hex[3:5])))[0]
            vel = float(vel_raw) * 0.1  # 单位换算
            if int(data_hex[2], 16):  # 符号/方向位
                vel = -vel
            return vel
        else:
            print(f'[ZDTTTL] 电机{addr}: 速度数据格式错误 -> {data}')
            return 0.0

    def read_bus_voltage(self, addr: int) -> float:
        """
        读取总线电压

        Args:
            addr: 电机地址

        Returns:
            总线电压(V), 读取失败返回 0.0
        """
        self._flush_input()
        self.read_sys_params(addr, 'S_VBUS')
        data, count = self._receive()

        if count == 0 or not data:
            return 0.0

        data_hex = data.split()
        if count >= 4 and int(data_hex[0], 16) == addr and int(data_hex[1], 16) == 0x24:
            vbus = int(data_hex[2], 16) * 0.1
            return vbus
        return 0.0

    def read_status_flags(self, addr: int) -> Optional[dict]:
        """
        读取使能/到位/堵转状态标志位

        Args:
            addr: 电机地址

        Returns:
            dict: {'enabled': bool, 'arrived': bool, 'stalled': bool} 或 None
        """
        self._flush_input()
        self.read_sys_params(addr, 'S_FLAG')
        data, count = self._receive()

        if count == 0 or not data:
            return None

        data_hex = data.split()
        if count >= 3 and int(data_hex[0], 16) == addr and int(data_hex[1], 16) == 0x3A:
            flag = int(data_hex[2], 16)
            return {
                'enabled': bool(flag & 0x01),
                'arrived': bool(flag & 0x02),
                'stalled': bool(flag & 0x04),
            }
        return None

    def wait_for_arrival(self, addr: int, timeout: float = 10.0, poll_interval: float = 0.05) -> bool:
        """
        轮询等待电机到位

        Args:
            addr: 电机地址
            timeout: 超时时间(秒)
            poll_interval: 轮询间隔(秒)

        Returns:
            True=到位, False=超时或堵转
        """
        start = time.time()
        while time.time() - start < timeout:
            flags = self.read_status_flags(addr)
            if flags is not None:
                if flags['arrived']:
                    return True
                if flags['stalled']:
                    print(f'[ZDTTTL] 电机{addr}: 堵转!')
                    return False
            time.sleep(poll_interval)
        print(f'[ZDTTTL] 电机{addr}: 等待到位超时 ({timeout}s)')
        return False

    def wait_for_origin_done(self, addr: int, timeout: float = 15.0, poll_interval: float = 0.1) -> bool:
        """
        轮询等待回零完成

        Args:
            addr: 电机地址
            timeout: 超时时间(秒)
            poll_interval: 轮询间隔(秒)

        Returns:
            True=回零完成, False=超时或失败
        """
        start = time.time()
        while time.time() - start < timeout:
            self._flush_input()
            self.read_sys_params(addr, 'S_ORG')
            data, count = self._receive()
            if count >= 3:
                data_hex = data.split()
                if int(data_hex[0], 16) == addr and int(data_hex[1], 16) == 0x3B:
                    flag = int(data_hex[2], 16)
                    if not (flag & 0x01):  # 回零已结束
                        if flag & 0x02:
                            print(f'[ZDTTTL] 电机{addr}: 回零失败!')
                            return False
                        return True
            time.sleep(poll_interval)
        print(f'[ZDTTTL] 电机{addr}: 回零超时 ({timeout}s)')
        return False

    # ======================== 便捷方法 ========================

    def move_angle(self, addr: int, angle: float, vel: int = 300, acc: int = 50,
                   pulses_per_rev: int = 3200, wait: bool = True, timeout: float = 10.0) -> bool:
        """
        按角度运动 (相对运动)

        Args:
            addr: 电机地址
            angle: 目标角度(度), 正=CW, 负=CCW
            vel: 速度(RPM)
            acc: 加速度
            pulses_per_rev: 每圈脉冲数 (默认200步电机16细分=3200)
            wait: 是否等待到位
            timeout: 等待超时(秒)

        Returns:
            True=运动完成, False=超时/堵转 (wait=False时始终返回True)
        """
        direction = self.CW if angle >= 0 else self.CCW
        pulses = int(abs(angle) / 360.0 * pulses_per_rev)
        self.pos_control(addr, direction, vel, acc, pulses, relative=False, sync=False)
        if wait:
            time.sleep(0.1)
            self._flush_input()
            return self.wait_for_arrival(addr, timeout=timeout)
        return True


# ======================== 测试主程序 ========================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ZDT TTL 步进电机串口控制测试')
    parser.add_argument('--port', type=str, default=None, help='串口设备路径 (默认: 自动检测CH340)')
    parser.add_argument('--addr', type=int, default=1, help='电机地址 (默认: 1)')
    parser.add_argument('--baudrate', type=int, default=115200, help='波特率 (默认: 115200)')
    parser.add_argument('--speed', type=int, default=300, help='运动速度RPM (默认: 300)')
    parser.add_argument('--acc', type=int, default=50, help='加速度 (默认: 50)')
    parser.add_argument('--pulses-per-rev', type=int, default=3200, help='每圈脉冲数 (默认: 3200)')
    args = parser.parse_args()

    MOTOR_ADDR = args.addr
    SPEED_RPM = args.speed
    ACC = args.acc
    PULSES_PER_REV = args.pulses_per_rev
    PULSES_90DEG = PULSES_PER_REV * 90 // 360
    PULSES_180DEG = PULSES_PER_REV * 180 // 360

    motor = ZDTTTL(port=args.port, baudrate=args.baudrate)

    try:
        # 1. 使能电机
        print('\n=== 使能电机 ===')
        motor.enable(MOTOR_ADDR, True)
        time.sleep(0.5)
        motor._flush_input()

        # 2. 将当前位置清零
        print('\n=== 步骤1: 记录当前位置为零位 ===')
        motor.reset_cur_pos_to_zero(MOTOR_ADDR)
        time.sleep(0.3)
        pos = motor.read_position(MOTOR_ADDR)
        print(f'  当前位置: {pos:.1f}°')

        # 3. 逆时针旋转90°
        print('\n=== 步骤2: 逆时针旋转90° ===')
        motor.pos_control(MOTOR_ADDR, ZDTTTL.CCW, SPEED_RPM, ACC, PULSES_90DEG)
        time.sleep(0.1)
        motor._flush_input()
        print('  等待电机到位...')
        if motor.wait_for_arrival(MOTOR_ADDR, timeout=3):
            print('  到位')
        time.sleep(0.2)
        pos = motor.read_position(MOTOR_ADDR)
        print(f'  当前位置: {pos:.1f}°')

        # 4. 顺时针旋转180°
        print('\n=== 步骤3: 顺时针旋转180° ===')
        motor.pos_control(MOTOR_ADDR, ZDTTTL.CW, SPEED_RPM, ACC, PULSES_180DEG)
        time.sleep(0.1)
        motor._flush_input()
        print('  等待电机到位...')
        if motor.wait_for_arrival(MOTOR_ADDR, timeout=3):
            print('  到位')
        time.sleep(0.2)
        pos = motor.read_position(MOTOR_ADDR)
        print(f'  当前位置: {pos:.1f}°')

        # 5. 逆时针旋转90°回到零位
        print('\n=== 步骤4: 回到零位 ===')
        motor.pos_control(MOTOR_ADDR, ZDTTTL.CCW, SPEED_RPM, ACC, PULSES_90DEG)
        time.sleep(0.1)
        motor._flush_input()
        print('  等待电机到位...')
        if motor.wait_for_arrival(MOTOR_ADDR, timeout=3):
            print('  到位')
        time.sleep(0.2)
        pos = motor.read_position(MOTOR_ADDR)
        print(f'  当前位置: {pos:.1f}°')

        # 6. 测试便捷方法 move_angle
        print('\n=== 步骤5: 使用 move_angle 顺时针转45° ===')
        motor.reset_cur_pos_to_zero(MOTOR_ADDR)
        time.sleep(0.3)
        if motor.move_angle(MOTOR_ADDR, 45.0, vel=SPEED_RPM, acc=ACC, pulses_per_rev=PULSES_PER_REV):
            print('  到位')
        pos = motor.read_position(MOTOR_ADDR)
        print(f'  当前位置: {pos:.1f}°')

        print('\n=== 步骤6: 使用 move_angle 逆时针转45°回到零位 ===')
        if motor.move_angle(MOTOR_ADDR, -45.0, vel=SPEED_RPM, acc=ACC, pulses_per_rev=PULSES_PER_REV):
            print('  到位')
        pos = motor.read_position(MOTOR_ADDR)
        print(f'  当前位置: {pos:.1f}°')

        # 7. 读取状态信息
        print('\n=== 状态信息 ===')
        flags = motor.read_status_flags(MOTOR_ADDR)
        if flags:
            print(f'  使能: {flags["enabled"]}, 到位: {flags["arrived"]}, 堵转: {flags["stalled"]}')
        else:
            print('  无法读取状态')

    except KeyboardInterrupt:
        print('\n用户中断')
        motor.stop(MOTOR_ADDR)
    except serial.SerialException as e:
        print(f'\n串口错误: {e}')
    finally:
        # 失能电机并关闭串口
        try:
            motor.enable(MOTOR_ADDR, False)
            time.sleep(0.1)
        except Exception:
            pass
        motor.close()
