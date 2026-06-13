# 导入 ugot 库，用于控制 UGOT 机器人
from ugot import ugot
# 导入 time 模块，用于延时
import time

from logger import get_logger

from config import ROBOT_IP

_log = get_logger()


def read_gyro(got):
    data = got.read_gyro_data()
    if data and any(v != 0 for v in data):
        return {
            "pitch": data[0],
            "roll": data[1],
            "yaw": data[2],
            "gyro_x": data[3],
            "gyro_y": data[4],
            "gyro_z": data[5],
            "accel_x": data[6],
            "accel_y": data[7],
            "accel_z": data[8],
        }
    return None


def main():
    got = ugot.UGOT()

    _log.success("=" * 48)
    _log.success("UGOT 陀螺仪数据读取")
    _log.success("=" * 48)

    ip = ROBOT_IP
    if not ip:
        _log.bind(action="scan").info("正在扫描局域网中的 UGOT 设备...")
        devices = got.scan_device()
        if not devices:
            _log.error("未找到任何 UGOT 设备")
            return
        name = list(devices.keys())[0]
        ip = list(devices.values())[0]
        _log.bind(device=name, ip=ip).info("发现设备")

    _log.bind(ip=ip, action="connect").info("正在连接...")
    got.initialize(ip)
    _log.success("连接成功")
    time.sleep(1)

    _log.info("按 Ctrl+C 停止")
    _log.info("格式: pitch roll yaw | gyro_x gyro_y gyro_z | accel_x accel_y accel_z")

    try:
        while True:
            data = read_gyro(got)
            if data:
                _log.bind(**data).info("陀螺仪数据")
            else:
                _log.warning("未获取到陀螺仪数据")
            time.sleep(0.2)
    except KeyboardInterrupt:
        _log.success("已停止")


if __name__ == "__main__":
    main()
