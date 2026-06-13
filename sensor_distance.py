# 导入 ugot 库，用于控制 UGOT 机器人
from ugot import ugot
# 导入 time 模块，用于延时
import time

from logger import get_logger

from config import ROBOT_IP
from utils import discover_infrared_id

_log = get_logger()


def read_distance(got, sensor_id):
    """读取红外测距传感器数值"""
    return got.read_distance_data(sensor_id)


def main():
    got = ugot.UGOT()

    _log.success("=" * 48)
    _log.success("UGOT 红外测距传感器读取")
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

    sensor_id = discover_infrared_id(got)
    _log.bind(sensor_id=sensor_id).info("传感器 ID")
    _log.info("按 Ctrl+C 停止")
    _log.info("开始读取循环")

    try:
        while True:
            distance = read_distance(got, sensor_id)
            if distance == -1:
                _log.bind(sensor_id=sensor_id, distance_cm=-1).warning("未获取到数据")
            else:
                _log.bind(sensor_id=sensor_id, distance_cm=distance).info("距离读数")
            time.sleep(0.5)
    except KeyboardInterrupt:
        _log.success("已停止")


if __name__ == "__main__":
    main()
