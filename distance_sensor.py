# 导入 ugot 库，用于控制 UGOT 机器人
from ugot import ugot
# 导入 time 模块，用于延时
import time

from logger import get_logger

from config import ROBOT_IP
# 红外测距传感器 ID
SENSOR_ID = 41

_log = get_logger()


def read_distance(got, sensor_id=SENSOR_ID):
    """读取红外测距传感器数值"""
    return got.read_distance_data(sensor_id)


def main():
    got = ugot.UGOT()

    _log.success("=" * 48)
    _log.success("UGOT 红外测距传感器读取")
    _log.success("=" * 48)

    _log.bind(ip=ROBOT_IP, action="connect").info("正在连接...")
    got.initialize(ROBOT_IP)
    _log.success("连接成功")
    time.sleep(1)

    _log.bind(sensor_id=SENSOR_ID).info("传感器 ID")
    _log.info("按 Ctrl+C 停止")
    _log.info("开始读取循环")

    try:
        while True:
            distance = read_distance(got, SENSOR_ID)
            if distance == -1:
                _log.bind(sensor_id=SENSOR_ID, distance_cm=-1).warning("未获取到数据")
            else:
                _log.bind(sensor_id=SENSOR_ID, distance_cm=distance).info("距离读数")
            time.sleep(0.5)
    except KeyboardInterrupt:
        _log.success("已停止")


if __name__ == "__main__":
    main()
