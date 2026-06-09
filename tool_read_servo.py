# 导入 ugot 库，用于控制 UGOT 机器人
from ugot import ugot
# 导入 time 模块，用于延时
import time

from logger import get_logger

from config import ROBOT_IP
# 5 号接口上的舵机 ID（机械臂 3 个关节）
SERVO_IDS = [51, 52, 53]
# 关节名称
JOINT_NAMES = {51: "关节1", 52: "关节2", 53: "关节3"}

_log = get_logger()


def read_servo_position(got, servo_id):
    """读取指定舵机角度"""
    result = got.read_servo_angle(servo_id)
    return result.get(str(servo_id), None)


def main():
    got = ugot.UGOT()

    _log.success("=" * 48)
    _log.success("UGOT 舵机角度读取 - 5号接口")
    _log.success("=" * 48)

    _log.bind(ip=ROBOT_IP, action="connect").info("正在连接...")
    got.initialize(ROBOT_IP)
    _log.success("连接成功")
    time.sleep(1)

    _log.bind(servo_ids=SERVO_IDS).info("舵机 ID")
    _log.info("按 Ctrl+C 停止")
    _log.info("开始读取循环")

    try:
        while True:
            angles = {}
            for sid in SERVO_IDS:
                angle = read_servo_position(got, sid)
                angles[f"joint_{sid}"] = angle
            _log.bind(**angles).info("舵机角度")
            time.sleep(0.5)
    except KeyboardInterrupt:
        _log.success("已停止")


if __name__ == "__main__":
    main()
