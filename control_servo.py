# 导入 ugot 库，用于控制 UGOT 机器人
from ugot import ugot
# 导入 time 模块，用于延时
import time

from logger import get_logger

from config import ROBOT_IP
# 5 号接口上的舵机 ID（机械臂 3 个关节）
SERVO_IDS = [51, 52, 53]
# 关节名称及角度限制
JOINTS = {
    51: {"name": "关节1", "min": -90, "max": 90},
    52: {"name": "关节2", "min": -80, "max": 110},
    53: {"name": "关节3", "min": -90, "max": 90},
}
# 默认动作时长（毫秒）
DEFAULT_DURATION = 800

_log = get_logger()


def set_servo_position(got, servo_id, angle, duration_ms=DEFAULT_DURATION, wait=True):
    """控制单个舵机转到指定角度"""
    _log.bind(servo_id=servo_id, angle=angle, duration_ms=duration_ms).debug("舵机移动开始")
    got.turn_servo_angle(servo_id, angle, duration_ms, wait=wait)
    if wait:
        time.sleep(duration_ms / 1000.0 + 0.1)
    _log.bind(servo_id=servo_id, angle=angle).debug("舵机移动完成")


def set_all_servo_positions(got, angle1, angle2, angle3, duration_ms=DEFAULT_DURATION, wait=True):
    """同时控制三个舵机转到指定角度"""
    angles = [angle1, angle2, angle3]
    _log.bind(joints=dict(zip(SERVO_IDS, angles)), duration_ms=duration_ms).debug("多舵机移动开始")
    for sid, angle in zip(SERVO_IDS, angles):
        got.turn_servo_angle(sid, angle, duration_ms, wait=False)
    if wait:
        time.sleep(duration_ms / 1000.0 + 0.1)
    _log.bind(joints=dict(zip(SERVO_IDS, angles))).debug("多舵机移动完成")


def main():
    got = ugot.UGOT()

    _log.success("=" * 48)
    _log.success("UGOT 舵机控制 - 5号接口")
    _log.success("=" * 48)

    _log.bind(ip=ROBOT_IP, action="connect").info("正在连接...")
    got.initialize(ROBOT_IP)
    _log.success("连接成功")
    time.sleep(1)

    try:
        _log.bind(joint1=90, joint2=90, joint3=0).info("起始位置")
        set_all_servo_positions(got, 90, 90, 0)
        time.sleep(0.5)

        _log.bind(action="clamp_release").info("夹手张开")
        got.mechanical_clamp_release()
        time.sleep(0.3)

        _log.bind(servo_id=52, from_deg=90, to_deg=160, duration_ms=2000).info("关节2 移动")
        set_servo_position(got, 52, 160, duration_ms=2000)

        _log.bind(action="clamp_close").info("夹手闭合")
        got.mechanical_clamp_close()
        time.sleep(0.5)

        _log.bind(joint1=90, joint2=20, joint3=-80, duration_ms=1500).info("抬起")
        set_all_servo_positions(got, 90, 20, -80, duration_ms=1500)

    except KeyboardInterrupt:
        _log.info("收到停止信号")
    except Exception:
        _log.opt(exception=True).error("发生异常")

    finally:
        _log.bind(joint1=90, joint2=20, joint3=-80).info("复位")
        set_all_servo_positions(got, 90, 20, -80)

    _log.success("=" * 48)
    _log.success("舵机控制演示结束")
    _log.success("=" * 48)


if __name__ == "__main__":
    main()
