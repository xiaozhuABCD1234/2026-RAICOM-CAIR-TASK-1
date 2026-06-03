# 导入 ugot 库，用于控制 UGOT 机器人
from ugot import ugot
# 导入 time 模块，用于延时
import time

# 目标机器人 IP 地址
ROBOT_IP = "192.168.1.22"
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


def set_servo_position(got, servo_id, angle, duration_ms=DEFAULT_DURATION, wait=True):
    """控制单个舵机转到指定角度

    Args:
        got: UGOT 机器人实例
        servo_id: 舵机 ID
        angle: 目标角度
        duration_ms: 动作时长（毫秒）
        wait: 是否阻塞等待完成
    """
    got.turn_servo_angle(servo_id, angle, duration_ms, wait=wait)
    if wait:
        time.sleep(duration_ms / 1000.0 + 0.1)


def set_all_servo_positions(got, angle1, angle2, angle3, duration_ms=DEFAULT_DURATION, wait=True):
    """同时控制三个舵机转到指定角度

    Args:
        got: UGOT 机器人实例
        angle1: 关节1 角度
        angle2: 关节2 角度
        angle3: 关节3 角度
        duration_ms: 动作时长（毫秒）
        wait: 是否阻塞等待完成
    """
    for sid, angle in zip(SERVO_IDS, [angle1, angle2, angle3]):
        got.turn_servo_angle(sid, angle, duration_ms, wait=False)
    if wait:
        time.sleep(duration_ms / 1000.0 + 0.1)


def main():
    got = ugot.UGOT()

    print("=" * 48)
    print("  UGOT 舵机控制 - 5号接口")
    print("=" * 48)

    print(f"[INFO] 正在连接 {ROBOT_IP} ...")
    got.initialize(ROBOT_IP)
    print("[INFO] 连接成功")
    time.sleep(1)

    try:
        # 起始位置：关节1=90°, 关节2=90°, 关节3=0°，夹手张开
        print("\n[起始] 关节1: +90°  关节2: +90°  关节3: 0°")
        set_all_servo_positions(got, 90, 90, 0)
        time.sleep(0.5)

        print("[夹手] 张开")
        got.mechanical_clamp_release()
        time.sleep(0.3)

        # 关节2 从 90° → 160°
        print("\n[关节2] 90° → 160° 开始")
        set_servo_position(got, 52, 160, duration_ms=2000)
        print("[关节2] 90° → 160° 完成")

        # 夹手闭合抓住东西
        print("[夹手] 闭合 - 抓住东西")
        got.mechanical_clamp_close()
        time.sleep(0.5)

        # 抬起：关节1=90°, 关节2=20°, 关节3=-80°
        print("\n[抬起] 关节1: +90°  关节2: +20°  关节3: -80°")
        set_all_servo_positions(got, 90, 20, -80, duration_ms=1500)

    except KeyboardInterrupt:
        print("\n[中断] 正在停止...")
    except Exception as e:
        print(f"\n[ERROR] 发生异常: {e}")

    finally:
        print("\n[复位] 关节1: +90°  关节2: +20°  关节3: -80°")
        set_all_servo_positions(got, 90, 20, -80)

    print("=" * 48)
    print("  舵机控制演示结束")
    print("=" * 48)


if __name__ == "__main__":
    main()
