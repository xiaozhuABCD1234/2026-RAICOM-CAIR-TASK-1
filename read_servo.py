# 导入 ugot 库，用于控制 UGOT 机器人
from ugot import ugot
# 导入 time 模块，用于延时
import time

# 目标机器人 IP 地址
ROBOT_IP = "192.168.1.22"
# 5 号接口上的舵机 ID（机械臂 3 个关节）
SERVO_IDS = [51, 52, 53]
# 关节名称
JOINT_NAMES = {51: "关节1", 52: "关节2", 53: "关节3"}


def read_servo_position(got, servo_id):
    """读取指定舵机角度

    Args:
        got: UGOT 机器人实例
        servo_id: 舵机 ID

    Returns:
        角度值 (float)，读取失败返回 None
    """
    result = got.read_servo_angle(servo_id)
    return result.get(str(servo_id), None)


def main():
    got = ugot.UGOT()

    print("=" * 48)
    print("  UGOT 舵机角度读取 - 5号接口")
    print("=" * 48)

    print(f"[INFO] 正在连接 {ROBOT_IP} ...")
    got.initialize(ROBOT_IP)
    print("[INFO] 连接成功")
    time.sleep(1)

    print(f"[INFO] 舵机 ID: {SERVO_IDS}")
    print("[INFO] 按 Ctrl+C 停止\n")

    try:
        while True:
            parts = []
            for sid in SERVO_IDS:
                angle = read_servo_position(got, sid)
                name = JOINT_NAMES.get(sid, f"ID{sid}")
                if angle is not None:
                    parts.append(f"{name}: {angle:+.0f}°")
                else:
                    parts.append(f"{name}: --")
            print(f"  [{time.strftime('%H:%M:%S')}]  {'  '.join(parts)}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] 已停止")


if __name__ == "__main__":
    main()
