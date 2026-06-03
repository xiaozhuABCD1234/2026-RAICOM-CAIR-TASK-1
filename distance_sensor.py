# 导入 ugot 库，用于控制 UGOT 机器人
from ugot import ugot
# 导入 time 模块，用于延时
import time

# 目标机器人 IP 地址
ROBOT_IP = "192.168.1.22"
# 红外测距传感器 ID
SENSOR_ID = 41


def read_distance(got, sensor_id=SENSOR_ID):
    """读取红外测距传感器数值

    Args:
        got: UGOT 机器人实例
        sensor_id: 传感器 ID

    Returns:
        距离值 (float)，单位 cm；-1 表示未获取到数据
    """
    return got.read_distance_data(sensor_id)


def main():
    got = ugot.UGOT()

    print("=" * 48)
    print("  UGOT 红外测距传感器读取")
    print("=" * 48)

    print(f"[INFO] 正在连接 {ROBOT_IP} ...")
    got.initialize(ROBOT_IP)
    print("[INFO] 连接成功")
    time.sleep(1)

    print(f"[INFO] 传感器 ID: {SENSOR_ID}")
    print("[INFO] 按 Ctrl+C 停止\n")

    try:
        while True:
            distance = read_distance(got, SENSOR_ID)
            if distance == -1:
                print(f"  [{time.strftime('%H:%M:%S')}] 距离: 未获取到数据")
            else:
                print(f"  [{time.strftime('%H:%M:%S')}] 距离: {distance:.1f} cm")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] 已停止")


if __name__ == "__main__":
    main()
