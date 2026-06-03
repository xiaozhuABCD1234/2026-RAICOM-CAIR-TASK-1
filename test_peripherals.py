# 导入 ugot 库，用于控制 UGOT 机器人
from ugot import ugot

# 导入 time 模块，用于延时
import time

# 目标机器人 IP 地址
ROBOT_IP = "192.168.1.22"


def main():
    got = ugot.UGOT()

    print("=" * 56)
    print("  UGOT 外设设备诊断")
    print("=" * 56)

    print(f"[INFO] 正在连接 {ROBOT_IP} ...")
    got.initialize(ROBOT_IP)
    print("[INFO] 连接成功")
    time.sleep(1)

    # 获取外设设备列表
    print("\n[INFO] 正在获取外设设备列表...")
    devices = got.get_peripheral_devices_list()

    if not devices:
        print("[WARN] 未检测到任何外设设备")
        return

    print(f"[INFO] 共检测到 {len(devices)} 个设备:\n")

    # 分类统计
    groups = {}

    for dev in devices:
        dev_type = dev.get("type", "unknown")
        dev_id = dev.get("deviceId", "?")
        dev_serial = dev.get("serial", "?")
        dev_fw = dev.get("firmware", "?")

        # 打印每个设备的详细信息
        print(
            f"  类型: {dev_type:<12s}  ID: {dev_id:<6s}  序列号: {dev_serial:<16s}  固件: {dev_fw}"
        )

        if dev_type not in groups:
            groups[dev_type] = []
        groups[dev_type].append(dev_id)

    # 汇总
    print("\n" + "=" * 56)
    print("  设备汇总")
    print("=" * 56)

    # 中文类型名映射
    type_names = {
        "motor": "电机 (motor)",
        "servo": "舵机 (servo)",
        "power": "电源 (power)",
        "Clamp": "夹手 (Clamp)",
        "Infrared": "红外测距 (Infrared)",
    }

    for dev_type, ids in groups.items():
        label = type_names.get(dev_type, dev_type)
        print(f"  {label:<20s}  {len(ids)} 个  → ID: {ids}")

    print("=" * 56)


if __name__ == "__main__":
    main()
