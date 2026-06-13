# 导入 ugot 库，用于控制 UGOT 机器人
from ugot import ugot
# 导入 time 模块，用于延时
import time

from logger import get_logger

from config import ROBOT_IP

_log = get_logger()

# 中文类型名映射
TYPE_NAMES = {
    "motor": "电机 (motor)",
    "servo": "舵机 (servo)",
    "power": "电源 (power)",
    "Clamp": "夹手 (Clamp)",
    "Infrared": "红外测距 (Infrared)",
}


def main():
    got = ugot.UGOT()

    _log.success("=" * 56)
    _log.success("UGOT 外设设备诊断")
    _log.success("=" * 56)

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

    _log.bind(action="get_peripherals").info("正在获取外设设备列表...")
    devices = got.get_peripheral_devices_list()

    if not devices:
        _log.warning("未检测到任何外设设备")
        return

    _log.bind(device_count=len(devices)).success(f"共检测到 {len(devices)} 个设备")

    groups = {}

    for dev in devices:
        dev_type = dev.get("type", "unknown")
        dev_id = dev.get("deviceId", "?")
        dev_serial = dev.get("serial", "?")
        dev_fw = dev.get("firmware", "?")

        _log.bind(
            device_type=dev_type,
            device_id=dev_id,
            serial=dev_serial,
            firmware=dev_fw,
        ).info("检测到外设")

        if dev_type not in groups:
            groups[dev_type] = []
        groups[dev_type].append(dev_id)

    _log.success("=" * 56)
    _log.success("设备汇总")
    _log.success("=" * 56)

    summary = {}
    for dev_type, ids in groups.items():
        label = TYPE_NAMES.get(dev_type, dev_type)
        count = len(ids)
        summary[dev_type] = {"label": label, "count": count, "ids": ids}
        _log.bind(device_type=label, count=count, device_ids=ids).info("设备分类汇总")

    _log.bind(summary=summary).success("外设诊断完成")
    _log.success("=" * 56)


if __name__ == "__main__":
    main()
