# 导入 cv2（OpenCV），用于图像处理
import cv2
# 导入 numpy，用于数组操作
import numpy as np
# 导入 ugot 库，用于控制 UGOT 机器人
from ugot import ugot
# 导入 re 模块，用于正则表达式匹配（校验 IP 地址）
import re
# 导入 time 模块，用于延时和超时控制
import time

# 从共享模块导入常量、工具函数和检测函数
from common import (ROBOT_IP, SEP, wait_port, detect_cubes)
from logger import get_logger

_log = get_logger()


def draw_results(frame, results, color):
    """在原图上绘制检测到的立方体边界框。"""
    output = frame.copy()
    for x, y, w, h, _ in results:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{color}"
        cv2.putText(output, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return output


# 入口：连接 UGOT 机器人，实时检测指定颜色的立方体
def main():
    import sys

    color = sys.argv[1] if len(sys.argv) > 1 else "red"

    _log.success(SEP)
    _log.success("UGOT 立方体颜色识别")
    _log.bind(color=color).success(f"检测颜色: {color}")
    _log.success(SEP)

    robot = ugot.UGOT()

    ip = ROBOT_IP
    if ip:
        if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ip):
            _log.bind(ip=ip).error("无效的 IP 地址")
            return
        _log.bind(ip=ip, source="config").info("使用指定 IP")
    else:
        _log.bind(action="scan").info("正在扫描局域网中的 UGOT 设备...")
        devices = robot.scan_device()
        if not devices:
            _log.error("未找到任何 UGOT 设备")
            return
        name = list(devices.keys())[0]
        ip = list(devices.values())[0]
        _log.bind(device=name, ip=ip).info("发现设备")

    _log.bind(port=50051, action="port_check").info("正在检测机器人端口...")
    if not wait_port(ip, 50051, timeout=15):
        _log.bind(ip=ip, port=50051).error("端口不可达")
        return
    _log.bind(ip=ip, port=50051).success("端口连通")

    _log.bind(action="init_sdk").info("正在初始化 SDK...")
    for attempt in range(3):
        try:
            robot.initialize(device_ip=ip)
            _log.success("初始化成功")
            break
        except Exception as e:
            _log.bind(attempt=attempt + 1, max_attempts=3).opt(exception=True).warning("初始化尝试失败")
            if attempt < 2:
                time.sleep(2)
    else:
        _log.bind(attempts=3, ip=ip).error("连续 3 次初始化失败，退出")
        return

    _log.bind(action="open_camera").info("正在打开摄像头...")
    robot.open_camera()
    _log.success("摄像头已打开")

    _log.info("按 q 键退出")
    _log.info("开始检测循环")
    try:
        while True:
            data = robot.read_camera_data()
            if data is None:
                _log.warning("摄像头读取帧失败")
                continue

            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                _log.warning("帧解码失败")
                continue

            cubes = detect_cubes(frame, color)
            output = draw_results(frame, cubes, color)

            _log.bind(cubes_detected=len(cubes), color=color).trace("检测结果")

            cv2.imshow("UGOT Cube Detector", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        _log.info("收到停止信号")
    finally:
        _log.info("正在停止...")
        try:
            robot.stop_chassis()
            _log.success("已停止")
        except Exception:
            _log.warning("停止时连接已断开")
        cv2.destroyAllWindows()
        _log.success(SEP)


if __name__ == "__main__":
    main()
