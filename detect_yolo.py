import argparse
import re
import time

import cv2
import numpy as np
from ugot import ugot
from ultralytics import YOLO

from utils import ROBOT_IP, wait_port
from logger import get_logger

_log = get_logger()

CLASS_NAMES = ["red", "green", "blue"]
COLOR_BGR = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
}


def draw_results(frame, results):
    output = frame.copy()
    for x, y, w, h, area, conf, cls_name in results:
        color = COLOR_BGR.get(cls_name, (0, 255, 0))
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(output, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(output, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return output


def main():
    parser = argparse.ArgumentParser(description="PyTorch YOLO 立方体检测")
    parser.add_argument("color", nargs="?", default=None,
                        help="过滤颜色: red / green / blue，不传则显示所有")
    parser.add_argument("--model", default="best.pt",
                        help="模型路径 (默认 best.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="置信度阈值 (默认 0.5)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理分辨率 (默认 640)")
    args = parser.parse_args()

    _log.success("UGOT PyTorch YOLO 立方体检测")
    if args.color:
        _log.bind(color=args.color).info(f"过滤颜色: {args.color}")

    _log.bind(model=args.model).info("加载 YOLO 模型")
    model = YOLO(args.model)
    _log.success("模型加载成功")

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

            results_raw = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            results = []
            boxes = results_raw[0].boxes
            if boxes is not None and boxes.xyxy is not None:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    conf = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
                    if args.color and cls_name != args.color:
                        continue
                    x = int(round(x1))
                    y = int(round(y1))
                    bw = int(round(x2 - x1))
                    bh = int(round(y2 - y1))
                    if bw <= 0 or bh <= 0:
                        continue
                    results.append((x, y, bw, bh, bw * bh, conf, cls_name))

            display = draw_results(frame, results)
            _log.bind(cubes_detected=len(results)).trace("检测结果")
            cv2.imshow("UGOT YOLO Detector", display)
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


if __name__ == "__main__":
    main()
