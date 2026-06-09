import sys
import tty
import termios
import select
import os
import atexit
from datetime import datetime

import argparse

import cv2
import numpy as np
from ugot import ugot
from ultralytics import YOLO

from common import ROBOT_IP
from logger import get_logger
from pt_cube_detector import CLASS_NAMES, draw_results

_log = get_logger()

MOVE_SPEED = 35
TURN_SPEED = 60

fd = sys.stdin.fileno()
_old_term = termios.tcgetattr(fd)


def _restore_term():
    termios.tcsetattr(fd, termios.TCSADRAIN, _old_term)


atexit.register(_restore_term)


def get_key():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.buffer.read(1).decode("utf-8", errors="ignore")
    return None


def main():
    parser = argparse.ArgumentParser(description="UGOT 遥控 + YOLO 检测")
    parser.add_argument("--model", default="best.pt",
                        help="模型路径 (默认 best.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="置信度阈值 (默认 0.5)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理分辨率 (默认 640)")
    parser.add_argument("--color", default=None,
                        help="过滤颜色: red / green / blue")
    parser.add_argument("--annotate", action="store_true",
                        help="显示检测标注框 (默认不显示)")
    args = parser.parse_args()

    robot = ugot.UGOT()
    _log.bind(action="init").info("正在连接机器人...")
    robot.initialize(device_ip=ROBOT_IP)
    _log.success("连接成功")

    _log.bind(action="open_camera").info("正在打开摄像头...")
    robot.open_camera()
    _log.success("摄像头已打开")

    _log.bind(model=args.model).info("加载 YOLO 模型")
    model = YOLO(args.model)
    _log.success("模型加载成功")

    os.makedirs("captures", exist_ok=True)

    print("=" * 60)
    print("  W/A/S/D  前进 / 左转 / 后退 / 右转")
    print("  SPACE    停止")
    print("  Enter    保存图片")
    print("  Q        退出")
    print("=" * 60)

    try:
        tty.setraw(fd)
        while True:
            data = robot.read_camera_data()
            if data is None:
                continue

            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # 始终运行 YOLO 推理
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

            display_frame = draw_results(frame, results) if args.annotate else frame

            cv2.imshow("UGOT Camera", display_frame)
            cv2.waitKey(1)

            k = get_key()
            if k is None:
                continue

            if k == "w":
                robot.mecanum_move_speed(0, MOVE_SPEED)
                print("\r[W] 前进  ", end="", flush=True)
            elif k == "s":
                robot.mecanum_move_speed(1, MOVE_SPEED)
                print("\r[S] 后退  ", end="", flush=True)
            elif k == "a":
                robot.mecanum_turn_speed(2, TURN_SPEED)
                print("\r[A] 左转  ", end="", flush=True)
            elif k == "d":
                robot.mecanum_turn_speed(3, TURN_SPEED)
                print("\r[D] 右转  ", end="", flush=True)
            elif k == " ":
                robot.stop_chassis()
                print("\r[SPACE] 停止  ", end="", flush=True)
            elif k == "\r":
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                fname = f"captures/capture_{ts}.jpg"
                cv2.imwrite(fname, frame)
                _log.bind(file=fname).success("已保存")
            elif k in ("q", "\x03"):
                break

    except KeyboardInterrupt:
        _log.info("收到停止信号")
    finally:
        _restore_term()
        try:
            robot.stop_chassis()
        except Exception:
            pass
        cv2.destroyAllWindows()
        _log.success("已退出")


if __name__ == "__main__":
    main()
