import argparse
import re
import threading
import time

import cv2
import numpy as np
from ugot import ugot
from ultralytics import YOLO

from utils import ROBOT_IP, wait_port, discover_infrared_id
from logger import get_logger

SEARCH_SPEED = 30
KP, KI, KD = 0.25, 0, 0.05
CHASE_SPEED = 15
# 转向速度上限由距离自适应决定（在控制循环中计算）
TARGET_DISTANCE = 9
DISTANCE_KP, DISTANCE_KI, DISTANCE_KD = 2.0, 0, 0.1
BACKWARD_SPEED = 7

_log = get_logger()

CLASS_NAMES = ["red", "green", "blue"]


def get_largest_cube(cubes):
    if not cubes:
        return None
    return max(cubes, key=lambda c: c[4])


def chase(robot, model_path, args, sensor_id=41):
    color = args.color
    headless = args.headless
    conf_thres = args.conf
    imgsz = args.imgsz

    _log.bind(color=color, headless=headless).info("开始追踪")
    _log.info("按 Ctrl+C 停止")
    if headless:
        _log.info("无头模式，不显示画面")

    state = {"offset": None, "area": 0, "found": False, "frame": None}
    lock = threading.Lock()
    stop_event = threading.Event()

    pid = robot.create_pid_controller()
    pid.set_pid(KP, KI, KD)
    _log.bind(pid="horizontal", kp=KP, ki=KI, kd=KD).info("水平 PID 配置")

    pid_dist = robot.create_pid_controller()
    pid_dist.set_pid(DISTANCE_KP, DISTANCE_KI, DISTANCE_KD)
    _log.bind(
        pid="distance",
        kp=DISTANCE_KP,
        ki=DISTANCE_KI,
        kd=DISTANCE_KD,
        target_cm=TARGET_DISTANCE,
    ).info("距离 PID 配置")

    @_log.catch(reraise=False)
    def control_loop():
        with _log.contextualize(thread="control"):
            while not stop_event.is_set():
                distance = robot.read_distance_data(sensor_id)
                if distance <= 0:
                    _log.bind(sensor_id=sensor_id, value=distance).critical(
                        "距离传感器无数据"
                    )
                    stop_event.set()
                    return
                dist_error = round(pid_dist.update(distance - TARGET_DISTANCE))

                with lock:
                    found = state["found"]
                    offset = state["offset"]
                    area = state["area"]

                if not found:
                    robot.mecanum_move_turn(0, 0, 2, SEARCH_SPEED)
                    _log.bind(
                        state="searching",
                        distance_cm=distance,
                        dist_error=dist_error,
                    ).trace("搜索旋转")
                else:
                    dic = round(pid.update(offset))
                    if distance > 25:
                        max_turn = 40
                    elif distance > 15:
                        max_turn = 25
                    else:
                        max_turn = 15
                    turn_speed = min(abs(dic), max_turn)

                    if dist_error < 0:
                        dist_forward = int(min(-dist_error, CHASE_SPEED))
                    elif dist_error > 0:
                        backward = int(min(dist_error, BACKWARD_SPEED))
                        robot.mecanum_move_speed(1, backward)
                        _log.bind(
                            state="backward",
                            distance_cm=distance,
                            offset_px=offset,
                            pid_h=dic,
                            area_px=area,
                            dist_error=dist_error,
                            backward_speed=backward,
                        ).trace("后退修正")
                        stop_event.wait(0.05)
                        continue
                    else:
                        dist_forward = 0

                    if dist_forward == 0:
                        if turn_speed < 3:
                            robot.stop_chassis()
                            _log.bind(
                                state="idle",
                                distance_cm=distance,
                                offset_px=offset,
                                pid_h=dic,
                                area_px=area,
                                dist_error=dist_error,
                            ).trace("待命")
                        elif dic < 0:
                            robot.mecanum_move_turn(0, 0, 3, turn_speed)
                            _log.bind(
                                state="turn_right",
                                distance_cm=distance,
                                offset_px=offset,
                                pid_h=dic,
                                area_px=area,
                                dist_error=dist_error,
                                turn_speed=turn_speed,
                            ).trace("右转对准")
                        else:
                            robot.mecanum_move_turn(0, 0, 2, turn_speed)
                            _log.bind(
                                state="turn_left",
                                distance_cm=distance,
                                offset_px=offset,
                                pid_h=dic,
                                area_px=area,
                                dist_error=dist_error,
                                turn_speed=turn_speed,
                            ).trace("左转对准")
                    elif turn_speed < 3:
                        robot.mecanum_move_speed(0, dist_forward)
                        _log.bind(
                            state="forward",
                            distance_cm=distance,
                            offset_px=offset,
                            pid_h=dic,
                            area_px=area,
                            dist_error=dist_error,
                            forward_speed=dist_forward,
                        ).trace("前进")
                    elif dic < 0:
                        robot.mecanum_move_turn(0, dist_forward, 3, turn_speed)
                        _log.bind(
                            state="forward_right",
                            distance_cm=distance,
                            offset_px=offset,
                            pid_h=dic,
                            area_px=area,
                            dist_error=dist_error,
                            forward_speed=dist_forward,
                            turn_speed=turn_speed,
                        ).trace("前进右转")
                    else:
                        robot.mecanum_move_turn(0, dist_forward, 2, turn_speed)
                        _log.bind(
                            state="forward_left",
                            distance_cm=distance,
                            offset_px=offset,
                            pid_h=dic,
                            area_px=area,
                            dist_error=dist_error,
                            forward_speed=dist_forward,
                            turn_speed=turn_speed,
                        ).trace("前进左转")

                stop_event.wait(0.05)

    with lock:
        state["frame"] = np.zeros((480, 640, 3), dtype=np.uint8)

    def vision_loop():
        with _log.contextualize(thread="vision"):
            try:
                local_model = YOLO(model_path)
                _log.success("视觉线程模型加载成功")
            except Exception:
                _log.opt(exception=True).error("视觉线程模型加载失败")
                stop_event.set()
                return

            _log.info("预热模型...")
            try:
                local_model(np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                            imgsz=imgsz, conf=conf_thres, verbose=False)
                _log.success("模型预热完成")
            except Exception:
                _log.opt(exception=True).warning("模型预热失败，继续运行")

            while not stop_event.is_set():
                try:
                    data = robot.read_camera_data()
                except Exception:
                    _log.opt(exception=True).warning("摄像头读取异常")
                    stop_event.wait(0.01)
                    continue
                if data is None:
                    stop_event.wait(0.01)
                    continue

                try:
                    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        _log.warning("帧解码失败")
                        continue
                except Exception:
                    _log.opt(exception=True).warning("帧解码异常")
                    continue

                with lock:
                    state["frame"] = frame

                frame_h, frame_w = frame.shape[:2]
                center_x = frame_w // 2

                try:
                    results_raw = local_model(frame, imgsz=imgsz, conf=conf_thres, verbose=False)
                except Exception:
                    _log.opt(exception=True).error("推理异常")
                    continue

                cubes = []
                boxes = results_raw[0].boxes
                if boxes is not None and boxes.xyxy is not None:
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                        conf = float(boxes.conf[i])
                        cls_id = int(boxes.cls[i])
                        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
                        if color and cls_name != color:
                            continue
                        x = int(round(x1))
                        y = int(round(y1))
                        bw = int(round(x2 - x1))
                        bh = int(round(y2 - y1))
                        if bw <= 0 or bh <= 0:
                            continue
                        cubes.append((x, y, bw, bh, bw * bh, conf, cls_name))

                largest = get_largest_cube(cubes)

                with lock:
                    if largest is not None:
                        x, y, w, h, area, conf, cls_name = largest
                        state["offset"] = (x + w // 2) - center_x
                        state["area"] = area
                        state["found"] = True
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.line(frame, (center_x, 0), (center_x, frame_h), (255, 255, 0), 1)
                    else:
                        state["found"] = False

    ctrl_thread = threading.Thread(target=control_loop, daemon=True)
    vis_thread = threading.Thread(target=vision_loop, daemon=True)

    try:
        ctrl_thread.start()
        vis_thread.start()
        while vis_thread.is_alive() and ctrl_thread.is_alive() and not stop_event.is_set():
            if not headless:
                with lock:
                    display_frame = state["frame"]
                if display_frame is not None:
                    cv2.imshow(f"Cube Chase - {color or 'all'}", display_frame)
                    if cv2.waitKey(50) & 0xFF == ord("q"):
                        stop_event.set()
                        break
            else:
                stop_event.wait(0.05)
    except KeyboardInterrupt:
        _log.info("收到停止信号")
    finally:
        stop_event.set()
        try:
            robot.stop_chassis()
        except Exception:
            pass
        if not headless:
            cv2.destroyAllWindows()
        _log.success("已停止")


def main():
    parser = argparse.ArgumentParser(description="PyTorch YOLO 立方体追踪")
    parser.add_argument("color", nargs="?", default=None,
                        help="追踪颜色: red / green / blue，不传则追踪所有")
    parser.add_argument("--model", default="best.pt",
                        help="模型路径 (默认 best.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="置信度阈值 (默认 0.5)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理分辨率 (默认 640)")
    parser.add_argument("--headless", action="store_true",
                        help="无头模式，不显示画面")
    args = parser.parse_args()

    _log.success("UGOT PyTorch YOLO 立方体追踪")
    if args.color:
        _log.bind(color=args.color).info(f"追踪颜色: {args.color}")

    _log.bind(model=args.model).info("使用模型 (在视觉线程中加载)")
    model_path = args.model

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
        except Exception:
            _log.bind(attempt=attempt + 1, max_attempts=3).opt(exception=True).warning("初始化尝试失败")
            if attempt < 2:
                time.sleep(2)
    else:
        _log.bind(attempts=3, ip=ip).error("连续 3 次初始化失败，退出")
        return

    sensor_id = discover_infrared_id(robot)
    _log.bind(sensor_id=sensor_id).info("红外传感器 ID")

    _log.bind(action="open_camera").info("正在打开摄像头...")
    robot.open_camera()
    _log.success("摄像头已打开")
    time.sleep(1)

    _log.info("进入追踪主循环")
    chase(robot, model_path, args, sensor_id=sensor_id)
    _log.info("追踪结束")


if __name__ == "__main__":
    main()
