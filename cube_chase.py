# 导入 cv2（OpenCV），用于图像处理
import cv2
# 导入 numpy，用于数组操作
import numpy as np
# 导入 ugot 库，用于控制 UGOT 机器人
from ugot import ugot
# 导入 re 模块，用于正则表达式匹配（校验 IP 地址）
import re
# 导入 time 模块，用于延时
import time
# 导入 threading 模块，用于双线程异步架构
import threading

# 从共享模块导入常量、工具函数和检测函数
from common import (ROBOT_IP, SEP, COLOR_RANGES, wait_port, detect_cubes)
from logger import get_logger

# 追踪参数
SEARCH_SPEED = 15
# PID 控制的比例系数、积分系数、微分系数
KP, KI, KD = 0.25, 0, 0.05
# 追击前进速度（cm/s）
CHASE_SPEED = 15
# 转向速度上限，防止 PID 输出过大
TURN_SPEED_MAX = 40
# 红外测距传感器 ID（与 distance_sensor.py 一致）
SENSOR_ID = 41
# 目标距离（cm），距离 PID 的目标值
TARGET_DISTANCE = 10
# 距离 PID 的比例系数、积分系数、微分系数
DISTANCE_KP, DISTANCE_KI, DISTANCE_KD = 2.0, 0, 0.1
# 后退速度（cm/s），太近时向后修正
BACKWARD_SPEED = 7

_log = get_logger()


def get_largest_cube(cubes):
    """从检测结果中选取面积最大的立方体。"""
    if not cubes:
        return None
    return max(cubes, key=lambda c: c[4])


def chase(robot, color, headless=False):
    """追踪指定颜色的立方体，使机器人始终正对目标。"""
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
    _log.bind(pid="distance", kp=DISTANCE_KP, ki=DISTANCE_KI, kd=DISTANCE_KD,
              target_cm=TARGET_DISTANCE).info("距离 PID 配置")

    # ===== 控制线程：50ms 周期发送电机指令 =====
    @_log.catch(reraise=False)
    def control_loop():
        with _log.contextualize(thread="control"):
            search_dir = 2
            search_since = 0
            while not stop_event.is_set():
                # ===== 距离传感器：PID 控制前进速度，精确维持目标距离 =====
                distance = robot.read_distance_data(SENSOR_ID)
                if distance <= 0:
                    _log.bind(sensor_id=SENSOR_ID, value=distance).critical("距离传感器无数据")
                    stop_event.set()
                    return
                dist_error = round(pid_dist.update(distance - TARGET_DISTANCE))

                with lock:
                    found = state["found"]
                    offset = state["offset"]
                    area = state["area"]

                if not found:
                    now = time.time()
                    if now - search_since > 3:
                        search_dir = 3 if search_dir == 2 else 2
                        search_since = now
                    robot.mecanum_move_turn(0, 0, search_dir, SEARCH_SPEED)
                    _log.bind(
                        state="searching",
                        direction="left" if search_dir == 2 else "right",
                        distance_cm=distance,
                        dist_error=dist_error,
                    ).trace("搜索旋转")
                else:
                    search_since = time.time()
                    dic = round(pid.update(offset))

                    turn_speed = min(abs(dic), TURN_SPEED_MAX)

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

    # ===== 视觉线程：取帧 + 检测 =====
    @_log.catch(reraise=False)
    def vision_loop():
        with _log.contextualize(thread="vision"):
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

                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    _log.warning("帧解码失败")
                    continue

                frame_h, frame_w = frame.shape[:2]
                center_x = frame_w // 2

                cubes = detect_cubes(frame, color)
                largest = get_largest_cube(cubes)

                with lock:
                    if largest is not None:
                        x, y, w, h, area = largest
                        state["offset"] = (x + w // 2) - center_x
                        state["area"] = area
                        state["found"] = True
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.line(frame, (center_x, 0), (center_x, frame_h), (255, 255, 0), 1)
                        state["frame"] = frame
                    else:
                        state["found"] = False
                        state["frame"] = frame

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
                    cv2.imshow(f"Cube Chase - {color}", display_frame)
                    if cv2.waitKey(50) & 0xFF == ord("q"):
                        stop_event.set()
                        break
            else:
                stop_event.wait(0.05)
    except KeyboardInterrupt:
        _log.info("收到停止信号")
    finally:
        stop_event.set()
        robot.stop_chassis()
        if not headless:
            cv2.destroyAllWindows()
        _log.success("已停止")


def main():
    """入口：解析命令行参数，连接机器人，启动追踪。"""
    import sys

    args = sys.argv[1:]
    headless = "--headless" in args
    if headless:
        args.remove("--headless")

    color = args[0] if args else "red"
    if color not in COLOR_RANGES:
        _log.bind(color=color, supported=list(COLOR_RANGES.keys())).error("不支持的颜色")
        return

    _log.success(SEP)
    _log.bind(color=color).success("UGOT 方块追踪")
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
    time.sleep(1)

    _log.info("进入追踪主循环")
    chase(robot, color, headless=headless)
    _log.info("追踪结束")


if __name__ == "__main__":
    main()
