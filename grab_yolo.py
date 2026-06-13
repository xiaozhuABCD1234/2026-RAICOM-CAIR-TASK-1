import cv2
import numpy as np
from ugot import ugot
import re
import time
import threading
import sys

from utils import ROBOT_IP, COLOR_RANGES, wait_port, detect_cubes, discover_infrared_id
from logger import get_logger

from chase_cv import SEARCH_SPEED, KP, KI, KD, CHASE_SPEED, TURN_SPEED_MAX, \
    TARGET_DISTANCE, DISTANCE_KP, DISTANCE_KI, DISTANCE_KD, \
    BACKWARD_SPEED, get_largest_cube
from control_servo import set_servo_position, set_all_servo_positions

GRAB_DISTANCE_THRESHOLD = 10
GRAB_OFFSET_THRESHOLD = 20
GRAB_STABLE_FRAMES = 10

_log = get_logger()


def track_and_wait(robot, color, headless, grab_event, sensor_id=41):
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
        stable_counter = 0
        with _log.contextualize(thread="control"):
            while not stop_event.is_set():
                distance = robot.read_distance_data(sensor_id)
                if distance <= 0:
                    _log.bind(sensor_id=sensor_id, value=distance).critical("距离传感器无数据")
                    stop_event.set()
                    return

                dist_error = round(pid_dist.update(distance - TARGET_DISTANCE))

                with lock:
                    found = state["found"]
                    offset = state["offset"]
                    area = state["area"]

                if not found:
                    stable_counter = 0
                    robot.mecanum_move_turn(0, 0, 2, SEARCH_SPEED)
                    _log.bind(state="searching", distance_cm=distance, dist_error=dist_error).trace("搜索旋转")
                else:
                    dic = round(pid.update(offset))
                    turn_speed = min(abs(dic), TURN_SPEED_MAX)

                    if dist_error < 0:
                        dist_forward = int(min(-dist_error, CHASE_SPEED))
                    elif dist_error > 0:
                        backward = int(min(dist_error, BACKWARD_SPEED))
                        robot.mecanum_move_speed(1, backward)
                        stable_counter = 0
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
                            abs_offset = abs(offset) if offset else 999
                            if distance <= GRAB_DISTANCE_THRESHOLD and abs_offset < GRAB_OFFSET_THRESHOLD:
                                stable_counter += 1
                                _log.bind(
                                    state="idle",
                                    distance_cm=distance,
                                    offset_px=offset,
                                    pid_h=dic,
                                    area_px=area,
                                    stable=stable_counter,
                                    needed=GRAB_STABLE_FRAMES,
                                ).trace("待命就绪")
                                if stable_counter >= GRAB_STABLE_FRAMES:
                                    _log.bind(
                                        distance_cm=distance,
                                        offset_px=offset,
                                    ).success("已对准方块，准备抓取")
                                    grab_event.set()
                                    stop_event.set()
                                    return
                            else:
                                stable_counter = 0
                                _log.bind(
                                    state="idle",
                                    distance_cm=distance,
                                    offset_px=offset,
                                    pid_h=dic,
                                    area_px=area,
                                    dist_error=dist_error,
                                    stable=stable_counter,
                                    needed=GRAB_STABLE_FRAMES,
                                ).trace("待命")
                        elif dic < 0:
                            stable_counter = 0
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
                            stable_counter = 0
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
                        stable_counter = 0
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
                        stable_counter = 0
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
                        stable_counter = 0
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

    ctrl_thread.start()
    vis_thread.start()

    _log.info("进入追踪主循环")
    try:
        while vis_thread.is_alive() and ctrl_thread.is_alive() and not stop_event.is_set():
            if headless:
                if grab_event.wait(0.05):
                    break
            else:
                with lock:
                    display_frame = state["frame"]
                if display_frame is not None:
                    cv2.imshow(f"Track & Grab - {color}", display_frame)
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord("q"):
                        _log.info("用户按下 Q 键，停止追踪")
                        stop_event.set()
                        break
                else:
                    stop_event.wait(0.05)
    except KeyboardInterrupt:
        _log.info("收到停止信号")
        stop_event.set()
    finally:
        stop_event.set()
        try:
            robot.stop_chassis()
        except Exception:
            pass
        if not headless:
            cv2.destroyAllWindows()


def main():
    args = sys.argv[1:]
    headless = "--headless" in args
    if headless:
        args.remove("--headless")

    color = args[0] if args else "red"
    if color not in COLOR_RANGES:
        _log.bind(color=color, supported=list(COLOR_RANGES.keys())).error("不支持的颜色")
        return

    _log.success("=" * 48)
    _log.success("UGOT 追踪 + 抓取")
    _log.success("=" * 48)

    got = ugot.UGOT()

    ip = ROBOT_IP
    if ip:
        if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ip):
            _log.bind(ip=ip).error("无效的 IP 地址")
            return
        _log.bind(ip=ip, source="config").info("使用指定 IP")
    else:
        _log.bind(action="scan").info("正在扫描局域网中的 UGOT 设备...")
        devices = got.scan_device()
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
            got.initialize(device_ip=ip)
            _log.success("初始化成功")
            break
        except Exception:
            _log.bind(attempt=attempt + 1, max_attempts=3).opt(exception=True).warning("初始化尝试失败")
            if attempt < 2:
                time.sleep(2)
    else:
        _log.bind(attempts=3, ip=ip).error("连续 3 次初始化失败，退出")
        return

    sensor_id = discover_infrared_id(got)
    _log.bind(sensor_id=sensor_id).info("红外传感器 ID")

    _log.bind(action="open_camera").info("正在打开摄像头...")
    got.open_camera()
    _log.success("摄像头已打开")
    time.sleep(1)

    try:
        _log.bind(joint1=90, joint2=90, joint3=0).info("机械臂归位")
        set_all_servo_positions(got, 90, 90, 0)
        time.sleep(0.5)

        _log.bind(action="clamp_release").info("夹手张开")
        got.mechanical_clamp_release()
        time.sleep(0.3)

        grab_event = threading.Event()
        track_and_wait(got, color, headless, grab_event, sensor_id=sensor_id)

        if not grab_event.is_set():
            _log.info("未触发抓取，结束")
            return

        _log.info("开始执行抓取序列")
        time.sleep(0.5)

        _log.bind(servo_id=52, from_deg=90, to_deg=160, duration_ms=2000).info("关节2 下压")
        set_servo_position(got, 52, 160, duration_ms=2000)

        _log.bind(action="clamp_close").info("夹手闭合")
        got.mechanical_clamp_close()
        time.sleep(0.5)

        _log.bind(joint1=90, joint2=20, joint3=-80, duration_ms=1500).info("抬起")
        set_all_servo_positions(got, 90, 20, -80, duration_ms=1500)

        _log.success("抓取完成")

    except KeyboardInterrupt:
        _log.info("收到停止信号")
    except Exception:
        _log.opt(exception=True).error("发生异常")
    finally:
        _log.bind(joint1=90, joint2=20, joint3=-80).info("复位")
        set_all_servo_positions(got, 90, 20, -80)

    _log.success("=" * 48)
    _log.success("追踪 + 抓取结束")
    _log.success("=" * 48)

    if not headless:
        _log.info("按 Ctrl+C 退出")
        try:
            while True:
                cv2.waitKey(100)
        except KeyboardInterrupt:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
