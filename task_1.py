# ============================================================
# task_1.py — 集成任务：语音指令 → 巡线 → YOLO 追踪 → 夹方块
# ============================================================

import re
import sys
import time
import socket
import threading
import tomllib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ugot import ugot
from ultralytics import YOLO
from loguru import logger as _core_logger

# ============================================================
# 配置 & 日志（内联 config.py / logger.py）
# ============================================================

_CONFIG_PATH = Path(__file__).parent / "config.toml"
with open(_CONFIG_PATH, "rb") as _f:
    _data = tomllib.load(_f)
ROBOT_IP = _data["network"]["robot_ip"]
CONSOLE_LEVEL = _data["logging"]["console_level"]

_logger_configured = False


def get_logger(script_name=None):
    global _logger_configured
    if not _logger_configured:
        _core_logger.remove()
        Path("logs").mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _core_logger.add(
            sys.stderr,
            format=lambda r: (
                "<green>{time:HH:mm:ss.SSS}</green> | "
                "<level>{level.name: <8}</level> | "
                "<level>{message}</level>"
                + (
                    " | "
                    + " ".join(
                        f"<cyan>{k}</cyan>=<level>{str(v).replace('{', '{{').replace('}', '}}')}</level>"
                        for k, v in r["extra"].items()
                    )
                    if r["extra"]
                    else ""
                )
            )
            + "\n",
            level=CONSOLE_LEVEL,
            colorize=True,
        )
        _core_logger.add(
            str(Path("logs") / f"{ts}.log"),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS!UTC} | {level.name} | {message}",
            level="TRACE",
            serialize=True,
            rotation="10 MB",
            retention="7 days",
        )
        _logger_configured = True
    return _core_logger.bind(script=script_name) if script_name else _core_logger


_log = get_logger("task_1")

# ============================================================
# 常量
# ============================================================

# ── 语音指令 (voice_command.py) ──
COLOR_MAP = {"红色": "red", "绿色": "green", "蓝色": "blue"}
_SHORT_COLORS = ["红", "绿", "蓝"]

# ── 巡线 (main.py) ──
LINE_KP, LINE_KI, LINE_KD = 0.23, 0, 0
LINE_SPEED = 20

# ── 卸货区导航 (goto_zone.py) ──
STOP_DISTANCE = 8  # 巡线时到达目的地距离 cm
CROSS_FORWARD_SHORT = 16  # 路口短前进距离 cm
CROSS_FORWARD_LONG = 22  # 路口长前进距离 cm
TURN_SPEED = 40  # 路口转弯速度
TURN_ANGLE = 90  # 路口转弯角度
CROSS_CONFIRM_FRAMES = 2  # 路口防抖确认帧数

# ── AprilTag 追踪 (goto_zone.py) ──
APRILTAG_SEARCH_SPEED = 15  # 搜索旋转速度
APRILTAG_KP, APRILTAG_KI, APRILTAG_KD = 0.12, 0, 0.10  # 追踪 PID
APRILTAG_CHASE_SPEED = 20  # 追踪前进速度 cm/s
APRILTAG_TURN_SPEED_MAX = 30  # 追踪转弯速度上限
APRILTAG_TARGET_DISTANCE = 9.3  # 追踪目标距离 cm
APRILTAG_STOP_DISTANCE = 9.5  # 追踪停止距离 cm
APRILTAG_SLOW_DISTANCE = 15  # 追踪减速距离 cm
TARGET_TAG_ID = 0  # 目标 AprilTag ID

# ── 舵机 (control_servo.py) ──
SERVO_IDS = [51, 52, 53]
DEFAULT_DURATION = 800

# ── PT 追踪 (pt_cube_chase.py) ──
SEARCH_SPEED = 30
TRACK_KP, TRACK_KI, TRACK_KD = 0.25, 0, 0.05
CHASE_SPEED = 15
DISTANCE_KP, DISTANCE_KI, DISTANCE_KD = 2.0, 0, 0.1
BACKWARD_SPEED = 7

# ── 抓取稳定判定 (track_and_grab.py) ──
GRAB_DISTANCE_THRESHOLD = 9.3
GRAB_DISTANCE_TOLERANCE = 0.5
GRAB_OFFSET_THRESHOLD = 20
GRAB_STABLE_FRAMES = 10

# ── YOLO (pt_cube_detector.py) ──
CLASS_NAMES = ["red", "green", "blue"]
MODEL_PATH = "best.pt"
YOLO_CONF = 0.5
YOLO_IMGSZ = 640

SEP2 = "─" * 10

# ============================================================
# 工具函数
# ============================================================


def wait_port(ip, port, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection((ip, port), timeout=2)
            s.close()
            return True
        except OSError:
            time.sleep(1)
    return False


def parse_command(text):
    color = None
    for cn, en in COLOR_MAP.items():
        if cn in text:
            color = en
            break
    if color is None:
        for sc, en in zip(_SHORT_COLORS, ["red", "green", "blue"]):
            if sc in text:
                color = en
                break
    zone = None
    m = re.search(r"[ABab]", text)
    if m:
        zone = m.group().upper()
    return color, zone


def set_servo_position(got, servo_id, angle, duration_ms=DEFAULT_DURATION, wait=True):
    _log.bind(servo_id=servo_id, angle=angle, duration_ms=duration_ms).debug("舵机移动")
    got.turn_servo_angle(servo_id, angle, duration_ms, wait=wait)
    if wait:
        time.sleep(duration_ms / 1000.0 + 0.1)


def set_all_servo_positions(got, a1, a2, a3, duration_ms=DEFAULT_DURATION, wait=True):
    angles = [a1, a2, a3]
    _log.bind(joints=dict(zip(SERVO_IDS, angles)), duration_ms=duration_ms).debug(
        "多舵机移动"
    )
    for sid, ang in zip(SERVO_IDS, angles):
        got.turn_servo_angle(sid, ang, duration_ms, wait=False)
    if wait:
        time.sleep(duration_ms / 1000.0 + 0.1)


def _discover_infrared_id(robot):
    devices = robot.get_peripheral_devices_list()
    for dev in devices:
        if dev.get("type") == "Infrared":
            return int(dev.get("deviceId"))
    _log.warning("未发现红外传感器，使用默认 ID 41")
    return 41


# ============================================================
# 阶段 1 — 语音指令
# ============================================================


def voice_command_phase(robot):
    _log.info("正在监听语音...")
    try:
        resp = robot.AUDIO.setAudioAsr(duration=5000)
        _log.bind(code=resp.code, msg=resp.msg, data=resp.data).info("ASR 原始响应")
        text = resp.data.strip() if resp.code == 0 and resp.data else ""
    except Exception:
        _log.opt(exception=True).error("语音识别异常")
        return None, None

    if not text:
        _log.warning("未识别到语音")
        robot.play_audio_tts("未识别到语音，请重试", 0, wait=True)
        return None, None

    _log.bind(raw=text).success("语音识别结果")
    color, zone = parse_command(text)
    _log.bind(color=color, zone=zone).info("解析结果")
    return color, zone


# ============================================================
# 阶段 2 — PID 巡线
# ============================================================


def line_follow_phase(robot):
    robot.load_models(["line_recognition"])
    robot.set_track_recognition_line(0)
    _log.bind(model="line_recognition", mode=0).success("车道线模型加载完成")

    _log.info(SEP2)
    _log.bind(kp=LINE_KP, ki=LINE_KI, kd=LINE_KD).info("PID 参数")
    _log.bind(speed=LINE_SPEED).info("巡线速度")
    _log.info(SEP2)

    time.sleep(2)

    pid = robot.create_pid_controller()
    pid.set_pid(LINE_KP, LINE_KI, LINE_KD)

    was_lost = True
    cross_count = 0
    last_is_cross = False
    cross_stable_frames = 0
    cooldown_until = 0

    _log.info("开始巡线")
    try:
        while True:
            info = robot.get_single_track_total_info()
            offset, line_type, _, _ = info

            is_cross = line_type == 2 or line_type == 3
            if is_cross:
                cross_stable_frames += 1
            else:
                cross_stable_frames = 0
                last_is_cross = False

            rising = (
                is_cross
                and cross_stable_frames >= CROSS_CONFIRM_FRAMES
                and not last_is_cross
            )

            if rising and cross_count < 3 and time.time() > cooldown_until:
                last_is_cross = True
                cross_count += 1
                robot.mecanum_move_speed_times(0, LINE_SPEED, 22, 1)
                time.sleep(0.8)
                _log.bind(cross_count=cross_count, line_type=line_type).info("路口右转")
                robot.mecanum_turn_speed_times(3, 40, 90, 2)
                time.sleep(1)
                cooldown_until = time.time() + 0.2
                if cross_count >= 3:
                    robot.stop_chassis()
                    _log.bind(cross=3).success("已完成 3 个路口，停止巡线")
                    _log.info("进入取货区")
                    robot.mecanum_move_speed_times(0, LINE_SPEED, 40, 1)
                    time.sleep(1.2)
                    robot.stop_chassis()
                    _log.success("已到达取货区")
                    return
                continue

            if line_type == 0:
                robot.mecanum_turn_speed(3, 30)
                if not was_lost:
                    _log.debug("丢失车道线")
                    was_lost = True
            else:
                dic = round(pid.update(offset))
                if dic >= 0:
                    robot.mecanum_move_turn(0, LINE_SPEED, 3, dic)
                else:
                    robot.mecanum_move_turn(0, LINE_SPEED, 2, -dic)
                if was_lost:
                    _log.debug("重新检测到车道线")
                was_lost = False

            time.sleep(0.05)

    finally:
        try:
            robot.stop_chassis()
        except Exception:
            pass


# ============================================================
# 阶段 3 — YOLO 追踪 + 稳定抓取 + 舵机夹方块
# ============================================================


def track_and_grab_phase(robot, color, sensor_id):
    _log.bind(color=color).info("开始 YOLO 追踪")

    state = {"offset": None, "area": 0, "found": False, "frame": None}
    lock = threading.Lock()
    stop_event = threading.Event()
    grab_event = threading.Event()

    pid_h = robot.create_pid_controller()
    pid_h.set_pid(TRACK_KP, TRACK_KI, TRACK_KD)
    _log.bind(pid="horizontal", kp=TRACK_KP, ki=TRACK_KI, kd=TRACK_KD).info("水平 PID")

    pid_dist = robot.create_pid_controller()
    pid_dist.set_pid(DISTANCE_KP, DISTANCE_KI, DISTANCE_KD)
    _log.bind(
        pid="distance",
        kp=DISTANCE_KP,
        ki=DISTANCE_KI,
        kd=DISTANCE_KD,
        target=GRAB_DISTANCE_THRESHOLD,
    ).info("距离 PID")

    # ── 控制线程 ──
    def control_loop():
        stable_counter = 0
        while not stop_event.is_set():
            distance = robot.read_distance_data(sensor_id)
            if distance <= 0:
                for _ in range(3):
                    stop_event.wait(0.05)
                    distance = robot.read_distance_data(sensor_id)
                    if distance > 0:
                        break
                else:
                    _log.bind(sensor=sensor_id, val=distance).critical(
                        "距离传感器连续无数据"
                    )
                    stop_event.set()
                    return

            dist_error = round(pid_dist.update(distance - GRAB_DISTANCE_THRESHOLD))

            with lock:
                found = state["found"]
                offset = state["offset"]

            if not found:
                stable_counter = 0
                robot.mecanum_move_turn(0, 0, 2, SEARCH_SPEED)
                _log.bind(state="searching", dist=distance).trace("搜索旋转")
            else:
                dic = round(pid_h.update(offset))

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
                    bwd = int(min(dist_error, BACKWARD_SPEED))
                    robot.mecanum_move_speed(1, bwd)
                    stable_counter = 0
                    _log.bind(state="backward", dist=distance, speed=bwd).trace("后退")
                    stop_event.wait(0.05)
                    continue
                else:
                    dist_forward = 0

                if dist_forward == 0:
                    if turn_speed < 3:
                        robot.stop_chassis()
                        if (
                            distance
                            <= GRAB_DISTANCE_THRESHOLD + GRAB_DISTANCE_TOLERANCE
                            and abs(offset) < GRAB_OFFSET_THRESHOLD
                        ):
                            stable_counter += 1
                            _log.bind(
                                state="idle",
                                dist=distance,
                                offset=offset,
                                stable=stable_counter,
                                need=GRAB_STABLE_FRAMES,
                            ).trace("待命就绪")
                            if stable_counter >= GRAB_STABLE_FRAMES:
                                _log.bind(dist=distance, offset=offset).success(
                                    "已对准，准备抓取"
                                )
                                grab_event.set()
                                stop_event.set()
                                return
                        else:
                            stable_counter = 0
                            _log.bind(state="idle", dist=distance, offset=offset).trace(
                                "待命"
                            )
                    elif dic < 0:
                        stable_counter = 0
                        robot.mecanum_move_turn(0, 0, 3, turn_speed)
                        _log.bind(
                            state="turn_right",
                            dist=distance,
                            offset=offset,
                            turn=turn_speed,
                        ).trace("右转")
                    else:
                        stable_counter = 0
                        robot.mecanum_move_turn(0, 0, 2, turn_speed)
                        _log.bind(
                            state="turn_left",
                            dist=distance,
                            offset=offset,
                            turn=turn_speed,
                        ).trace("左转")
                elif turn_speed < 3:
                    if (
                        distance <= GRAB_DISTANCE_THRESHOLD + GRAB_DISTANCE_TOLERANCE
                        and abs(offset) < GRAB_OFFSET_THRESHOLD
                    ):
                        stable_counter += 1
                        _log.bind(
                            state="forward_grab",
                            dist=distance,
                            offset=offset,
                            stable=stable_counter,
                            need=GRAB_STABLE_FRAMES,
                        ).trace("微调前进已就绪")
                        if stable_counter >= GRAB_STABLE_FRAMES:
                            _log.bind(dist=distance, offset=offset).success(
                                "已对准，准备抓取"
                            )
                            grab_event.set()
                            stop_event.set()
                            return
                    else:
                        stable_counter = 0
                    robot.mecanum_move_speed(0, dist_forward)
                    _log.bind(state="forward", dist=distance, speed=dist_forward).trace(
                        "前进"
                    )
                elif dic < 0:
                    stable_counter = 0
                    robot.mecanum_move_turn(0, dist_forward, 3, turn_speed)
                    _log.bind(
                        state="fwd_right",
                        dist=distance,
                        fwd=dist_forward,
                        turn=turn_speed,
                    ).trace("前进右转")
                else:
                    stable_counter = 0
                    robot.mecanum_move_turn(0, dist_forward, 2, turn_speed)
                    _log.bind(
                        state="fwd_left",
                        dist=distance,
                        fwd=dist_forward,
                        turn=turn_speed,
                    ).trace("前进左转")

            stop_event.wait(0.05)

    # ── 视觉线程 (YOLO) ──
    def vision_loop():
        try:
            model = YOLO(MODEL_PATH)
            _log.success("YOLO 模型加载成功")
        except Exception:
            _log.opt(exception=True).error("YOLO 加载失败")
            stop_event.set()
            return

        model(
            np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8),
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF,
            verbose=False,
        )
        _log.success("模型预热完成")

        while not stop_event.is_set():
            try:
                data = robot.read_camera_data()
            except Exception:
                stop_event.wait(0.01)
                continue
            if data is None:
                stop_event.wait(0.01)
                continue

            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            fh, fw = frame.shape[:2]
            cx = fw // 2

            results = model(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)

            cubes = []
            boxes = results[0].boxes
            if boxes is not None and boxes.xyxy is not None:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    cls_id = int(boxes.cls[i])
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else ""
                    if cls_name != color:
                        continue
                    x, y = int(round(x1)), int(round(y1))
                    bw, bh = int(round(x2 - x1)), int(round(y2 - y1))
                    if bw <= 0 or bh <= 0:
                        continue
                    cubes.append((x, y, bw, bh, bw * bh))

            largest = max(cubes, key=lambda c: c[4]) if cubes else None

            with lock:
                if largest is not None:
                    x, y, bw, bh, area = largest
                    state["offset"] = (x + bw // 2) - cx
                    state["area"] = area
                    state["found"] = True
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv2.line(frame, (cx, 0), (cx, fh), (255, 255, 0), 1)
                    cv2.putText(
                        frame,
                        color,
                        (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                else:
                    state["found"] = False
                state["frame"] = frame

    with lock:
        state["frame"] = np.zeros((480, 640, 3), dtype=np.uint8)

    ctrl_thread = threading.Thread(target=control_loop, daemon=True)
    vis_thread = threading.Thread(target=vision_loop, daemon=True)
    ctrl_thread.start()
    vis_thread.start()

    _log.info("进入追踪主循环")
    try:
        while (
            vis_thread.is_alive() and ctrl_thread.is_alive() and not stop_event.is_set()
        ):
            with lock:
                disp = state["frame"]
            if disp is not None:
                cv2.imshow(f"YOLO Track - {color}", disp)
                cv2.waitKey(50)
            else:
                stop_event.wait(0.05)
    finally:
        stop_event.set()
        try:
            robot.stop_chassis()
        except Exception:
            pass
        cv2.destroyAllWindows()

    if not grab_event.is_set():
        _log.info("未触发抓取")
        return False

    # ── 抓取序列 ──
    _log.info("执行抓取序列")
    _log.bind(j1=90, j2=90, j3=0).info("机械臂归位")
    set_all_servo_positions(robot, 90, 90, 0)
    time.sleep(0.5)

    _log.info("夹爪张开")
    robot.mechanical_clamp_release()
    time.sleep(0.3)

    _log.bind(servo=52, to=160).info("关节2 下压")
    set_servo_position(robot, 52, 160, duration_ms=2000)

    _log.info("夹爪闭合")
    robot.mechanical_clamp_close()
    time.sleep(0.5)

    _log.bind(j1=90, j2=20, j3=-80).info("抬起机械臂")
    set_all_servo_positions(robot, 90, 20, -80, duration_ms=1500)

    _log.success("抓取完成")
    return True


# ============================================================
# 阶段 4 — 从取货区导航到 A/B 卸货区
# ============================================================


def _get_target_tag(tags, target_id):
    """从 AprilTag 列表中查找目标 Tag"""
    for tag in tags:
        if tag[0] == target_id:
            return tag
    return None


def _chase_apriltag(robot, target_id, target_dist, sensor_id):
    """追踪 AprilTag 到达目标距离

    Returns: True 到达目标距离，False 被中断
    """
    _log.bind(tag_id=target_id, target_dist=target_dist).info("开始追踪 AprilTag")

    state = {"tag": None}
    lock = threading.Lock()
    stop_event = threading.Event()
    reached = False

    pid = robot.create_pid_controller()
    pid.set_pid(APRILTAG_KP, APRILTAG_KI, APRILTAG_KD)
    _log.bind(pid="horizontal", kp=APRILTAG_KP, ki=APRILTAG_KI, kd=APRILTAG_KD).info(
        "水平 PID 配置"
    )

    def control_loop():
        nonlocal reached
        while not stop_event.is_set():
            with lock:
                tag = state["tag"]

            if tag is None:
                robot.mecanum_move_xyz(0, 0, APRILTAG_SEARCH_SPEED)
                _log.bind(state="searching").trace("搜索旋转")
            else:
                _id, cx, cy = tag[:3]

                distance = robot.read_distance_data(sensor_id)
                if distance <= 0:
                    for _ in range(3):
                        stop_event.wait(0.05)
                        distance = robot.read_distance_data(sensor_id)
                        if distance > 0:
                            break
                    else:
                        _log.bind(sensor_id=sensor_id, value=distance).critical(
                            "距离传感器连续无数据"
                        )
                        stop_event.set()
                        return

                offset_px = cx - (640 // 2)
                dic = round(pid.update(offset_px))
                z_speed = dic

                if distance < APRILTAG_STOP_DISTANCE:
                    y_speed = 0
                elif distance < APRILTAG_SLOW_DISTANCE:
                    y_speed = int(
                        np.clip(
                            (distance - APRILTAG_STOP_DISTANCE) * 3,
                            5,
                            APRILTAG_CHASE_SPEED,
                        )
                    )
                else:
                    y_speed = APRILTAG_CHASE_SPEED

                if abs(y_speed) < 1 and abs(z_speed) < 3:
                    robot.stop_chassis()
                    _log.bind(
                        state="idle", distance=distance, offset_px=offset_px
                    ).trace("待命")
                    if distance < APRILTAG_STOP_DISTANCE:
                        reached = True
                        break
                else:
                    z_speed = int(
                        np.clip(
                            z_speed, -APRILTAG_TURN_SPEED_MAX, APRILTAG_TURN_SPEED_MAX
                        )
                    )
                    robot.mecanum_move_xyz(0, y_speed, z_speed)
                    _log.bind(
                        state="chase",
                        distance=distance,
                        offset_px=offset_px,
                        y_speed=y_speed,
                        z_speed=z_speed,
                    ).trace("追踪")

            stop_event.wait(0.05)

    def vision_loop():
        while not stop_event.is_set():
            try:
                tags = robot.get_apriltag_total_info()
            except Exception:
                _log.opt(exception=True).warning("AprilTag 推理异常")
                stop_event.wait(0.05)
                continue

            target = _get_target_tag(tags, target_id) if tags else None
            with lock:
                state["tag"] = target

    ctrl_thread = threading.Thread(target=control_loop, daemon=True)
    vis_thread = threading.Thread(target=vision_loop, daemon=True)

    try:
        ctrl_thread.start()
        vis_thread.start()
        while (
            vis_thread.is_alive() and ctrl_thread.is_alive() and not stop_event.is_set()
        ):
            stop_event.wait(0.05)
    finally:
        stop_event.set()
        try:
            robot.stop_chassis()
        except Exception:
            pass

    return reached


def unload_phase(robot, zone, sensor_id):
    """从取货区导航到 A/B 卸货区

    子阶段：
      4a: 追踪 AprilTag 到达卸货区附近
      4b: 巡线进入卸货区
    """
    _log.bind(zone=zone).info("开始导航到卸货区")

    # ── 子阶段 4a: 追踪 AprilTag ──
    _log.info("子阶段 4a: 追踪 AprilTag")
    robot.load_models(["apriltag_qrcode"])
    _log.success("AprilTag 模型加载完成")
    time.sleep(1)

    reached = _chase_apriltag(robot, TARGET_TAG_ID, APRILTAG_TARGET_DISTANCE, sensor_id)
    if not reached:
        _log.warning("未追踪到 AprilTag")
        return

    # 左转进入巡线区域
    _log.bind(action="turn_left", speed=TURN_SPEED, angle=TURN_ANGLE).debug(
        "左转进入巡线区域"
    )
    robot.mecanum_turn_speed_times(2, TURN_SPEED, TURN_ANGLE, 2)
    time.sleep(1)
    robot.stop_chassis()
    time.sleep(0.5)

    # ── 子阶段 4b: 巡线进入卸货区 ──
    _log.info("子阶段 4b: 巡线进入卸货区")
    gotoA = zone == "A"
    cross_count = 0

    # ── 加载车道线识别模型 ──
    robot.load_models(["line_recognition"])
    robot.set_track_recognition_line(0)
    _log.success("车道线模型加载完成")

    # ── 等待模型初始化 ──
    _log.info("等待模型就绪...")
    time.sleep(1)

    # ── 创建 PID 控制器 ──
    pid = robot.create_pid_controller()
    pid.set_pid(LINE_KP, LINE_KI, LINE_KD)

    # ── 模型预热 ──
    _log.info("预热模型...")
    for _ in range(30):
        robot.get_single_track_total_info()
        time.sleep(0.05)

    # ── 状态变量 ──
    was_lost = True
    last_is_cross = False
    cross_stable_frames = 0
    cooldown_until = 0

    # ── 巡线主循环 ──
    try:
        _log.info("开始巡线到卸货区")
        while True:
            info = robot.get_single_track_total_info()
            offset, line_type, _, _ = info
            distance = robot.read_distance_data(sensor_id)
            if distance <= 0:
                for _ in range(3):
                    time.sleep(0.05)
                    distance = robot.read_distance_data(sensor_id)
                    if distance > 0:
                        break

            # 到达目的地
            if distance > 0 and distance < STOP_DISTANCE:
                _log.info("到达目的地")
                break

            # 路口检测
            is_cross = line_type in (2, 3)
            if is_cross:
                cross_stable_frames += 1
            else:
                cross_stable_frames = 0
                last_is_cross = False

            rising = (
                is_cross
                and cross_stable_frames >= CROSS_CONFIRM_FRAMES
                and not last_is_cross
            )

            if rising and time.time() > cooldown_until:
                last_is_cross = True
                cross_count += 1
                _log.bind(cross_count=cross_count, zone=zone).debug("检测到路口")

                # 判断是否需要右转
                if gotoA:
                    should_turn = cross_count <= 2
                else:
                    should_turn = cross_count <= 3 and cross_count != 2

                # 判断是否到达目的地
                if gotoA:
                    arrived = cross_count > 2
                else:
                    arrived = cross_count > 3

                if arrived:
                    robot.mecanum_move_speed_times(
                        0, LINE_SPEED, CROSS_FORWARD_SHORT, 1
                    )
                    time.sleep(0.8)
                    _log.bind(zone=zone).success("已到达卸货区")
                    break
                elif should_turn:
                    robot.mecanum_move_speed_times(0, LINE_SPEED, CROSS_FORWARD_LONG, 1)
                    time.sleep(0.8)
                    _log.bind(
                        action="turn_right", speed=TURN_SPEED, angle=TURN_ANGLE
                    ).debug("右转")
                    robot.mecanum_turn_speed_times(3, TURN_SPEED, TURN_ANGLE, 2)
                    time.sleep(1)
                    cooldown_until = time.time() + 0.2

            # 丢线处理
            if line_type == 0:
                robot.mecanum_turn_speed(3, 30)
                if not was_lost:
                    _log.debug("丢失车道线")
                    was_lost = True
            # 正常跟线
            else:
                dic = round(pid.update(offset))
                if dic >= 0:
                    robot.mecanum_move_turn(0, LINE_SPEED, 3, dic)
                else:
                    robot.mecanum_move_turn(0, LINE_SPEED, 2, -dic)
                if was_lost:
                    _log.debug("重新检测到车道线")
                was_lost = False

            time.sleep(0.05)

    finally:
        _log.bind(action="clamp_release").info("夹手张开")
        robot.mechanical_clamp_release()
        try:
            robot.stop_chassis()
        except Exception:
            pass
        _log.success("卸货区导航结束")


# ============================================================
# 连接辅助
# ============================================================


def _connect_robot():
    robot = ugot.UGOT()

    ip = ROBOT_IP
    if ip:
        if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ip):
            _log.bind(ip=ip).error("无效 IP 地址")
            return None
        _log.bind(ip=ip, source="config").info("使用指定 IP")
    else:
        _log.info("扫描 UGOT 设备...")
        devices = robot.scan_device()
        if not devices:
            _log.error("未找到设备")
            return None
        name = list(devices.keys())[0]
        ip = list(devices.values())[0]
        _log.bind(device=name, ip=ip).info("发现设备")

    _log.bind(port=50051).info("检测端口...")
    if not wait_port(ip, 50051, timeout=15):
        _log.bind(ip=ip, port=50051).error("端口不可达")
        return None
    _log.success("端口连通")

    _log.info("初始化 SDK...")
    for attempt in range(3):
        try:
            robot.initialize(device_ip=ip)
            _log.success("初始化成功")
            return robot
        except Exception:
            _log.bind(attempt=attempt + 1).warning("初始化失败")
            if attempt < 2:
                time.sleep(2)
    _log.error("连续 3 次初始化失败")
    return None


# ============================================================
# Main
# ============================================================


def main():
    _log.success("=" * 48)
    _log.success("UGOT 集成任务：语音 → 巡线 → 追踪 → 夹方块 → 卸货")
    _log.success("=" * 48)

    robot = _connect_robot()
    if robot is None:
        return

    sensor_id = _discover_infrared_id(robot)
    _log.bind(sensor_id=sensor_id).info("红外传感器 ID")

    try:
        # ── 阶段 1: 语音指令 ──
        _log.info(SEP2)
        _log.info("阶段 1/4 — 语音指令")
        robot.set_volume(100)
        time.sleep(0.5)
        robot.play_audio_tts("请说出指令（格式：搬运 X 色块到 X 区）", 0, wait=True)
        color, zone = voice_command_phase(robot)
        if color is None:
            robot.play_audio_tts("未识别到目标颜色，请再说一次", 0, wait=True)
            color, _ = voice_command_phase(robot)
            if color is None:
                robot.play_audio_tts("仍未识别到目标颜色，程序退出", 0, wait=True)
                return

        # 如果未识别到区域，重试一次
        if zone is None:
            robot.play_audio_tts("未识别到目标区域，请再说一次", 0, wait=True)
            time.sleep(1)
            _, zone = voice_command_phase(robot)
            if zone is None:
                robot.play_audio_tts("仍未识别到区域，程序退出", 0, wait=True)
                return

        confirm = f"收到指令，搬运{['红色','绿色','蓝色'][['red','green','blue'].index(color)]}色块到{zone}区"
        robot.play_audio_tts(confirm, 0, wait=True)
        _log.success(confirm)
        time.sleep(1)

        # ── 阶段 2: 巡线导航 ──
        _log.info(SEP2)
        _log.info("阶段 2/4 — 巡线导航到取货区")
        line_follow_phase(robot)

        # ── 阶段 3: 追踪 + 抓取 ──
        _log.info(SEP2)
        _log.info("阶段 3/4 — YOLO 追踪 + 抓取")
        _log.bind(color=color).info("打开摄像头")
        robot.open_camera()
        time.sleep(1)

        ok = track_and_grab_phase(robot, color, sensor_id)
        if not ok:
            _log.warning("未完成抓取")
            robot.play_audio_tts("抓取失败，程序退出", 0, wait=True)
            return

        # ── 阶段 4: 导航到卸货区 ──
        _log.info(SEP2)
        _log.info("阶段 4/4 — 导航到卸货区")
        unload_phase(robot, zone, sensor_id)
        robot.play_audio_tts(f"已到达{zone}区，任务完成", 0, wait=True)
        _log.success(f"已到达{zone}区")

    except KeyboardInterrupt:
        _log.info("收到停止信号")
    except Exception:
        _log.opt(exception=True).error("运行异常")
    finally:
        try:
            robot.stop_chassis()
        except Exception:
            pass
        try:
            set_all_servo_positions(robot, 90, 20, -80)
        except Exception:
            pass
        _log.success("=" * 48)
        _log.success("任务结束")
        _log.success("=" * 48)


if __name__ == "__main__":
    main()
