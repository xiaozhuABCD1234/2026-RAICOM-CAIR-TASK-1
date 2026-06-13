"""从取货区导航到 A/B 卸货区：追踪 AprilTag → 自动巡线

用法:
    python goto_zone.py                          # 默认追踪 id=0 的 tag，去 B 区
    python goto_zone.py --headless               # 无头模式，不显示画面
    python goto_zone.py --target-id 1            # 追踪 id=1 的 tag
    python goto_zone.py --target-dist 20         # 目标距离 20cm
    python goto_zone.py --goto-a                 # 去 A 区（默认去 B 区）
"""

import argparse
import re
import threading
import time

import cv2
import numpy as np
from ugot import ugot

from utils import ROBOT_IP, wait_port, discover_infrared_id
from logger import get_logger

# ── AprilTag 追踪参数 ──
SEARCH_SPEED = 15  # 搜索旋转速度（未检测到 tag 时的旋转速度）
KP, KI, KD = 0.12, 0, 0.10  # 追踪 PID：P=水平偏移修正，I=禁用，D=阻尼
CHASE_SPEED = 20  # 追踪前进速度 cm/s
TURN_SPEED_MAX = 30  # 追踪转弯速度上限
TARGET_DISTANCE = 9.3  # 目标距离 cm，到达此距离视为追踪完成
TARGET_TAG_ID = 0  # 默认追踪的 AprilTag ID

# ── 巡线参数 ──
LINE_KP, LINE_KI, LINE_KD = 0.23, 0, 0  # 巡线 PID（当前只有 P 有效）
LINE_SPEED = 20  # 巡线前进速度 cm/s

# ── 距离阈值 ──
STOP_DISTANCE = 8  # 巡线时到达目的地距离 cm
CHASE_STOP_DISTANCE = 9.5  # 追踪停止距离 cm
CHASE_SLOW_DISTANCE = 15  # 追踪减速距离 cm

# ── 路口动作参数 ──
CROSS_FORWARD_SHORT = 16  # 路口短前进距离 cm
CROSS_FORWARD_LONG = 22  # 路口长前进距离 cm
TURN_SPEED = 40  # 路口转弯速度
TURN_ANGLE = 90  # 路口转弯角度
CROSS_CONFIRM_FRAMES = 2  # 路口防抖确认帧数

_log = get_logger()


def get_target_tag(tags, target_id):
    for tag in tags:
        if tag[0] == target_id:
            return tag
    return None


def read_distance(got, sensor_id):
    """读取红外测距传感器数值"""
    return got.read_distance_data(sensor_id)


def chase(robot, headless, target_id, target_dist, sensor_id=41):
    _log.bind(
        tag_id=target_id,
        headless=headless,
        target_dist=target_dist,
    ).info("开始追踪 AprilTag")
    if headless:
        _log.info("无头模式，不显示画面")

    state = {"tag": None, "frame": None}
    lock = threading.Lock()
    stop_event = threading.Event()
    reached = False

    pid = robot.create_pid_controller()
    pid.set_pid(KP, KI, KD)
    _log.bind(pid="horizontal", kp=KP, ki=KI, kd=KD).info("水平 PID 配置")

    @_log.catch(reraise=False)
    def control_loop():
        nonlocal reached
        with _log.contextualize(thread="control"):
            while not stop_event.is_set():
                with lock:
                    tag = state["tag"]

                if tag is None:
                    robot.mecanum_move_xyz(0, 0, SEARCH_SPEED)
                    _log.bind(state="searching").trace("搜索旋转")
                else:
                    _id, cx, cy = tag[:3]
                    area = tag[5]

                    distance = read_distance(robot, sensor_id)
                    if distance <= 0:
                        _log.bind(sensor_id=sensor_id, value=distance).critical(
                            "距离传感器无数据"
                        )
                        stop_event.set()
                        return
                    offset_px = cx - (640 // 2)
                    dic = round(pid.update(offset_px))
                    z_speed = dic

                    if distance < CHASE_STOP_DISTANCE:
                        y_speed = 0
                    elif distance < CHASE_SLOW_DISTANCE:
                        y_speed = int(np.clip((distance - CHASE_STOP_DISTANCE) * 3, 5, CHASE_SPEED))
                    else:
                        y_speed = CHASE_SPEED

                    if abs(y_speed) < 1 and abs(z_speed) < 3:
                        robot.stop_chassis()
                        _log.bind(
                            state="idle",
                            distance=distance,
                            offset_px=offset_px,
                            area=area,
                        ).trace("待命")
                        if distance < CHASE_STOP_DISTANCE:
                            reached = True
                            break
                    else:
                        z_speed = int(np.clip(z_speed, -TURN_SPEED_MAX, TURN_SPEED_MAX))
                        robot.mecanum_move_xyz(0, y_speed, z_speed)
                        _log.bind(
                            state="chase",
                            distance=distance,
                            offset_px=offset_px,
                            y_speed=y_speed,
                            z_speed=z_speed,
                            area=area,
                        ).trace("追踪")

                stop_event.wait(0.05)

    @_log.catch(reraise=False)
    def vision_loop():
        with _log.contextualize(thread="vision"):
            while not stop_event.is_set():
                try:
                    tags = robot.get_apriltag_total_info()
                except Exception:
                    _log.opt(exception=True).warning("AprilTag 推理异常")
                    stop_event.wait(0.05)
                    continue

                target = get_target_tag(tags, target_id) if tags else None

                try:
                    data = robot.read_camera_data()
                except Exception:
                    _log.opt(exception=True).warning("摄像头读取异常")
                    stop_event.wait(0.01)
                    continue

                frame = None
                if data is not None:
                    frame = cv2.imdecode(
                        np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is None:
                        _log.warning("帧解码失败")

                if frame is not None:
                    frame_h, frame_w = frame.shape[:2]
                    center_x = frame_w // 2

                    if target is not None:
                        _id, cx, cy, th, tw, area = target[:6]
                        x1 = int(round(cx - tw / 2))
                        y1 = int(round(cy - th / 2))
                        x2 = int(round(cx + tw / 2))
                        y2 = int(round(cy + th / 2))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"ID:{_id}"
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
                        cv2.line(
                            frame, (center_x, 0), (center_x, frame_h), (255, 255, 0), 1
                        )

                    with lock:
                        state["frame"] = frame
                else:
                    with lock:
                        state["frame"] = None

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
            if not headless:
                with lock:
                    display_frame = state["frame"]
                if display_frame is not None:
                    cv2.imshow(f"Tag Chase - ID:{target_id}", display_frame)
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

    return reached


def line_follow(robot, gotoA: bool, sensor_id=41):
    """基础巡线：PID 跟线 + 丢线原地右旋搜索

    可调参数（文件顶部常量）：
      LINE_KP   = 0.23   P 增益，越大转向越激进，太大会震荡
      LINE_KI   = 0      I 增益（当前禁用），可消除稳态误差
      LINE_KD   = 0      D 增益（当前禁用），可抑制震荡
      LINE_SPEED = 30    巡线前进速度 cm/s，越大越快，越不容易转弯
    """
    cross_count: int = 0

    _log.bind(speed=LINE_SPEED, kp=LINE_KP, ki=LINE_KI, kd=LINE_KD).info("开始巡线")

    # ── 加载车道线识别模型 ──
    robot.load_models(["line_recognition"])
    robot.set_track_recognition_line(0)
    _log.success("车道线模型加载完成")

    # ── 等待模型初始化 ──
    # 模型刚加载后前几秒推理结果不可靠，需要等一下
    # 如果车还是会冲，可以试试加大到 2~3 秒
    _log.info("等待模型就绪...")
    time.sleep(1)

    # ── 创建 PID 控制器 ──
    # pid.update(offset) 接收像素偏移量，返回修正值
    #   offset > 0（车偏右）→ 输出正数（需要右转）
    #   offset < 0（车偏左）→ 输出负数（需要左转）
    pid = robot.create_pid_controller()
    pid.set_pid(LINE_KP, LINE_KI, LINE_KD)

    # ── 模型预热 ──
    # 刚加载的模型前几帧输出可能乱报（如误判路口）
    # 丢弃若干帧推理结果，等模型输出稳定
    # 帧数 × 0.05s = 预热时间，30帧 ≈ 1.5秒
    # 如果还有误报，可以加到 40~50 帧
    _log.info("预热模型...")
    for _ in range(30):
        robot.get_single_track_total_info()
        time.sleep(0.05)

    # ── 状态变量 ──
    # was_lost: 记录上一帧是否丢线，仅影响日志输出，不影响控制
    was_lost = True
    last_is_cross = False
    cross_stable_frames = 0
    cooldown_until = 0

    # ── 巡线主循环 ──
    try:
        _log.info("开始巡线")
        while True:
            # 获取车道线识别结果
            # 返回值: (offset, line_type, x, y)
            #   offset   : 像素偏移量，0=在中心，正=车偏右，负=车偏左
            #   line_type: 线型，0=丢线，其他=检测到线
            info = robot.get_single_track_total_info()
            offset, line_type, _, _ = info
            distance = read_distance(robot, sensor_id)
            if distance < STOP_DISTANCE:
                _log.info("到达目的地")
                break

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
                _log.bind(cross_count=cross_count, gotoA=gotoA).debug("检测到路口")

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
                    robot.mecanum_move_speed_times(0, LINE_SPEED, CROSS_FORWARD_SHORT, 1)
                    time.sleep(0.8)
                    _log.info("到达目的地")
                    break
                elif should_turn:
                    robot.mecanum_move_speed_times(0, LINE_SPEED, CROSS_FORWARD_LONG, 1)
                    time.sleep(0.8)
                    _log.bind(action="turn_right", speed=TURN_SPEED, angle=TURN_ANGLE).debug("右转")
                    robot.mecanum_turn_speed_times(3, TURN_SPEED, TURN_ANGLE, 2)
                    time.sleep(1)
                    cooldown_until = time.time() + 0.2
            # ── 丢线处理 ──
            # line_type == 0 表示摄像头没检测到车道线
            # 策略：原地右旋，等待摄像头重新捕获车道线
            # 速度 30 可以调：太慢找回线慢，太快可能转过头
            if line_type == 0:
                robot.mecanum_turn_speed(3, 30)  # direction=3 是右转
                if not was_lost:
                    _log.debug("丢失车道线")
                    was_lost = True

            # ── 正常跟线处理 ──
            else:
                # PID 计算修正值
                # dic > 0 → 右转修正, dic < 0 → 左转修正
                # 如果车偏了但不转弯，可能是符号反了，交换 2/3 即可
                dic = round(pid.update(offset))

                # mecanum_move_turn(前进方向, 速度, 转弯方向, 转弯量)
                #   前进方向: 0=前进
                #   速度: LINE_SPEED cm/s
                #   转弯方向: 2=左转, 3=右转
                #   转弯量: 0~100，越大转得越急
                if dic >= 0:
                    robot.mecanum_move_turn(0, LINE_SPEED, 3, dic)  # 右转修正
                else:
                    robot.mecanum_move_turn(0, LINE_SPEED, 2, -dic)  # 左转修正

                if was_lost:
                    _log.debug("重新检测到车道线")
                was_lost = False

            # ── 循环间隔 50ms（20Hz）──
            # 减小可以提高响应速度，太小会占 CPU
            time.sleep(0.05)

    # ── Ctrl+C 优雅退出 ──
    except KeyboardInterrupt:
        _log.info("巡线被中断")

    # ── 清理：无论正常退出还是中断，都停车 ──
    finally:
        try:
            robot.stop_chassis()
        except Exception:
            pass
        _log.success("巡线结束")


def main():
    parser = argparse.ArgumentParser(description="从取货区导航到 A/B 卸货区")
    parser.add_argument("--headless", action="store_true", help="无头模式，不显示画面")
    parser.add_argument(
        "--target-id",
        type=int,
        default=TARGET_TAG_ID,
        help=f"目标 Tag ID (默认 {TARGET_TAG_ID})",
    )
    parser.add_argument(
        "--target-dist",
        type=float,
        default=TARGET_DISTANCE,
        help=f"目标距离 cm (默认 {TARGET_DISTANCE})",
    )
    parser.add_argument(
        "--goto-a",
        action="store_true",
        help="去 A 区（默认去 B 区）",
    )
    args = parser.parse_args()

    _log.success("从取货区导航到卸货区")
    _log.bind(
        tag_id=args.target_id,
        target_dist=args.target_dist,
        target_zone="A" if args.goto_a else "B",
    ).info("配置参数")

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
            _log.bind(attempt=attempt + 1, max_attempts=3).opt(exception=True).warning(
                "初始化尝试失败"
            )
            if attempt < 2:
                time.sleep(2)
    else:
        _log.bind(attempts=3, ip=ip).error("连续 3 次初始化失败，退出")
        return

    _log.bind(action="open_camera").info("正在打开摄像头...")
    robot.open_camera()
    _log.success("摄像头已打开")
    time.sleep(1)

    sensor_id = discover_infrared_id(robot)
    _log.bind(sensor_id=sensor_id).info("红外传感器 ID")

    _log.bind(model="apriltag_qrcode").info("正在加载 AprilTag 模型...")
    robot.load_models(["apriltag_qrcode"])
    _log.success("模型加载成功")

    reached = chase(robot, args.headless, args.target_id, args.target_dist, sensor_id=sensor_id)
    _log.info("追踪结束")

    if reached:
        _log.bind(action="turn_left", speed=TURN_SPEED, angle=TURN_ANGLE).debug("左转")
        robot.mecanum_turn_speed_times(2, TURN_SPEED, TURN_ANGLE, 2)
        time.sleep(1)
        _log.success("已到达目标位置，开始巡线")
        robot.stop_chassis()
        time.sleep(0.5)
        line_follow(robot, args.goto_a, sensor_id=sensor_id)
    else:
        _log.info("未到达目标位置，跳过巡线")


if __name__ == "__main__":
    main()
