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

# 追踪参数
SEARCH_SPEED = 15          # 搜索旋转速度
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


def get_largest_cube(cubes):
    """从检测结果中选取面积最大的立方体。"""
    # 无检测结果则返回 None
    if not cubes:
        return None
    # 按面积（c[4]）降序，取最大值
    return max(cubes, key=lambda c: c[4])


def chase(robot, color, headless=False):
    """追踪指定颜色的立方体，使机器人始终正对目标。"""
    # 打印追踪信息
    print(f"[追踪] 目标颜色: {color}")
    print("[INFO] 按 Ctrl+C 停止")
    if headless:
        print("[INFO] 无头模式，不显示画面")

    # 共享状态，由视觉线程写入、控制线程读取
    state = {"offset": None, "area": 0, "found": False, "frame": None}
    # 互斥锁，保护共享状态读写
    lock = threading.Lock()
    # 停止事件，通知各线程退出
    stop_event = threading.Event()

    # 创建水平 PID 控制器，用于平滑校正水平偏移
    pid = robot.create_pid_controller()
    pid.set_pid(KP, KI, KD)
    print(f"[INFO] 水平 PID  |  kp={KP}  ki={KI}  kd={KD}")

    # 创建距离 PID 控制器，用于精确维持目标距离
    pid_dist = robot.create_pid_controller()
    pid_dist.set_pid(DISTANCE_KP, DISTANCE_KI, DISTANCE_KD)
    print(f"[INFO] 距离 PID  |  kp={DISTANCE_KP}  ki={DISTANCE_KI}  kd={DISTANCE_KD}  目标={TARGET_DISTANCE}cm")

    # ===== 控制线程：50ms 周期发送电机指令 =====
    def control_loop():
        # 搜索方向：2=左转, 3=右转
        search_dir = 2
        # 上次切换搜索方向的时刻
        search_since = 0
        # 主循环：直至 stop_event 被设置
        while not stop_event.is_set():
            # ===== 距离传感器：PID 控制前进速度，精确维持目标距离 =====
            distance = robot.read_distance_data(SENSOR_ID)
            if distance <= 0:
                # 传感器无数据，报错退出
                print(f"\n[ERROR] 距离传感器无数据 (返回值={distance})")
                stop_event.set()
                return
            # pid_dist.update(distance - TARGET_DISTANCE)
            #   error = 0 - (distance - 10) = 10 - distance
            #   dist_error > 0 → 太近，dist_error < 0 → 太远
            dist_error = round(pid_dist.update(distance - TARGET_DISTANCE))

            # 加锁读取共享状态
            with lock:
                found = state["found"]
                offset = state["offset"]
                area = state["area"]

            if not found:
                # 未检测到目标，旋转搜索
                now = time.time()
                # 每 3 秒切换一次搜索方向（交替左右转）
                if now - search_since > 3:
                    search_dir = 3 if search_dir == 2 else 2
                    search_since = now
                # 发送旋转指令（固定速度搜索）
                robot.mecanum_move_turn(0, 0, search_dir, SEARCH_SPEED)
                print(f"\r[搜索] 未检测到 {color}，{'左转' if search_dir == 2 else '右转'}搜索...  ", end="", flush=True)
            else:
                # 记录当前时间，避免切回搜索时立即触发切换
                search_since = time.time()
                # 水平 PID 计算转向修正值
                # 注意：SDK PID 的 __SetPoint=0，update(offset) ≈ -Kp*offset
                # offset>0(目标偏右) → dic<0 → 应右转(3)
                # offset<0(目标偏左) → dic>0 → 应左转(2)
                dic = round(pid.update(offset))

                # 距离 PID 决定前进速度
                if dist_error < 0:
                    # 太远 → 前进，速度正比于误差
                    dist_forward = int(min(-dist_error, CHASE_SPEED))
                elif dist_error > 0:
                    # 太近 → 后退修正
                    backward = int(min(dist_error, BACKWARD_SPEED))
                    robot.mecanum_move_speed(1, backward)
                    print(f"\r[后退] {distance:.1f}cm  偏移={offset:+d}  PID={dic:+d}  面积={area:.0f}  ", end="", flush=True)
                    stop_event.wait(0.05)
                    continue
                else:
                    # 正好在目标距离
                    dist_forward = 0

                turn_speed = min(abs(dic), TURN_SPEED_MAX)

                if dist_forward == 0:
                    # 已在目标距离，仅转向对准
                    if turn_speed < 3:
                        robot.stop_chassis()
                        print(f"\r[待命] {distance:.1f}cm  偏移={offset:+d}  PID={dic:+d}  面积={area:.0f}  ", end="", flush=True)
                    elif dic < 0:
                        robot.mecanum_move_turn(0, 0, 3, turn_speed)
                        print(f"\r[右转] {distance:.1f}cm  偏移={offset:+d}  PID={dic:+d}  面积={area:.0f}  ", end="", flush=True)
                    else:
                        robot.mecanum_move_turn(0, 0, 2, turn_speed)
                        print(f"\r[左转] {distance:.1f}cm  偏移={offset:+d}  PID={dic:+d}  面积={area:.0f}  ", end="", flush=True)
                elif turn_speed < 3:
                    # PID 输出极小，纯前进避免微振荡
                    robot.mecanum_move_speed(0, dist_forward)
                    print(f"\r[前进] {distance:.1f}cm  偏移={offset:+d}  PID={dic:+d}  F={dist_forward}  面积={area:.0f}  ", end="", flush=True)
                elif dic < 0:
                    # offset>0(目标偏右) → dic<0 → 前进+右转
                    robot.mecanum_move_turn(0, dist_forward, 3, turn_speed)
                    print(f"\r[右转] {distance:.1f}cm  偏移={offset:+d}  PID={dic:+d}  F={dist_forward}  面积={area:.0f}  ", end="", flush=True)
                else:
                    # offset<0(目标偏左) → dic>0 → 前进+左转
                    robot.mecanum_move_turn(0, dist_forward, 2, turn_speed)
                    print(f"\r[左转] {distance:.1f}cm  偏移={offset:+d}  PID={dic:+d}  F={dist_forward}  面积={area:.0f}  ", end="", flush=True)

            # 控制周期 50ms（20Hz）
            stop_event.wait(0.05)

    # ===== 视觉线程：取帧 + 检测 =====
    def vision_loop():
        # 主循环：直至 stop_event 被设置
        while not stop_event.is_set():
            # 从机器人摄像头读取一帧数据（带异常保护，防止网络抖动杀死线程）
            try:
                data = robot.read_camera_data()
            except Exception:
                # 异常时短暂休眠，防止空转烧 CPU
                stop_event.wait(0.01)
                continue
            # 读取失败则跳过本帧
            if data is None:
                stop_event.wait(0.01)
                continue

            # 解码 JPEG 字节为 OpenCV BGR 图像
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            # 解码失败则跳过本帧
            if frame is None:
                continue

            # 获取图像尺寸并计算水平中心线
            frame_h, frame_w = frame.shape[:2]
            center_x = frame_w // 2

            # 检测指定颜色的立方体
            cubes = detect_cubes(frame, color)
            # 取面积最大的立方体作为追踪目标
            largest = get_largest_cube(cubes)

            # 加锁更新共享状态
            with lock:
                if largest is not None:
                    # 解包检测结果：边界框 + 面积
                    x, y, w, h, area = largest
                    # 计算目标中心相对于图像中心的水平偏移
                    state["offset"] = (x + w // 2) - center_x
                    state["area"] = area
                    state["found"] = True
                    # 绘制检测框（绿色矩形）
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # 绘制水平中心参考线（黄色）
                    cv2.line(frame, (center_x, 0), (center_x, frame_h), (255, 255, 0), 1)
                    state["frame"] = frame
                else:
                    # 未检测到目标
                    state["found"] = False
                    state["frame"] = frame

    # 创建双线程（守护线程，主线程退出时自动终止）
    ctrl_thread = threading.Thread(target=control_loop, daemon=True)
    vis_thread = threading.Thread(target=vision_loop, daemon=True)

    try:
        # 启动控制线程和视觉线程
        ctrl_thread.start()
        vis_thread.start()
        # 主线程负责显示画面（macOS OpenCV 要求 imshow 在主线程调用）
        # 同时监控两个子线程是否存活，任一死亡则退出
        while vis_thread.is_alive() and ctrl_thread.is_alive() and not stop_event.is_set():
            if not headless:
                # 加锁读取最新帧用于显示
                with lock:
                    display_frame = state["frame"]
                if display_frame is not None:
                    # 显示追踪画面窗口
                    cv2.imshow(f"Cube Chase - {color}", display_frame)
                    # 等待 50ms，检测 'q' 键按下则退出
                    if cv2.waitKey(50) & 0xFF == ord("q"):
                        stop_event.set()
                        break
            else:
                # 无头模式：仅等待 50ms 后继续循环
                stop_event.wait(0.05)
    except KeyboardInterrupt:
        # Ctrl+C 中断
        print("\n[INFO] 收到停止信号")
    finally:
        # 通知所有子线程停止
        stop_event.set()
        # 停止机器人底盘
        robot.stop_chassis()
        # 关闭所有 OpenCV 窗口
        if not headless:
            cv2.destroyAllWindows()
        print(f"\n[INFO] 已停止")


def main():
    """入口：解析命令行参数，连接机器人，启动追踪。"""
    import sys

    # 解析命令行参数
    args = sys.argv[1:]
    headless = "--headless" in args
    if headless:
        args.remove("--headless")

    # 目标颜色，默认红色
    color = args[0] if args else "red"
    # 校验颜色是否支持
    if color not in COLOR_RANGES:
        print(f"[ERROR] 不支持的颜色: {color}，可选: {' / '.join(COLOR_RANGES.keys())}")
        return

    # 打印程序标题
    print(SEP)
    print(f"  UGOT 方块追踪 - {color}")
    print(SEP)

    # 创建 UGOT 机器人对象实例
    robot = ugot.UGOT()

    # 使用全局变量 ROBOT_IP
    ip = ROBOT_IP
    # 如果指定了 IP
    if ip:
        # 用正则校验 IP 格式是否为 x.x.x.x
        if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ip):
            # 格式不合法，打印错误并退出
            print(f"[ERROR] 无效的 IP 地址: {ip}")
            return
        # 格式合法，打印使用的 IP
        print(f"[INFO] 使用指定 IP: {ip}")
    # 否则走自动扫描
    else:
        # 提示正在扫描
        print("[INFO] 正在扫描局域网中的 UGOT 设备...")
        # 调用 SDK 扫描设备
        devices = robot.scan_device()
        # 如果没找到任何设备
        if not devices:
            # 打印错误并退出
            print("[ERROR] 未找到任何 UGOT 设备")
            return
        # 取第一个扫描到的设备，打印名称和 IP
        print(f"  {list(devices.items())[0][0]} → {list(devices.values())[0]}")
        # 将第一个设备的 IP 赋值给 ip
        ip = list(devices.values())[0]

    # 检测机器人 50051 端口是否可达
    print("[INFO] 正在检测机器人端口 50051...")
    if not wait_port(ip, 50051, timeout=15):
        print("[ERROR] 端口 50051 不可达")
        return
    print(f"[INFO] 端口连通 → {ip}:50051")

    # 打印初始化提示
    print("[INFO] 正在初始化 SDK...")
    # 最多重试 3 次
    for attempt in range(3):
        try:
            # 以指定 IP 初始化 SDK
            robot.initialize(device_ip=ip)
            print("[INFO] 初始化成功")
            break
        except Exception as e:
            print(f"[WARN] 第 {attempt + 1}/3 次尝试失败: {e}")
            if attempt < 2:
                # 等待 2 秒后重试
                time.sleep(2)
    # for-else：3 次都失败则退出
    else:
        print("[ERROR] 连续 3 次初始化失败，退出")
        return

    # 打开机器人摄像头
    print("[INFO] 正在打开摄像头...")
    robot.open_camera()
    print("[INFO] 摄像头已打开")
    # 等待摄像头初始化完成
    time.sleep(1)

    # 进入追踪主循环
    chase(robot, color, headless=headless)


if __name__ == "__main__":
    main()
