# 导入 ugot 库，用于控制 UGOT 机器人
from ugot import ugot

# 导入 re 模块，用于正则表达式匹配（校验 IP 地址）
import re

# 导入 time 模块，用于延时和超时控制
import time

# 从共享模块导入常量、工具函数
from common import ROBOT_IP, SEP, wait_port

# 短分隔线字符 × 10
SEP2 = "─" * 10
# PID 控制的比例系数、积分系数、微分系数
KP, KI, KD = 0.23, 0, 0
# 巡线速度，单位 cm/s
SPEED = 30


def _align(robot):
    for _ in range(5):
        off = robot.get_single_track_total_info()[0]
        if abs(off) < 8:
            break
        robot.mecanum_turn_speed(3 if off > 0 else 2, 20)
        time.sleep(0.1)
    robot.mecanum_stop()


def line_follow(robot, pid, speed, offset):
    dic = round(pid.update(offset))
    if dic >= 0:
        robot.mecanum_move_turn(0, speed, 3, dic)
    else:
        robot.mecanum_move_turn(0, speed, 2, -dic)
    return dic


def turn_left(robot):
    print("  <<< 左转 <<<")
    robot.mecanum_turn_speed_times(2, 40, 90, 2)
    time.sleep(1)
    _align(robot)


def turn_right(robot):
    print("  >>> 右转 >>>")
    robot.mecanum_turn_speed_times(3, 40, 90, 2)
    time.sleep(1)
    _align(robot)


def lost_line(robot):
    robot.mecanum_turn_speed(3, 30)


# 主函数：初始化连接 → 加载模型 → PID 巡线循环
def main():
    # 创建 UGOT 机器人对象实例
    robot = ugot.UGOT()

    # 打印程序标题
    print(SEP)
    # 打印程序名称
    print("  UGOT 麦伦车 - PID 巡线程序")
    # 打印分隔线
    print(SEP)

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
    # 等待端口就绪，最长 15 秒
    if not wait_port(ip, 50051, timeout=15):
        # 端口不可达，打印错误并退出
        print("[ERROR] 端口 50051 不可达")
        return
    # 端口连通，打印确认信息
    print(f"[INFO] 端口连通 → {ip}:50051")

    # 打印初始化提示
    print("[INFO] 正在初始化 SDK...")
    # 最多重试 3 次
    for attempt in range(3):
        # 尝试初始化连接
        try:
            # 以指定 IP 初始化 SDK
            robot.initialize(device_ip=ip)
            # 成功则打印并跳出循环
            print("[INFO] 初始化成功")
            break
        # 捕获所有异常
        except Exception as e:
            # 打印警告：第几次失败及原因
            print(f"[WARN] 第 {attempt + 1}/3 次尝试失败: {e}")
            # 如果不是最后一次尝试
            if attempt < 2:
                # 等待 2 秒后重试
                time.sleep(2)
    # for-else：如果循环未 break（3 次都失败）
    else:
        # 打印错误并退出
        print("[ERROR] 连续 3 次初始化失败，退出")
        return

    # 加载视觉模型
    print("[INFO] 正在加载视觉模型...")
    # 加载车道线识别模型
    robot.load_models(["line_recognition"])
    # 设置为单轨巡线模式（模式 0）
    robot.set_track_recognition_line(0)
    # 打印加载完成提示
    print("[INFO] 车道线识别模型加载完成（单轨模式）")

    # 打印短分隔线
    print(SEP2)
    # 打印当前 PID 参数
    print(f"PID 参数  │  kp={KP}  ki={KI}  kd={KD}")
    # 打印巡线速度
    print(f"巡线速度  │  {SPEED} cm/s")
    # 打印短分隔线
    print(SEP2)

    time.sleep(2)

    # 创建 PID 控制器实例
    pid = robot.create_pid_controller()
    # 设置 PID 控制器的 Kp, Ki, Kd 参数
    pid.set_pid(KP, KI, KD)

    # 标记是否之前丢失过线（用于状态切换打印）
    was_lost = True
    # 当前的方向修正值，初始为 0
    dic = 0
    # 已过的路口计数
    cross_count = 0
    # 上一帧是否为路口（用于上升沿检测）
    last_is_cross = False
    # 路口冷却时间截止点（防止一次路口被重复触发）
    cooldown_until = 0

    # 开始主巡线循环
    try:
        # 无限循环直至键盘中断
        while True:
            # 获取单轨车道线信息：偏移量、线类型等
            info = robot.get_single_track_total_info()
            # 解包：偏移量 offset，线类型 line_type，其余忽略
            offset, line_type, _, _ = info

            # 判断是否为路口（线类型 == 2 或 3）
            is_cross = line_type == 2 or line_type == 3
            # 上升沿检测：当前是路口且上一帧不是
            rising = is_cross and not last_is_cross
            # 更新上一帧状态为当前帧状态
            last_is_cross = is_cross

            # 如果检测到路口上升沿、路口数未满 3 且已过冷却时间
            if rising and cross_count < 3 and time.time() > cooldown_until:
                # 路口计数 +1
                cross_count += 1

                # 显式直行穿过路口（不偏转）
                robot.mecanum_move_speed(0, SPEED)
                time.sleep(1)

                # 执行右转
                print(f"\n  >>> 第 {cross_count} 个路口 → 右转 >>>")
                turn_right(robot)
                # 设置冷却时间：当前时间 + 0.3 秒（防止转弯后残影误触）
                cooldown_until = time.time() + 0.2
                # 如果已完成 3 个路口，停止机器人并退出循环
                if cross_count >= 3:
                    robot.stop_chassis()
                    print("[INFO] 已完成 3 个路口，停止巡线")
                    break
                # 跳过本次循环的巡线修正（右转后再继续）
                continue

            # 如果未检测到线（丢失车道线）
            if line_type == 0:
                lost_line(robot)
                # 方向修正值归零
                dic = 0
                # 如果之前未丢失，现在标记为丢失
                if not was_lost:
                    was_lost = True
            # 检测到线，正常巡线
            else:
                dic = line_follow(robot, pid, SPEED, offset)
                # 标记为未丢失
                was_lost = False
                # 根据方向修正值打印箭头指示
                if dic > 0:
                    # 正偏：打印 ← 及其大小
                    print(f"  {offset:+4d}  ←{dic}")
                elif dic < 0:
                    # 负偏：打印 → 及绝对值
                    print(f"  {offset:+4d}  →{abs(dic)}")
                else:
                    # 无偏：打印 ↑
                    print(f"  {offset:+4d}  ↑")

            # 主循环每帧间隔 0.05 秒（约 20 Hz）
            time.sleep(0.05)

    # 捕获 Ctrl+C 键盘中断
    except KeyboardInterrupt:
        # 打印停止提示
        print("[INFO] 收到停止信号")
    # 无论是否异常，最终都要停止机器人
    finally:
        # 打印正在停止
        print("[INFO] 正在停止...")
        # 尝试停止底盘
        try:
            # 调用 SDK 停止底盘
            robot.stop_chassis()
            # 打印已停止
            print("[INFO] 已停止")
        # 如果连接已断开导致异常
        except Exception:
            # 打印警告
            print("[WARN] 停止时连接已断开")
        # 打印结束分隔线
        print(SEP)


# Python 入口判断：当前文件被直接运行时执行 main()
if __name__ == "__main__":
    # 调用主函数
    main()
