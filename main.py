# 导入 ugot 库，用于控制 UGOT 机器人
from ugot import ugot

# 导入 re 模块，用于正则表达式匹配（校验 IP 地址）
import re

# 导入 time 模块，用于延时和超时控制
import time

# 从共享模块导入常量、工具函数
from common import ROBOT_IP, SEP, wait_port
from logger import get_logger

# 短分隔线字符 × 10
SEP2 = "─" * 10
# PID 控制的比例系数、积分系数、微分系数
KP, KI, KD = 0.23, 0, 0
# 巡线速度，单位 cm/s
SPEED = 30

_log = get_logger()


# def _align(robot):
#     for i in range(5):
#         off = robot.get_single_track_total_info()[0]
#         _log.bind(iteration=i, offset=off).debug("对齐迭代")
#         if abs(off) < 8:
#             _log.bind(final_offset=off, iterations=i + 1).debug("对齐完成")
#             break
#         robot.mecanum_turn_speed(3 if off > 0 else 2, 20)
#         time.sleep(0.1)
#     robot.mecanum_stop()


def line_follow(robot, pid, speed, offset):
    dic = round(pid.update(offset))
    if dic >= 0:
        robot.mecanum_move_turn(0, speed, 3, dic)
    else:
        robot.mecanum_move_turn(0, speed, 2, -dic)
    return dic


def turn_left(robot):
    _log.bind(action="turn_left", speed=40, angle=90).debug("左转")
    robot.mecanum_turn_speed_times(2, 40, 90, 2)
    time.sleep(1)
    # _align(robot)


def turn_right(robot):
    _log.bind(action="turn_right", speed=40, angle=90).debug("右转")
    robot.mecanum_turn_speed_times(3, 40, 90, 2)
    time.sleep(1)
    # _align(robot)


def lost_line(robot):
    robot.mecanum_turn_speed(3, 30)


# 主函数：初始化连接 → 加载模型 → PID 巡线循环
def main():
    _log.success(SEP)
    _log.success("UGOT 麦伦车 - PID 巡线程序")
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
        except Exception:
            _log.bind(attempt=attempt + 1, max_attempts=3).opt(exception=True).warning(
                "初始化尝试失败"
            )
            if attempt < 2:
                time.sleep(2)
    else:
        _log.bind(attempts=3, ip=ip).error("连续 3 次初始化失败，退出")
        return

    _log.bind(action="load_model").info("正在加载视觉模型...")
    robot.load_models(["line_recognition"])
    robot.set_track_recognition_line(0)
    _log.bind(model="line_recognition", mode=0).success("车道线识别模型加载完成")

    _log.info(SEP2)
    _log.bind(kp=KP, ki=KI, kd=KD).info("PID 参数")
    _log.bind(speed_cm_s=SPEED).info("巡线速度")
    _log.info(SEP2)

    time.sleep(2)

    pid = robot.create_pid_controller()
    pid.set_pid(KP, KI, KD)

    was_lost = True
    dic = 0
    cross_count = 0
    last_is_cross = False
    cooldown_until = 0

    _log.info("开始巡线主循环")
    try:
        while True:
            info = robot.get_single_track_total_info()
            offset, line_type, _, _ = info

            is_cross = line_type == 2 or line_type == 3
            rising = is_cross and not last_is_cross
            last_is_cross = is_cross

            if rising and cross_count < 3 and time.time() > cooldown_until:
                cross_count += 1
                robot.mecanum_move_speed(0, SPEED)
                time.sleep(1)

                _log.bind(cross_count=cross_count, line_type=line_type).info("路口右转")
                turn_right(robot)
                cooldown_until = time.time() + 0.2
                if cross_count >= 3:
                    robot.stop_chassis()
                    _log.bind(cross_count=3).success("已完成 3 个路口，停止巡线")
                    break
                continue

            if line_type == 0:
                lost_line(robot)
                dic = 0
                if not was_lost:
                    _log.debug("丢失车道线")
                    was_lost = True
            else:
                dic = line_follow(robot, pid, SPEED, offset)
                if was_lost:
                    _log.debug("重新检测到车道线")
                was_lost = False

                if dic > 0:
                    _log.bind(
                        offset=offset, dic=dic, direction="left", line_type=line_type
                    ).trace("巡线修正")
                elif dic < 0:
                    _log.bind(
                        offset=offset,
                        dic=abs(dic),
                        direction="right",
                        line_type=line_type,
                    ).trace("巡线修正")
                else:
                    _log.bind(
                        offset=offset, dic=0, direction="straight", line_type=line_type
                    ).trace("巡线修正")

            time.sleep(0.05)

    except KeyboardInterrupt:
        _log.info("收到停止信号")
    finally:
        _log.info("正在停止...")
        try:
            robot.stop_chassis()
            _log.success("已停止")
        except Exception:
            _log.warning("停止时连接已断开")
        _log.success(SEP)


if __name__ == "__main__":
    main()
