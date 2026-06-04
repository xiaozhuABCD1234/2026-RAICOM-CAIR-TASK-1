# 导入 cv2（OpenCV），用于图像处理
import cv2
# 导入 numpy，用于数组操作
import numpy as np
# 导入 ugot 库，用于控制 UGOT 机器人
from ugot import ugot
# 导入 re 模块，用于正则表达式匹配（校验 IP 地址）
import re
# 导入 time 模块，用于延时和超时控制
import time

# 从共享模块导入常量、工具函数和检测函数
from common import (ROBOT_IP, SEP, wait_port, detect_cubes)


def draw_results(frame, results, color):
    """在原图上绘制检测到的立方体边界框。"""
    output = frame.copy()
    for x, y, w, h, _ in results:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{color}"
        cv2.putText(output, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return output


# 入口：连接 UGOT 机器人，实时检测指定颜色的立方体
def main():
    import sys

    # 默认检测红色
    color = sys.argv[1] if len(sys.argv) > 1 else "red"

    # 打印程序标题
    print(SEP)
    print("  UGOT 立方体颜色识别")
    print(f"  检测颜色: {color}")
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

    # 打开机器人摄像头
    print("[INFO] 正在打开摄像头...")
    robot.open_camera()
    print("[INFO] 摄像头已打开")

    print("[INFO] 按 q 键退出")
    try:
        while True:
            # 从机器人摄像头读取一帧
            data = robot.read_camera_data()
            if data is None:
                print("[WARN] 摄像头读取帧失败")
                continue

            # 解码 JPEG 字节为 OpenCV BGR 图像
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                print("[WARN] 帧解码失败")
                continue

            # 检测指定颜色的立方体
            cubes = detect_cubes(frame, color)
            # 绘制检测结果
            output = draw_results(frame, cubes, color)

            # 打印检测到的数量
            print(f"\r检测到 {len(cubes)} 个 {color} 立方体  ", end="", flush=True)

            # 显示画面
            cv2.imshow("UGOT Cube Detector", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # 捕获 Ctrl+C 键盘中断
    except KeyboardInterrupt:
        print("\n[INFO] 收到停止信号")
    # 无论是否异常，最终都要释放资源
    finally:
        # 打印正在停止
        print("\n[INFO] 正在停止...")
        # 尝试停止底盘
        try:
            robot.stop_chassis()
            print("[INFO] 已停止")
        except Exception:
            print("[WARN] 停止时连接已断开")
        cv2.destroyAllWindows()
        print(SEP)


if __name__ == "__main__":
    main()
