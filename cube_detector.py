# 导入 cv2（OpenCV），用于图像处理和颜色识别
import cv2
# 导入 numpy，用于数组操作和掩码计算
import numpy as np
# 导入 ugot 库，用于控制 UGOT 机器人
from ugot import ugot
# 导入 re 模块，用于正则表达式匹配（校验 IP 地址）
import re
# 导入 time 模块，用于延时和超时控制
import time
# 导入 socket 模块，用于端口连通性检测
import socket

# 目标机器人 IP 地址，留空则走自动扫描
ROBOT_IP = "192.168.1.22"
# 分隔线字符 × 54，用于终端输出美观
SEP = "─" * 54

# 颜色名称到 HSV 范围的映射
# 格式: {颜色: (H_min, S_min, V_min, H_max, S_max, V_max)}
# 红色在 HSV 中跨越 0 度边界，需要两段范围
COLOR_RANGES = {
    "red": [
        (0, 100, 100, 10, 255, 255),
        (170, 100, 100, 180, 255, 255),
    ],
    "green": [
        (35, 50, 50, 85, 255, 255),
    ],
    "blue": [
        (100, 50, 50, 130, 255, 255),
    ],
}

# 高斯模糊核大小（奇数），用于降噪
BLUR_KSIZE = (5, 5)
# 形态学操作的核大小
MORPH_KSIZE = (5, 5)
# 最小轮廓面积阈值，过滤噪声小色块
MIN_AREA = 500
# 立方体宽高比容差范围（正方形允许一定误差）
ASPECT_RATIO_MIN, ASPECT_RATIO_MAX = 0.75, 1.35
# 轮廓凸性阈值（凸包面积 / 轮廓面积），过滤不规则形状
CONVEXITY_MIN = 0.88


# 等待目标 IP 的指定端口就绪，超时则返回 False
def wait_port(ip, port, timeout=10):
    # 计算截止时间 = 当前时间 + 超时秒数
    deadline = time.time() + timeout
    # 循环检测直到超时
    while time.time() < deadline:
        # 尝试创建 TCP 连接
        try:
            # 创建到 (ip, port) 的 socket 连接，单次连接超时 2 秒
            s = socket.create_connection((ip, port), timeout=2)
            # 连接成功则立即关闭
            s.close()
            # 返回 True 表示端口可达
            return True
        # 任何 OSError（连接被拒、超时等）都算不可达
        except OSError:
            # 等待 1 秒后重试
            time.sleep(1)
    # 超时仍未成功，返回 False
    return False


def detect_cubes(frame, color):
    """
    检测图像中指定颜色的立方体。

    参数:
        frame:  BGR 格式的 numpy 数组（OpenCV 图像）
        color:  颜色字符串，可选 "red" / "green" / "blue"

    返回:
        list[tuple[int, int, int, int]]  每个元素为 (x, y, w, h) 边界框
    """
    # 校验颜色是否支持
    if color not in COLOR_RANGES:
        raise ValueError(f"不支持的颜色: {color}，可选: {list(COLOR_RANGES.keys())}")

    # 转换为 HSV 颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(hsv, BLUR_KSIZE, 0)

    # 生成该颜色的合并掩码（红色需要合并两段）
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for h_min, s_min, v_min, h_max, s_max, v_max in COLOR_RANGES[color]:
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask |= cv2.inRange(blurred, lower, upper)

    # 形态学操作：开运算去噪点，闭运算填充孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KSIZE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 过滤面积过小的区域
        if area < MIN_AREA:
            continue

        # 获取最小外接矩形
        x, y, w, h = cv2.boundingRect(cnt)
        # 计算宽高比并过滤（立方体正面投影应接近正方形）
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < ASPECT_RATIO_MIN or aspect_ratio > ASPECT_RATIO_MAX:
            continue

        # 凸性过滤：计算轮廓凸包面积比
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        if convexity < CONVEXITY_MIN:
            continue

        results.append((x, y, w, h))

    return results


def draw_results(frame, results, color):
    """在原图上绘制检测到的立方体边界框。"""
    output = frame.copy()
    for x, y, w, h in results:
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
