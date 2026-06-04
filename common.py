# 导入 cv2（OpenCV），用于图像处理
import cv2

# 导入 numpy，用于数组操作
import numpy as np

# 导入 time 模块，用于延时和超时控制
import time

# 导入 socket 模块，用于端口连通性检测
import socket

from config import ROBOT_IP

# 分隔线字符 × 54，用于终端输出美观

# 颜色名称到 HSV 范围的映射
# 格式: {颜色: [(H_min, S_min, V_min, H_max, S_max, V_max), ...]}
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
        list[tuple[int, int, int, int, float]]  每个元素为 (x, y, w, h, area)
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

        results.append((x, y, w, h, area))

    return results
