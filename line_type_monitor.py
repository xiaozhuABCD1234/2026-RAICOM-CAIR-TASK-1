from ugot import ugot
import re
import time
from common import ROBOT_IP, wait_port
from logger import get_logger

_log = get_logger()
robot = ugot.UGOT()

ip = ROBOT_IP
if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", ip):
    _log.bind(ip=ip).error("无效的 IP 地址")
    exit(1)

if not wait_port(ip, 50051, timeout=15):
    _log.bind(ip=ip, port=50051).error("端口不可达")
    exit(1)

robot.initialize(device_ip=ip)
robot.load_models(["line_recognition"])
robot.set_track_recognition_line(0)
_log.success("模型加载完成")

time.sleep(1)

try:
    while True:
        info = robot.get_single_track_total_info()
        offset, line_type, x, y = info
        _log.bind(line_type=line_type, offset=offset, x=x, y=y).info("当前轨道信息")
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    robot.stop_chassis()
