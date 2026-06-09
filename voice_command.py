import re
import time
import sys

from ugot import ugot

from utils import ROBOT_IP, wait_port
from logger import get_logger

_log = get_logger()

COLOR_MAP = {
    "红色": "red",
    "绿色": "green",
    "蓝色": "blue",
}

_SHORT_COLORS = ["红", "绿", "蓝"]


def parse_command(text: str) -> dict:
    color = None
    zone = None

    for cn_name, en_name in COLOR_MAP.items():
        if cn_name in text:
            color = en_name
            break
    if color is None:
        for short_cn, en_name in zip(_SHORT_COLORS, ["red", "green", "blue"]):
            if short_cn in text:
                color = en_name
                break

    zone_match = re.search(r'[ABab]', text)
    if zone_match:
        zone = zone_match.group().upper()

    return {"color": color, "zone": zone}


def asr_test(robot):
    _log.info("ASR 诊断测试（等待 10 秒不说话，观察原始响应）...")
    robot.play_sound("received", wait=True)
    time.sleep(0.5)
    _log.info("开始静音测试...")
    response = robot.AUDIO.setAudioAsr(begin_vad=1000, end_vad=1000, duration=10000)
    _log.bind(code=response.code, msg=response.msg, data=response.data).critical("静音测试 ASR 响应")

    _log.info("现在请说话，测试语音识别...")
    robot.play_sound("received", wait=True)
    time.sleep(0.5)
    response = robot.AUDIO.setAudioAsr(begin_vad=1500, end_vad=1000, duration=15000)
    _log.bind(code=response.code, msg=response.msg, data=response.data).critical("语音测试 ASR 响应")

    _log.info("使用 start_audio_asr_doa 测试...")
    result = robot.start_audio_asr_doa(duration=15)
    _log.bind(result=result).critical("ASR+DOA 结果")


def main():
    asr_test_mode = "--asr-test" in sys.argv
    if asr_test_mode:
        sys.argv.remove("--asr-test")

    _log.success("UGOT 语音指令识别")

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
            _log.bind(attempt=attempt + 1, max_attempts=3).opt(exception=True).warning("初始化尝试失败")
            if attempt < 2:
                time.sleep(2)
    else:
        _log.bind(attempts=3, ip=ip).error("连续 3 次初始化失败，退出")
        return

    if asr_test_mode:
        asr_test(robot)
        return

    robot.set_volume(80)
    _log.bind(volume=80).info("音量已设置")
    time.sleep(0.5)

    _log.info("请在听到提示音后说出指令")
    _log.bind(format='请搬运 X 色块，运输至 Y 号存储区').info("指令格式")

    robot.play_sound("received", wait=True)
    time.sleep(0.5)

    _log.info("正在监听语音...")
    try:
        response = robot.AUDIO.setAudioAsr(duration=20000)
        _log.bind(code=response.code, msg=response.msg, data=response.data).info("ASR 原始响应")
        result = response.data.strip() if response.code == 0 and response.data else ""
    except Exception:
        _log.opt(exception=True).error("语音识别异常")
        return

    if not result:
        _log.warning("未识别到语音内容")
        robot.play_audio_tts("未识别到语音，请重试", 0, wait=True)
        return

    _log.bind(raw_text=result).success("语音识别结果")

    parsed = parse_command(result)
    _log.bind(color=parsed["color"], zone=parsed["zone"]).info("解析结果")

    if not parsed["color"]:
        _log.warning("未识别到目标颜色")
    if not parsed["zone"]:
        _log.warning("未识别到目标存储区")

    if parsed["color"] and parsed["zone"]:
        confirm_text = f"收到指令，搬运{parsed["color"]}色块到{parsed["zone"]}区"
        _log.success(confirm_text)
        robot.play_audio_tts(confirm_text, 0, wait=True)

    print(f"\n=== 语音指令解析结果 ===")
    print(f"原始指令: {result}")
    print(f"目标颜色: {parsed['color']}")
    print(f"目标存储区: {parsed['zone']}")
    print("=======================\n")


if __name__ == "__main__":
    main()
