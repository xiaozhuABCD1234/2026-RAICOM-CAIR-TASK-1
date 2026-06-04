"""UGOT 项目统一日志配置模块。

使用方式:
    from logger import get_logger
    logger = get_logger()
    logger.info("消息")
    logger.success("成功")
    logger.bind(kp=0.23).info("PID 配置")
"""

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger as _core_logger

from config import CONSOLE_LEVEL


_configured = False
_console_handler_id: int | None = None
_file_handler_id: int | None = None


def setup_logging(script_name: str | None = None):
    """配置 loguru 双 sink 日志系统（进程级单次初始化）。

    - 控制台 sink   : level="SUCCESS" , 人类可读彩色格式
    - 文件 sink     : level="TRACE"   , JSON Lines , 10MB 轮转 , 7 天保留

    Args:
        script_name: 可选，绑定到 extra["script"] 字段
    Returns:
        配置好的 logger（若提供 script_name 则已 bind）
    """
    global _configured, _console_handler_id, _file_handler_id

    if _configured:
        if script_name:
            return _core_logger.bind(script=script_name)
        return _core_logger

    _core_logger.remove()

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{timestamp}.log"

    # ---- 控制台 sink：无图标彩色格式，只显示 SUCCESS 及以上 ----
    _console_handler_id = _core_logger.add(
        sys.stderr,
        format=lambda r: (
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level.name: <8}</level> | "
            "<level>{message}</level>"
            + (" | " + " ".join(f"<cyan>{k}</cyan>=<level>{v}</level>" for k, v in r["extra"].items()) if r["extra"] else "")
        ) + "\n",
        level=CONSOLE_LEVEL,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )

    # ---- 文件 sink：JSON Lines 结构化日志，记录全部级别 ----
    _file_handler_id = _core_logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS!UTC} | {level.name} | {name}:{function}:{line} | {message}",
        level="TRACE",
        serialize=True,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

    _configured = True
    _core_logger.bind(startup=timestamp).success("日志系统初始化完成")

    if script_name:
        return _core_logger.bind(script=script_name)
    return _core_logger


def get_logger(script_name: str | None = None):
    """获取已配置的 logger 实例。

    Args:
        script_name: 可选，绑定到 extra["script"] 字段，便于 JSON 日志溯源
    Returns:
        loguru.Logger 实例
    """
    return setup_logging(script_name)
