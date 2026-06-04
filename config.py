import tomllib
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.toml"

with open(_CONFIG_PATH, "rb") as _f:
    _data = tomllib.load(_f)

ROBOT_IP = _data["network"]["robot_ip"]
CONSOLE_LEVEL = _data["logging"]["console_level"]
