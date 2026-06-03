from main import wait_port, turn_right, ROBOT_IP
from ugot import ugot


def main():
    robot = ugot.UGOT()

    if not wait_port(ROBOT_IP, 50051, timeout=15):
        print("[ERROR] 端口不可达")
        return

    robot.initialize(device_ip=ROBOT_IP)
    print("[INFO] 初始化成功")

    robot.load_models(["line_recognition"])
    robot.set_track_recognition_line(0)
    print("[INFO] 模型加载完成")

    turn_right(robot)

    robot.stop_chassis()
    print("[INFO] 已停止")


if __name__ == "__main__":
    main()
