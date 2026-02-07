"""
使用 D435 相机检测 ArUco 标记并可视化 Optical / ROS 坐标系下的 6D 位姿
"""

import cv2
import numpy as np
from cv2 import aruco
from realsense.d435 import D435
from aruco.aruco_detector import ArucoDetector


def main():
    # 1. 初始化 D435 相机
    cam = D435()
    cam.start()

    # 2. 初始化 ArUco 检测器 (Dict4x4, 50mm, ID=0)
    detector = ArucoDetector(
        marker_size=0.05,
        target_id=0,
        dictionary=aruco.DICT_4X4_50,
    )
    detector.set_camera_intrinsics_from_d435(cam)

    print("ArUco detection started. Press 'q' to quit.")

    try:
        while True:
            rgb, depth = cam.get_frame(aligned=True)
            if rgb is None:
                continue

            # 3. 检测并估计位姿
            result = detector.estimate_pose(rgb)

            # 4. 可视化
            vis = detector.draw_detections(rgb, result)

            if result is not None:
                pos_opt = result["optical"]["position"]
                quat_opt = result["optical"]["quaternion"]
                pos_ros = result["ros"]["position"]
                quat_ros = result["ros"]["quaternion"]

                print(
                    f"[Optical] pos: ({pos_opt[0]:+.4f}, {pos_opt[1]:+.4f}, {pos_opt[2]:+.4f})  "
                    f"quat: ({quat_opt[0]:+.4f}, {quat_opt[1]:+.4f}, {quat_opt[2]:+.4f}, {quat_opt[3]:+.4f})"
                )
                print(
                    f"[ROS]     pos: ({pos_ros[0]:+.4f}, {pos_ros[1]:+.4f}, {pos_ros[2]:+.4f})  "
                    f"quat: ({quat_ros[0]:+.4f}, {quat_ros[1]:+.4f}, {quat_ros[2]:+.4f}, {quat_ros[3]:+.4f})"
                )

            cv2.imshow("D435 ArUco Detection", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
