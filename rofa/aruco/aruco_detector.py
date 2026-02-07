import cv2
import numpy as np
from cv2 import aruco
from scipy.spatial.transform import Rotation


class ArucoDetector:
    """
    ArUco 码检测器，基于 OpenCV ArUco 模块。
    检测指定 ID 的 ArUco 标记并估计其 6D 位姿。

    坐标系说明：
        - 光学坐标系 (Optical Frame):  X-右, Y-下, Z-前 （相机默认）
        - ROS 机器人坐标系 (ROS Frame): X-前, Y-左, Z-上
    """

    # 光学坐标系 -> ROS 坐标系的旋转矩阵 (固定变换)
    # optical: X-right, Y-down, Z-forward
    # ROS:     X-forward, Y-left, Z-up
    T_ROS_FROM_OPTICAL = np.array([
        [0,  0,  1,  0],
        [-1, 0,  0,  0],
        [0,  -1, 0,  0],
        [0,  0,  0,  1],
    ], dtype=np.float64)

    def __init__(self, marker_size=0.05, target_id=0, dictionary=aruco.DICT_4X4_50):
        """
        初始化 ArUco 检测器
        :param marker_size: Marker 边长，单位：米 (默认 0.05 = 50mm)
        :param target_id: 目标 Marker ID (默认 0)
        :param dictionary: ArUco 字典类型 (默认 DICT_4X4_50)
        """
        self.marker_size = marker_size
        self.target_id = target_id

        # 创建 ArUco 字典和检测参数
        self.aruco_dict = aruco.getPredefinedDictionary(dictionary)
        self.aruco_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # 相机内参 (需要在检测前设置)
        self.camera_matrix = None
        self.dist_coeffs = None

    def set_camera_intrinsics(self, fx, fy, cx, cy, dist_coeffs=None):
        """
        设置相机内参
        :param fx: 焦距 x
        :param fy: 焦距 y
        :param cx: 主点 x
        :param cy: 主点 y
        :param dist_coeffs: 畸变系数 (可选, 默认为零畸变)
        """
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1],
        ], dtype=np.float64)

        if dist_coeffs is not None:
            self.dist_coeffs = np.array(dist_coeffs, dtype=np.float64)
        else:
            self.dist_coeffs = np.zeros(5, dtype=np.float64)

    def set_camera_intrinsics_from_d435(self, d435_camera):
        """
        从 D435 相机实例自动获取内参并设置
        :param d435_camera: 已启动的 D435 实例
        """
        intrinsics = d435_camera.get_intrinsics()
        if intrinsics is None:
            raise RuntimeError("Failed to get intrinsics from D435. Is the camera started?")

        self.set_camera_intrinsics(
            fx=intrinsics["fx"],
            fy=intrinsics["fy"],
            cx=intrinsics["cx"],
            cy=intrinsics["cy"],
            dist_coeffs=intrinsics["distortion_coeffs"],
        )
        print(f"[ArucoDetector] Camera intrinsics set from D435: "
              f"fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}, "
              f"cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")

    def detect(self, color_image):
        """
        在彩色图像中检测所有 ArUco 标记
        :param color_image: BGR 格式的彩色图像 (numpy array)
        :return: (corners, ids) - corners 为角点列表, ids 为对应 ID 列表; 未检测到时 ids 为 None
        """
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        return corners, ids

    def estimate_pose(self, color_image):
        """
        检测目标 ArUco 标记并估计其 6D 位姿
        :param color_image: BGR 格式的彩色图像
        :return: dict 包含光学坐标系和 ROS 坐标系下的位姿，未检测到返回 None
                 {
                     "optical": {"rvec", "tvec", "rotation_matrix", "position", "quaternion"},
                     "ros":     {"rotation_matrix", "position", "quaternion"},
                     "corners": 目标 Marker 的角点坐标
                 }
        """
        if self.camera_matrix is None:
            raise RuntimeError("Camera intrinsics not set. Call set_camera_intrinsics() first.")

        corners, ids = self.detect(color_image)

        if ids is None:
            return None

        # 查找目标 ID
        ids_flat = ids.flatten()
        target_indices = np.where(ids_flat == self.target_id)[0]

        if len(target_indices) == 0:
            return None

        idx = target_indices[0]
        target_corners = corners[idx]

        # 使用 solvePnP 估计位姿
        # ArUco Marker 在自身坐标系下的 3D 角点 (中心为原点)
        half = self.marker_size / 2.0
        obj_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float64)

        image_points = target_corners.reshape(4, 2).astype(np.float64)

        success, rvec, tvec = cv2.solvePnP(
            obj_points, image_points,
            self.camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if not success:
            return None

        # --- 光学坐标系下的位姿 ---
        rotation_matrix_opt, _ = cv2.Rodrigues(rvec)
        position_opt = tvec.flatten()  # [x, y, z] 单位: 米
        quat_opt = Rotation.from_matrix(rotation_matrix_opt).as_quat()  # [x, y, z, w]

        # --- ROS 坐标系下的位姿 ---
        # 构造 4x4 齐次变换矩阵 (光学坐标系)
        T_optical = np.eye(4)
        T_optical[:3, :3] = rotation_matrix_opt
        T_optical[:3, 3] = position_opt

        # 转换到 ROS 坐标系
        T_ros = self.T_ROS_FROM_OPTICAL @ T_optical
        rotation_matrix_ros = T_ros[:3, :3]
        position_ros = T_ros[:3, 3]
        quat_ros = Rotation.from_matrix(rotation_matrix_ros).as_quat()  # [x, y, z, w]

        return {
            "optical": {
                "rvec": rvec.flatten(),
                "tvec": tvec.flatten(),
                "rotation_matrix": rotation_matrix_opt,
                "position": position_opt,
                "quaternion": quat_opt,  # [x, y, z, w]
            },
            "ros": {
                "rotation_matrix": rotation_matrix_ros,
                "position": position_ros,
                "quaternion": quat_ros,  # [x, y, z, w]
            },
            "corners": target_corners,
        }

    def draw_detections(self, color_image, pose_result=None):
        """
        在图像上绘制检测结果
        :param color_image: BGR 彩色图像 (会被修改)
        :param pose_result: estimate_pose() 的返回值，若提供则绘制坐标轴
        :return: 标注后的图像
        """
        output = color_image.copy()

        # 绘制所有检测到的 Marker
        corners, ids = self.detect(color_image)
        if ids is not None:
            aruco.drawDetectedMarkers(output, corners, ids)

        # 如果有位姿结果，绘制坐标轴
        if pose_result is not None:
            rvec = pose_result["optical"]["rvec"]
            tvec = pose_result["optical"]["tvec"]
            cv2.drawFrameAxes(
                output, self.camera_matrix, self.dist_coeffs,
                rvec, tvec, self.marker_size * 0.5,
            )

            # 在图像上显示位姿信息
            pos_opt = pose_result["optical"]["position"]
            pos_ros = pose_result["ros"]["position"]
            cv2.putText(output,
                        f"Optical: x={pos_opt[0]:.3f} y={pos_opt[1]:.3f} z={pos_opt[2]:.3f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(output,
                        f"ROS:     x={pos_ros[0]:.3f} y={pos_ros[1]:.3f} z={pos_ros[2]:.3f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return output


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from realsense.d435 import D435

    # 初始化相机
    cam = D435()
    cam.start()

    # 初始化 ArUco 检测器: Dict4x4, 50mm, ID=0
    detector = ArucoDetector(marker_size=0.05, target_id=0, dictionary=aruco.DICT_4X4_50)
    detector.set_camera_intrinsics_from_d435(cam)

    print("Press 'q' to quit...")

    try:
        while True:
            rgb, depth = cam.get_frame(aligned=True)
            if rgb is None:
                continue

            result = detector.estimate_pose(rgb)

            if result is not None:
                pos_opt = result["optical"]["position"]
                quat_opt = result["optical"]["quaternion"]
                pos_ros = result["ros"]["position"]
                quat_ros = result["ros"]["quaternion"]

                print(f"[Optical] pos: ({pos_opt[0]:.4f}, {pos_opt[1]:.4f}, {pos_opt[2]:.4f})  "
                      f"quat: ({quat_opt[0]:.4f}, {quat_opt[1]:.4f}, {quat_opt[2]:.4f}, {quat_opt[3]:.4f})")
                print(f"[ROS]     pos: ({pos_ros[0]:.4f}, {pos_ros[1]:.4f}, {pos_ros[2]:.4f})  "
                      f"quat: ({quat_ros[0]:.4f}, {quat_ros[1]:.4f}, {quat_ros[2]:.4f}, {quat_ros[3]:.4f})")

            vis = detector.draw_detections(rgb, result)
            cv2.imshow("ArUco Detection", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
