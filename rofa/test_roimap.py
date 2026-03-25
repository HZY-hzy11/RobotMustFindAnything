from pathlib import Path

import cv2
import numpy as np


if __name__ == "__main__":
    dataset_dir = Path(__file__).resolve().parent / "benchmark" / "office2"
    info_path = dataset_dir / "info.txt"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    info = {}
    if info_path.exists():
        for line in info_path.read_text(encoding="utf-8").splitlines():
            if "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", 1)]
            info[key] = value

    depth_shift = float(info.get("m_depthShift", "1000"))
    color_width = int(info.get("m_colorWidth", "640"))
    color_height = int(info.get("m_colorHeight", "480"))

    frame_ids = sorted(
        path.name.replace(".color.jpg", "")
        for path in dataset_dir.glob("*.color.jpg")
    )
    if not frame_ids:
        raise RuntimeError(f"No RGB frames found in {dataset_dir}")

    print(f"Dataset: {dataset_dir}")
    print(f"Frames: {len(frame_ids)}")
    print(f"Resolution: {color_width}x{color_height}")
    print(f"Depth shift: {depth_shift}")
    print("Controls: q=quit, space=pause/resume, a/left=prev, d/right=next")

    index = 0
    paused = False

    while True:
        frame_id = frame_ids[index]
        color_path = dataset_dir / f"{frame_id}.color.jpg"
        depth_path = dataset_dir / f"{frame_id}.depth.png"
        pose_path = dataset_dir / f"{frame_id}.pose.txt"

        color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        pose = np.loadtxt(str(pose_path), dtype=np.float32)

        if color is None:
            raise RuntimeError(f"Failed to read color image: {color_path}")
        if depth is None:
            raise RuntimeError(f"Failed to read depth image: {depth_path}")

        depth_m = depth.astype(np.float32) / depth_shift
        valid_depth = depth_m[depth_m > 0]
        max_depth = np.percentile(valid_depth, 99) if valid_depth.size else 1.0
        max_depth = max(max_depth, 1e-6)

        depth_vis = np.clip(depth_m / max_depth * 255.0, 0, 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        pose_text = f"t = [{pose[0, 3]:.3f}, {pose[1, 3]:.3f}, {pose[2, 3]:.3f}] m"
        stats_text = (
            f"{frame_id} | depth(min/max)= "
            f"{valid_depth.min():.3f}/{valid_depth.max():.3f} m"
            if valid_depth.size
            else f"{frame_id} | no valid depth"
        )

        cv2.putText(
            color,
            stats_text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            color,
            pose_text,
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            depth_vis,
            "Depth (colormap)",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        canvas = np.hstack([color, depth_vis])
        cv2.imshow("office2 PosedRGBD", canvas)

        key = cv2.waitKey(0 if paused else 30) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
            continue
        if key in (ord("a"), 81):
            index = (index - 1) % len(frame_ids)
            paused = True
            continue
        if key in (ord("d"), 83):
            index = (index + 1) % len(frame_ids)
            paused = True
            continue
        if not paused:
            index = (index + 1) % len(frame_ids)

    cv2.destroyAllWindows()
