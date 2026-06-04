"""随机选 5 个视频，输出人脸检测 crops 视频，检查嘴唇信息是否保留。"""

import os, sys, random, time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from face_detector import VisualPreprocessor


def main():
    search_dirs = [
        "./测试用例/特殊人脸切换示例",
        "./测试用例/视频",
    ]
    all_videos = []
    for d in search_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".mp4"):
                    all_videos.append(os.path.join(d, f))

    print(f"共找到 {len(all_videos)} 个视频")
    random.seed(42)
    picked = sorted(random.sample(all_videos, min(5, len(all_videos))))
    for v in picked:
        print(f"  {v}")

    out_dir = "./测试结果_crops检查"
    os.makedirs(out_dir, exist_ok=True)

    for video_path in picked:
        base = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 1e-3:
            fps = 24.0

        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        print(f"\n处理: {base}  ({len(frames)} 帧, fps={fps})")

        # 每个视频新建 detector，确保 tracker 状态干净
        detector = VisualPreprocessor(
            crop_size=128,
            model_path="blaze_face_full_range.tflite",
            detect_every_n=5,
            score_smooth_alpha=0.8,
            box_smooth_alpha=0,
            min_detection_confidence=0.5,
            use_lip_center_crop=True,
        )
        t0 = time.perf_counter()
        r = detector.process_chunk(frames)
        dt = time.perf_counter() - t0

        n_face = sum(1 for b in r["face_boxes"] if b is not None)
        print(f"  耗时 {dt:.2f}s, 检测到人脸 {n_face}/{len(frames)} 帧")

        crops = r["crops"]
        if not crops:
            print(f"  无 crops，跳过")
            continue

        h, w = crops[0].shape[:2]
        out_path = os.path.join(out_dir, f"{base}_crops.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for crop in crops:
            # crop is float32 RGB [0,255], convert to uint8 BGR for writing
            frame_bgr = np.clip(crop, 0, 255).astype(np.uint8)[:, :, ::-1]
            writer.write(frame_bgr)
        writer.release()
        print(f"  已保存: {out_path}")

        print(f"\n完成，输出目录: {out_dir}")


if __name__ == "__main__":
    main()
