# face_detector.py — 人脸门控（三层流水线第 2 层）

封装 MediaPipe BlazeFace 检测 + 目标跟踪。作为流水线的门控层：接收上游的 `v_i`，有人脸时输出 `crops + face_valid` 供推理层消费，无人脸时跳过不输出。

## 门控逻辑

```
v_i (视频帧) → VisualPreprocessor.process_chunk(v_i)
  → any_detected == True  → 放行: crops + face_valid → 送入推理层
  → any_detected == False → 跳过: 不送入推理层
```

## 用法 1：main.py 命令行（文件批处理）

```bash
python main.py --face-only
```

常用参数：

```bash
python main.py --face-only \
    --files 73 \                          # 只处理指定编号，留空=全部
    --detect_every_n 5 \                  # 检测间隔帧数
    --score_smooth_alpha 0.8 \            # 置信度平滑
    --box_smooth_alpha 0 \                # 框平滑 (0=不平滑)
    --min_detection_confidence 0.5 \      # 最低检测阈值
    --mediapipe_detect_max_side 320       # 检测前降采样最大边长
```

输出：RTF 统计 + 标注视频（`./测试结果_标注视频/{编号}_annotated.mp4`）

内部调用链：`main.py` → `VisualPreprocessor.process_chunk()` → 逐帧 `FaceMediaPipeStreamTracker.process_bgr()`

## 用法 2：直接调用流式（逐帧 / 摄像头）

```python
import cv2
from face_detector import VisualPreprocessor

detector = VisualPreprocessor(model_path="blaze_face_full_range.tflite")

cap = cv2.VideoCapture(0)  # 或视频文件
while True:
    ret, frame = cap.read()
    if not ret:
        break
    r = detector.process_frame(frame)
    # r["face_detected"]   -> bool                     是否检测到人脸
    # r["face_box"]        -> List[float]|None         人脸框 [x1,y1,x2,y2]
    # r["detection_score"] -> float|None               置信度 0~1
    # r["mouth_center"]    -> Tuple[float,float]|None  嘴部中心 (x,y)
    # r["crop"]            -> np.ndarray               裁剪 (128x128 RGB float32)

    if r["face_detected"]:
        x1, y1, x2, y2 = map(int, r["face_box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("face", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 换新视频/会话时重置跟踪状态
detector.reset()
```
