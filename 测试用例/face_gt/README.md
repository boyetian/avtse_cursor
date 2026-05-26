# 人脸跟踪 Ground Truth

用于 `eval_face_metrics.py` 自动计算 **IoU** 与 **MOTA**。

## 文件命名

- 与视频对应：`测试用例/视频/03.mp4` → `测试用例/face_gt/03.json`

## JSON 格式

```json
{
  "video_id": "03",
  "fps": 25.0,
  "target_track_id": 1,
  "frames": [
    {
      "frame_idx": 0,
      "objects": [
        {"track_id": 1, "x1": 120, "y1": 80, "x2": 280, "y2": 260}
      ]
    }
  ]
}
```

- 坐标：原图像素，轴对齐框 `x1,y1` 左上，`x2,y2` 右下。
- `target_track_id`：当前发言人（评测只对该 id 计 GT）。
- 无发言人帧：`"objects": []`。

## 人工评测

填写 `manual_scores.csv`：

| 列 | 含义 |
|----|------|
| `error_detect_sec` | IoU 持续偏低（或跟错人）的累计秒数 |
| `total_sec` | 视频总时长 |
| `id_switch_count` | 框跳到非发言人的次数 |

```bash
python eval_face_metrics.py --manual-csv ./测试用例/face_gt/manual_scores.csv
```

## 自动评测

```bash
python eval_face_metrics.py --video-dir ./测试用例/视频 --gt-dir ./测试用例/face_gt --face-detector mediapipe
python eval_face_metrics.py --face-detector haar --debug-video-dir ./测试结果_人脸框
```
