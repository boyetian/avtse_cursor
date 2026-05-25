#!/usr/bin/env python3
"""Test ONNX model inference with 5D RGB or 4D grayscale ref input."""

import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
import soundfile as sf


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to grayscale using the same formula as the model.

    Args:
        rgb: shape [T, H, W, 3] or [B, T, H, W, 3]

    Returns:
        shape [T, H, W] or [B, T, H, W]
    """
    return 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]


def load_video_frames(video_path: str, num_frames: int, image_size: int = 96) -> np.ndarray:
    """Load and resize video frames.

    Returns:
        frames: [T, H, W, 3], float32, range [0, 1]
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (image_size, image_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()

    # Pad or trim to num_frames
    if len(frames) < num_frames:
        padding = [frames[-1]] * (num_frames - len(frames))
        frames.extend(padding)
    else:
        frames = frames[:num_frames]

    return np.stack(frames, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Test ONNX model inference")
    parser.add_argument("--onnx", type=str, required=True, help="ONNX model path")
    parser.add_argument("--audio", type=str, required=True, help="Input audio path")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default="output_onnx.wav", help="Output audio path")
    parser.add_argument("--audio_sr", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--ref_sr", type=float, default=30.0, help="Reference frame rate")
    parser.add_argument("--image_size", type=int, default=96, help="Face image size")
    parser.add_argument("--ref_is_gray", action="store_true", help="ONNX expects 4D grayscale ref (no RGB channel)")
    args = parser.parse_args()

    # 1. Load ONNX model
    print(f"Loading ONNX model: {args.onnx}")
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    # Print input/output info
    print("\nModel inputs:")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: shape={inp.shape}, dtype={inp.type}")
    print("\nModel outputs:")
    for out in sess.get_outputs():
        print(f"  {out.name}: shape={out.shape}, dtype={out.type}")

    # 2. Load audio
    print(f"\nLoading audio: {args.audio}")
    audio, sr = sf.read(args.audio, dtype="float32", always_2d=True)
    audio = audio.T.astype(np.float32)  # [C, T]
    if sr != args.audio_sr:
        print(f"WARNING: audio sr={sr} != expected {args.audio_sr}")

    # Use mono if multiple channels
    if audio.shape[0] > 1:
        audio = np.mean(audio, axis=0, keepdims=True)

    # 3. Load video frames
    audio_len_samples = audio.shape[1]
    num_ref_frames = max(2, int(np.ceil(float(audio_len_samples) / float(args.audio_sr) * float(args.ref_sr))))
    print(f"Audio length: {audio_len_samples} samples ({audio_len_samples/args.audio_sr:.2f}s)")
    print(f"Expecting {num_ref_frames} ref frames")

    print(f"\nLoading video: {args.video}")
    ref_frames = load_video_frames(args.video, num_ref_frames, args.image_size)
    print(f"Loaded ref frames shape: {ref_frames.shape}")

    # 4. Prepare ref input for ONNX
    if args.ref_is_gray:
        # Convert RGB to gray as preprocessing
        print("\nConverting RGB to grayscale (preprocessing for RKNN-friendly ONNX)")
        ref_gray = rgb_to_gray(ref_frames)  # [T, H, W]
        ref_input = ref_gray[np.newaxis, ...]  # [1, T, H, W]
    else:
        # Keep 5D RGB for original ONNX
        print("\nUsing 5D RGB ref (original ONNX)")
        ref_input = ref_frames[np.newaxis, ...]  # [1, T, H, W, 3]

    print(f"ref input shape: {ref_input.shape}")

    # 5. Run inference
    print("\nRunning inference...")
    t0 = time.time()
    result = sess.run(None, {"mixture": audio, "ref": ref_input.astype(np.float32)})
    t1 = time.time()
    print(f"Inference done in {t1-t0:.3f}s")

    output_audio = result[0]
    print(f"Output shape: {output_audio.shape}")

    # 6. Save output
    output_path = args.output
    sf.write(output_path, output_audio[0], args.audio_sr)
    print(f"\nSaved output to: {output_path}")

    # 7. Audio quality check
    rms_input = np.sqrt(np.mean(audio ** 2))
    rms_output = np.sqrt(np.mean(output_audio ** 2))
    print(f"\nRMS - Input: {rms_input:.6f}, Output: {rms_output:.6f}")

    if rms_output < 1e-6:
        print("WARNING: Output is near silent!")
    elif rms_output > 10:
        print("WARNING: Output is very loud!")


if __name__ == "__main__":
    main()
