Place models here before running on device:

RKNN (INFERENCE_BACKEND=RKNN):
  DFSMN_AEC_opt.rknn
  zipenhancer_full.rknn
  Convert on host: scripts/convert_dfsmn_aec_rknn.py, convert_zipenhancer_rknn.py

ONNX (INFERENCE_BACKEND=ONNX):
  DFSMN_AEC_opt.onnx
  zipenhancer_full.onnx
  Use same checkpoints as Linux ONNX build (no RKNN conversion).
