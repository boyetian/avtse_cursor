Place models here for adb smoke / full SDK / board GTests:

RKNN (INFERENCE_BACKEND=RKNN):
  av_mossformer2.rknn   # from convert_av_mossformer_rknn.py

ONNX (INFERENCE_BACKEND=ONNX):
  av_mossformer2.onnx   # same as Linux; board GTest: cpp/test/scripts/build_android_onnx_test.sh

Fixed RKNN shapes must match the conversion script (default hop 500ms + context 100ms @ 16kHz):

  mixture: [1, 9600]
  ref:     [1, 20, 96, 96, 3]
