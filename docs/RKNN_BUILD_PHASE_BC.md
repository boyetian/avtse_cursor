# RKNN Phase B/C — build notes

## Recommended export (scatter, ~47 MB)

```bash
conda activate CVS  # torch + onnx
cd AV_TSE
python export_onnx.py --fixed --decoder_ola scatter \
  --fp32_out checkpoints/AV_Mossformer/av_mossformer2_fixed.onnx \
  --context_ms 100 --infer_chunk_ms 500 --skip_quant --skip_fp16 --skip_fp16_int8
```

## RKNN convert (default: no explicit `input_size_list`)

```bash
conda activate RKNN-Toolkit2
python convert_av_mossformer_rknn.py \
  --model checkpoints/AV_Mossformer/av_mossformer2_fixed.onnx \
  --ref_frames 18 --audio_len 9600
```

Use `--with_input_size` only for legacy attempts.

## Decoder OLA modes (`--decoder_ola`)

| Mode | ONNX | RKNN notes |
|------|------|------------|
| `scatter` (default) | ~47 MB, 1× ScatterElements | May pass OpEmit with default convert; LayoutMatch may still abort |
| `conv` | ConvTranspose, 0 Scatter | RKNN fold_constant may treat output as all-constants |
| `gather_add` | ~750 MB, 0 Scatter | fold_constant TypeError in toolkit 2.3.2 |

## Split deploy (B3)

```bash
# Sep → RKNN (output sep_pack = cat(mixture_w, est_mask) on dim=1)
python export_onnx.py --fixed --export_part sep \
  --fp32_out checkpoints/AV_Mossformer/av_mossformer2_sep_fixed.onnx ...

# Decoder → ORT on CPU
python export_onnx.py --fixed --export_part decoder --decoder_ola scatter \
  --fp32_out checkpoints/AV_Mossformer/av_mossformer2_decoder_fixed.onnx ...

python scripts/rknn_hybrid_infer.py --rknn ...sep_fixed.rknn \
  --decoder_onnx ...decoder_fixed.onnx --mixture_npy ... --ref_npy ...
```

## Verify

```bash
python scripts/verify_einsum_replacement.py
```
