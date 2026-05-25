# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def si_sdr(est, ref):
    est = est - np.mean(est)
    ref = ref - np.mean(ref)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    s_target = alpha * ref
    e_noise = est - s_target
    val = 10 * np.log10(np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + 1e-12) + 1e-12)
    return val

target, sr_t = sf.read('测试用例/target/03.wav')
if target.ndim > 1:
    target = target[:, 0]

files = [
    '测试结果/03_out.wav',
]

for f in files:
    if not os.path.exists(f):
        print(f'{f}: not found')
        continue
    est, sr_e = sf.read(f)
    if est.ndim > 1:
        est = est[:, 0]
    min_len = min(len(est), len(target))
    est = est[:min_len]
    ref = target[:min_len]
    val = si_sdr(est.astype(np.float64), ref.astype(np.float64))
    print(f'{f}: SI-SDR = {val:.3f} dB')
