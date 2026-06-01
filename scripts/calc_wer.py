import os
import csv

GT_DIR = r"D:\GitDocs\AV_TSE_Cursor\测试结果\groundtruth"
ASR1_DIR = r"D:\GitDocs\AV_TSE_Cursor\测试结果\测试用例asr"
ASR2_DIR = r"D:\GitDocs\AV_TSE_Cursor\测试结果\讯飞\audio_asr"
OUT_CSV = r"D:\GitDocs\AV_TSE_Cursor\测试结果\wer_result.csv"


def levenshtein(ref, hyp):
    """计算两个字符串的编辑距离"""
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[m][n]


def calc_wer(ref, hyp):
    """WER = 编辑距离 / 参考文本长度 × 100%。空参考：两空=0%，仅ref空=100%"""
    if not ref:
        return 0.0 if not hyp else 100.0
    dist = levenshtein(ref, hyp)
    return round(dist / len(ref) * 100, 2)


def read_text(dirpath, num):
    """读取文件内容，文件不存在返回 None"""
    filepath = os.path.join(dirpath, f"{num}.txt")
    if not os.path.exists(filepath):
        return None
    with open(filepath, encoding="utf-8") as f:
        return f.read().strip()


rows = []
# GT != 0 统计
s1_wer_gt = []   # 测试用例ASR WER列表
s2_wer_gt = []   # 讯飞ASR WER列表
s1_miss_gt = 0
s2_miss_gt = 0
# GT == 0 统计
s1_wer_empty = []
s2_wer_empty = []
s1_miss_empty = 0
s2_miss_empty = 0

for num in range(1, 212):
    gt = read_text(GT_DIR, num)
    asr2 = read_text(ASR2_DIR, num)

    # 只统计 讯飞/audio_asr 里有的样本
    if asr2 is None:
        continue

    asr1 = read_text(ASR1_DIR, num)

    if gt is None:
        continue

    gt_has_text = bool(gt)

    # 测试用例ASR
    if asr1 is None:
        wer1 = "无样本"
        if gt_has_text: s1_miss_gt += 1
        else: s1_miss_empty += 1
    else:
        wer1 = calc_wer(gt, asr1)
        if gt_has_text: s1_wer_gt.append(wer1)
        else: s1_wer_empty.append(wer1)

    # 讯飞ASR
    if asr2 is None:
        wer2 = "无样本"
        if gt_has_text: s2_miss_gt += 1
        else: s2_miss_empty += 1
    else:
        wer2 = calc_wer(gt, asr2)
        if gt_has_text: s2_wer_gt.append(wer2)
        else: s2_wer_empty.append(wer2)

    rows.append([num, len(gt), wer1, wer2])

def avg_or_na(lst):
    return round(sum(lst) / len(lst), 2) if lst else "N/A"

# 写 CSV
with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["编号", "GT字数", "测试用例ASR_WER(%)", "讯飞ASR_WER(%)"])
    for row in rows:
        w.writerow(row)

    w.writerow([])
    w.writerow(["统计", "", "", ""])

    # GT != 0
    w.writerow(["-- GT != 0 (有参考文本) --", "", "", ""])
    w.writerow(["测试用例ASR 有效样本", len(s1_wer_gt), "平均WER(%)", avg_or_na(s1_wer_gt)])
    w.writerow(["测试用例ASR 无样本", s1_miss_gt, "", ""])
    w.writerow(["讯飞ASR 有效样本", len(s2_wer_gt), "平均WER(%)", avg_or_na(s2_wer_gt)])
    w.writerow(["讯飞ASR 无样本", s2_miss_gt, "", ""])

    w.writerow([])
    # GT == 0
    w.writerow(["-- GT == 0 (空参考) --", "", "", ""])
    w.writerow(["测试用例ASR 有效样本", len(s1_wer_empty), "平均WER(%)", avg_or_na(s1_wer_empty)])
    w.writerow(["测试用例ASR 无样本", s1_miss_empty, "", ""])
    w.writerow(["讯飞ASR 有效样本", len(s2_wer_empty), "平均WER(%)", avg_or_na(s2_wer_empty)])
    w.writerow(["讯飞ASR 无样本", s2_miss_empty, "", ""])

    w.writerow([])
    total1 = len(s1_wer_gt) + len(s1_wer_empty)
    total2 = len(s2_wer_gt) + len(s2_wer_empty)
    w.writerow(["-- 合计 --", "", "", ""])
    w.writerow(["测试用例ASR 总有效样本", total1, "总平均WER(%)", avg_or_na(s1_wer_gt + s1_wer_empty)])
    w.writerow(["讯飞ASR 总有效样本", total2, "总平均WER(%)", avg_or_na(s2_wer_gt + s2_wer_empty)])
    w.writerow(["总编号数", len(rows), "", ""])

print(f"WER 计算结果已保存: {OUT_CSV}")
print(f"--- GT != 0 ---")
print(f"测试用例ASR: {len(s1_wer_gt)} 样本, 平均WER {avg_or_na(s1_wer_gt)}%, 无样本 {s1_miss_gt}")
print(f"讯飞ASR:     {len(s2_wer_gt)} 样本, 平均WER {avg_or_na(s2_wer_gt)}%, 无样本 {s2_miss_gt}")
print(f"--- GT == 0 ---")
print(f"测试用例ASR: {len(s1_wer_empty)} 样本, 平均WER {avg_or_na(s1_wer_empty)}%, 无样本 {s1_miss_empty}")
print(f"讯飞ASR:     {len(s2_wer_empty)} 样本, 平均WER {avg_or_na(s2_wer_empty)}%, 无样本 {s2_miss_empty}")
