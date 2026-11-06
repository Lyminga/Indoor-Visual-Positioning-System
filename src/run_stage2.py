# -*- coding: utf-8 -*-
"""
run_stage2.py —— 第二阶段实验：两阶段检索（VLAD 粗筛 + ORB 几何验证精排）
==========================================================================
在全部 query（data/query/same 24 张 + data/query_new 177 张，共 201 张）上
对比三种定位策略：
  single   纯 VLAD，直接信 top1
  vote     VLAD top5 相似度加权投票
  two_stage VLAD 粗筛 top10 + verify_match 按 RANSAC 内点数重排

产出（results/ 下）：
  exp_stage2_strategy.csv   三种策略的 R@1 / R@5 / 耗时
  exp_stage2_confusion.csv  两阶段策略的区域混淆矩阵
  exp_stage2_confusion.png  混淆矩阵热力图
  exp_stage2_errors.csv     两阶段错误样本清单（归因用）
"""
import csv
import os
import sys
import time
from collections import Counter

import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vlad_encoder as ve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FRAMES = os.path.join(ROOT, "data", "frames")


def load_queries():
    """合并原 same 组与补拍 query_new，返回 [(path, true_region), ...]"""
    queries = []
    rows = list(csv.DictReader(open(
        os.path.join(ROOT, "data", "meta", "query_meta.csv"), encoding="utf-8")))
    for r in rows:
        if r["category"] == "same":
            queries.append((os.path.join(ROOT, "data", "query", "same", r["filename"]),
                            r["region_id"]))
    new_dir = os.path.join(ROOT, "data", "query_new")
    rows = list(csv.DictReader(open(os.path.join(new_dir, "query_new_meta.csv"),
                                    encoding="utf-8")))
    for r in rows:
        p = os.path.join(new_dir, r["region_id"], r["filename"])
        if os.path.exists(p):  # R6 补拍为空，自然跳过
            queries.append((p, r["region_id"]))
    return queries


def main():
    feature, K = "ORB", 32
    tag = f"{feature}_K{K}"
    cfg = ve.load_config(os.path.join(RESULTS, f"vlad_config_{tag}.json"))
    index = faiss.read_index(os.path.join(RESULTS, f"map_{tag}.index"))
    meta = list(csv.DictReader(open(os.path.join(RESULTS, f"map_{tag}_meta.csv"),
                                    encoding="utf-8")))
    assert index.ntotal == len(meta)

    queries = load_queries()
    print(f"共 {len(queries)} 张 query，基线 {tag}")

    # 每张 query 只编码一次、只检索一次 top10；三种策略共用粗筛结果
    stats = {s: {"n1": 0, "n5": 0, "times": []} for s in ("single", "vote", "two_stage")}
    preds = {s: [] for s in stats}
    errors = []

    for i, (path, true_r) in enumerate(queries):
        t0 = time.perf_counter()
        q = ve.encode_single_image(path, cfg).reshape(1, -1)
        D, I = index.search(q, 10)
        enc_search = time.perf_counter() - t0

        regions = [meta[j]["region_id"] for j in I[0] if 0 <= j < len(meta)]
        dists = [d for d, j in zip(D[0], I[0]) if 0 <= j < len(meta)]
        top5 = regions[:5]

        # 策略1：单阶段 top1
        preds["single"].append((true_r, regions[0]))
        stats["single"]["n1"] += regions[0] == true_r
        stats["single"]["n5"] += true_r in top5
        stats["single"]["times"].append(enc_search)

        # 策略2：top5 相似度加权投票（权重 = 1/(d+eps)，只取前5）
        score = {}
        for d, r in zip(dists[:5], top5):
            score[r] = score.get(r, 0.0) + 1.0 / (d + 1e-6)
        vote = max(score, key=score.get)
        preds["vote"].append((true_r, vote))
        stats["vote"]["n1"] += vote == true_r
        stats["vote"]["n5"] += true_r in top5
        stats["vote"]["times"].append(enc_search)

        # 策略3：两阶段，top10 候选逐帧几何验证后按内点重排
        t1 = time.perf_counter()
        cand = []
        for d, j in zip(D[0], I[0]):
            if not (0 <= j < len(meta)):
                continue
            inl, _ = ve.verify_match(path, os.path.join(FRAMES, meta[j]["filename"]), cfg)
            cand.append((inl, meta[j]["region_id"], meta[j]["filename"]))
        cand.sort(key=lambda c: c[0], reverse=True)
        two_top1 = cand[0][1]
        stats["two_stage"]["times"].append(enc_search + (time.perf_counter() - t1))
        preds["two_stage"].append((true_r, two_top1))
        stats["two_stage"]["n1"] += two_top1 == true_r
        stats["two_stage"]["n5"] += true_r in top5  # 粗筛召回不变，便于对比精排收益
        if two_top1 != true_r:
            errors.append({"query": os.path.basename(path), "true": true_r,
                           "pred": two_top1, "top1_candidate": cand[0][2],
                           "inliers": cand[0][0]})

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(queries)} done")

    n = len(queries)
    rows = []
    for s in stats:
        st = stats[s]
        rows.append({"strategy": s, "recall@1": round(st["n1"] / n, 4),
                     "recall@5": round(st["n5"] / n, 4),
                     "total_ms": round(1000 * float(np.median(st["times"])), 1)})
        print(f"{s:9s} R@1={st['n1'] / n:.3f} ({st['n1']}/{n})  "
              f"R@5={st['n5'] / n:.3f}  median={rows[-1]['total_ms']}ms")
    with open(os.path.join(RESULTS, "exp_stage2_strategy.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 两阶段混淆矩阵
    regions = [f"R{i}" for i in range(8)]
    cm = np.zeros((8, 8), dtype=int)
    for t, p in preds["two_stage"]:
        cm[int(t[1])][int(p[1])] += 1
    with open(os.path.join(RESULTS, "exp_stage2_confusion.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + regions)
        for i, r in enumerate(regions):
            w.writerow([r] + list(cm[i]))

    with open(os.path.join(RESULTS, "exp_stage2_errors.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query", "true", "pred", "top1_candidate", "inliers"])
        w.writeheader()
        w.writerows(errors)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(regions); ax.set_yticklabels(regions)
    ax.set_xlabel("预测区域"); ax.set_ylabel("真实区域")
    for a in range(8):
        for b in range(8):
            ax.text(b, a, cm[a, b], ha="center", va="center",
                    color="red" if a != b and cm[a, b] else "black")
    ax.set_title(f"两阶段策略混淆矩阵（{tag}，{n} 张 query）")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "exp_stage2_confusion.png"), dpi=150)
    print(f"\n[DONE] 两阶段错误 {len(errors)} 张，明细见 exp_stage2_errors.csv")


if __name__ == "__main__":
    main()
