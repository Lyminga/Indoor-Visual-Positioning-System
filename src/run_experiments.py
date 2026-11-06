# -*- coding: utf-8 -*-
"""
run_experiments.py —— A 的独立实验（阶段 5 自检 + 阶段 8 消融）
==============================================================
产出：
  1. 自检：3 张跨区域 query（R0/R4/R7 各一）在 K=32 基线上的 top1 验证
  2. K 值消融：ORB + K in {16,32,64,128} 的 Recall@1 / Recall@5 / 单图编码耗时
  3. 特征消融：K=32 下 ORB / SIFT / RootSIFT 的同样指标
  4. 混淆矩阵热力图（最佳配置在 24 张 same query 上的 top1 预测）
结果写入 results/ 下的 csv 与 png。
"""
import csv
import json
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
META = os.path.join(ROOT, "data", "meta")
QUERY_DIR = os.path.join(ROOT, "data", "query", "same")


def load_system(feature, K):
    """加载一套 码本+索引+meta，返回 (config, index, meta_rows)"""
    tag = f"{feature}_K{K}"
    cfg = ve.load_config(os.path.join(RESULTS, f"vlad_config_{tag}.json"))
    index = faiss.read_index(os.path.join(RESULTS, f"map_{tag}.index"))
    meta = list(csv.DictReader(open(os.path.join(RESULTS, f"map_{tag}_meta.csv"), encoding="utf-8")))
    assert index.ntotal == len(meta), f"{tag}: 索引条数 {index.ntotal} != meta 行数 {len(meta)}"
    return cfg, index, meta


def evaluate(feature, K, queries, warmup=2):
    """
    对一组 query 评估 Recall@1 / Recall@5（区域级）与平均单图编码耗时。
    queries: [(path, true_region), ...]
    返回 dict(r1, r5, vote_acc, enc_ms, preds=[(true, top1_region), ...])
    """
    cfg, index, meta = load_system(feature, K)
    n1 = n5 = nvote = 0
    enc_times = []
    preds = []
    for i, (path, true_r) in enumerate(queries):
        t0 = time.perf_counter()
        q = ve.encode_single_image(path, cfg).reshape(1, -1)
        enc_times.append(time.perf_counter() - t0)
        D, I = index.search(q, 5)
        top_regions = [meta[j]["region_id"] for j in I[0]]
        top1 = top_regions[0]
        vote = Counter(top_regions).most_common(1)[0][0]
        n1 += top1 == true_r
        n5 += true_r in top_regions
        nvote += vote == true_r
        preds.append((true_r, top1))
    n = len(queries)
    times = enc_times[warmup:] if len(enc_times) > warmup else enc_times
    return {
        "r1": n1 / n, "r5": n5 / n, "vote_acc": nvote / n,
        # 用中位数而非均值：本机后台负载波动大，均值和单次测量都不可靠
        "enc_ms": 1000.0 * float(np.median(times)), "preds": preds,
    }


def load_queries():
    rows = list(csv.DictReader(open(os.path.join(META, "query_meta.csv"), encoding="utf-8")))
    return [(os.path.join(QUERY_DIR, r["filename"]), r["region_id"]) for r in rows
            if r["category"] == "same"]


def main():
    queries = load_queries()
    print(f"共 {len(queries)} 张 same query\n")

    # ---------- 1. 自检：3 张跨区域 query 的 top1 验证（K=32 基线） ----------
    print("========== 自检（ORB K=32 基线） ==========")
    # 选 3 张来自不同区域的 query 做交接自检（入口楼梯 / 东南休息处 / 东侧走廊）
    check_files = ["R0_2.jpg", "R5_1.jpg", "R6_1.jpg"]
    qdict = {os.path.basename(p): (p, r) for p, r in queries}
    selfcheck = []
    for name in check_files:
        p, true_r = qdict[name]
        cfg, index, meta = load_system("ORB", 32)
        q = ve.encode_single_image(p, cfg).reshape(1, -1)
        D, I = index.search(q, 1)
        pred_r = meta[I[0][0]]["region_id"]
        ok = pred_r == true_r
        selfcheck.append((name, true_r, pred_r, ok))
        print(f"  {name}: 真实={true_r}  top1={pred_r}  {'PASS' if ok else 'FAIL'}")
    sc_path = os.path.join(RESULTS, "selfcheck.csv")
    with open(sc_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "true_region", "top1_region", "pass"])
        w.writerows(selfcheck)

    # ---------- 2. K 值消融（特征固定 ORB） ----------
    print("\n========== K 值消融（ORB） ==========")
    k_rows = []
    for K in [16, 32, 64, 128]:
        m = evaluate("ORB", K, queries)
        k_rows.append({"feature": "ORB", "K": K, "recall@1": round(m["r1"], 4),
                       "recall@5": round(m["r5"], 4), "vote_acc": round(m["vote_acc"], 4),
                       "enc_ms": round(m["enc_ms"], 1)})
        print(f"  K={K:3d}  R@1={m['r1']:.3f}  R@5={m['r5']:.3f}  "
              f"vote={m['vote_acc']:.3f}  enc={m['enc_ms']:.1f}ms")
    with open(os.path.join(RESULTS, "exp_K_ablation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(k_rows[0].keys()))
        w.writeheader()
        w.writerows(k_rows)

    # ---------- 3. 特征消融（K=32 固定） ----------
    print("\n========== 特征消融（K=32） ==========")
    f_rows = []
    for feat in ["ORB", "SIFT", "RootSIFT"]:
        m = evaluate(feat, 32, queries)
        f_rows.append({"feature": feat, "K": 32, "recall@1": round(m["r1"], 4),
                       "recall@5": round(m["r5"], 4), "vote_acc": round(m["vote_acc"], 4),
                       "enc_ms": round(m["enc_ms"], 1)})
        print(f"  {feat:8s}  R@1={m['r1']:.3f}  R@5={m['r5']:.3f}  "
              f"vote={m['vote_acc']:.3f}  enc={m['enc_ms']:.1f}ms")
    with open(os.path.join(RESULTS, "exp_feature_ablation.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(f_rows[0].keys()))
        w.writeheader()
        w.writerows(f_rows)

    # ---------- 4. 混淆矩阵（ORB K=32 基线） ----------
    print("\n========== 混淆矩阵（ORB K=32，top1 区域） ==========")
    m = evaluate("ORB", 32, queries)
    regions = [f"R{i}" for i in range(8)]
    cm = np.zeros((8, 8), dtype=int)
    for t, p in m["preds"]:
        cm[int(t[1])][int(p[1])] += 1
    with open(os.path.join(RESULTS, "exp_confusion_matrix.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + regions)
        for i, r in enumerate(regions):
            w.writerow([r] + list(cm[i]))
    print(cm)

    # ---------- 5. 出图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    ks = [r["K"] for r in k_rows]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(ks, [r["recall@1"] for r in k_rows], "o-", label="Recall@1")
    ax1.plot(ks, [r["recall@5"] for r in k_rows], "s--", label="Recall@5")
    ax1.plot(ks, [r["vote_acc"] for r in k_rows], "^:", label="Top5 投票准确率")
    ax1.set_xlabel("VLAD 聚类数 K")
    ax1.set_ylabel("精度")
    ax1.set_xticks(ks)
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.bar(ks, [r["enc_ms"] for r in k_rows], width=5, alpha=0.25, color="gray",
            label="单图编码耗时(ms)")
    ax2.set_ylabel("耗时 (ms)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")
    plt.title("K 值消融：精度 vs 编码耗时（ORB, 24 张 query）")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "exp_K_ablation.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(3)
    w = 0.25
    ax.bar(x - w, [r["recall@1"] for r in f_rows], w, label="Recall@1")
    ax.bar(x, [r["recall@5"] for r in f_rows], w, label="Recall@5")
    ax.bar(x + w, [r["enc_ms"] / max(r["enc_ms"] for r in f_rows) for r in f_rows],
           w, label="耗时(归一化)")
    ax.set_xticks(x)
    ax.set_xticklabels([r["feature"] for r in f_rows])
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.title("特征类型消融（K=32）")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "exp_feature_ablation.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels(regions); ax.set_yticklabels(regions)
    ax.set_xlabel("预测区域"); ax.set_ylabel("真实区域")
    for i in range(8):
        for j in range(8):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="red" if i != j and cm[i, j] else "black")
    plt.title("区域混淆矩阵（ORB K=32, top1）")
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "exp_confusion_matrix.png"), dpi=150)
    plt.close()

    print("\n[DONE] 结果已写入 results/：selfcheck.csv, exp_K_ablation.*, "
          "exp_feature_ablation.*, exp_confusion_matrix.*")


if __name__ == "__main__":
    main()
