# -*- coding: utf-8 -*-
"""
build_index.py —— 训练 K-Means 码本并构建 FAISS 索引（交付物 2）
================================================================
流程（对每个 K in --ks，默认 16/32/64/128）：
  1. 遍历 frames_meta.csv 中的全部地图帧，提取局部描述子
  2. 用【仅地图帧】的描述子训练 KMeans 码本 -> results/codebook_{feat}_K{K}.npy
     （绝不混入 query 特征，否则数据泄露）
  3. encode_batch 编码全部地图帧 -> VLAD 矩阵 (N, K*desc_dim)
  4. 构建 FAISS IndexFlatL2 -> results/map_{feat}_K{K}.index
     索引内向量顺序 == frames_meta.csv 行顺序 == map meta csv 行顺序
  5. 写 map_{feat}_K{K}_meta.csv 与 vlad_config_{feat}_K{K}.json
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vlad_encoder as ve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = os.path.join(ROOT, "data", "meta")
FRAMES = os.path.join(ROOT, "data", "frames")
RESULTS = os.path.join(ROOT, "results")

# 真实描述子维度以 OpenCV 输出为准（ORB=32 维 uint8，SIFT/RootSIFT=128 维 float）
DESC_DIM = {"ORB": 32, "SIFT": 128, "ROOTSIFT": 128}

# KMeans 训练最多采样多少描述子（太多会很慢，20 万条足够收敛）
MAX_TRAIN_DESC = 200_000
KMEANS_ITER = 30

# 关键点少于该数的地图帧视为退化帧（严重运动模糊等），不入索引，
# 否则其 VLAD 接近零向量，会成为"枢纽"对所有 query 距离都约等于 1.0，污染检索
MIN_KEYPOINTS = 30


def collect_descriptors(image_paths, cfg):
    """对全部地图帧提描述子并堆叠成大矩阵，同时返回每张图的关键点数"""
    detector = ve._make_detector(cfg)
    blocks, counts = [], []
    for i, p in enumerate(image_paths):
        import cv2
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        d = ve._extract_descriptors(img, cfg, detector=detector)
        counts.append(d.shape[0])
        if d.shape[0]:
            blocks.append(d)
        if (i + 1) % 50 == 0:
            print(f"    提特征 {i + 1}/{len(image_paths)}")
    return np.vstack(blocks), counts


def train_codebook(desc, K, seed=42):
    """用 faiss.Kmeans 训练码本（比 sklearn 快一个量级），返回 K x dim float32"""
    import faiss
    if desc.shape[0] > MAX_TRAIN_DESC:
        rng = np.random.default_rng(seed)
        idx = rng.choice(desc.shape[0], MAX_TRAIN_DESC, replace=False)
        desc = desc[idx]
    d = desc.shape[1]
    km = faiss.Kmeans(d, K, niter=KMEANS_ITER, seed=seed, verbose=False)
    km.train(desc)
    return km.centroids.astype(np.float32)


def build_for(feature, K, image_paths, filenames, region_ids):
    """训练一个码本 + 构建一套索引，返回 config dict"""
    import faiss

    feature = feature.upper()
    dim = DESC_DIM[feature]
    cfg = {
        "feature": feature,
        "nfeatures": 2000,
        "K": K,
        "descriptor_dim": dim,
        "vlad_dim": K * dim,
        "normalization": "power_law + L2",
        "frame_size": [640, 480],
        "codebook_path": os.path.join(RESULTS, f"codebook_{feature}_K{K}.npy"),
    }

    t0 = time.time()
    print(f"  [{feature} K={K}] 收集地图帧描述子 ...")
    all_desc = collect_descriptors(image_paths, cfg)
    print(f"  [{feature} K={K}] 共 {all_desc.shape[0]} 条描述子，训练 KMeans ...")
    codebook = train_codebook(all_desc, K)
    np.save(cfg["codebook_path"], codebook)

    print(f"  [{feature} K={K}] 批量编码 {len(image_paths)} 张地图帧 ...")
    mat = ve.encode_batch(image_paths, cfg)          # (N, vlad_dim)，顺序与 csv 一致

    index = faiss.IndexFlatL2(cfg["vlad_dim"])
    index.add(mat.astype(np.float32))
    idx_path = os.path.join(RESULTS, f"map_{feature}_K{K}.index")
    faiss.write_index(index, idx_path)

    meta_path = os.path.join(RESULTS, f"map_{feature}_K{K}_meta.csv")
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "region_id"])
        w.writerows(zip(filenames, region_ids))

    cfg["frame_count"] = len(image_paths)
    cfg["build_seconds"] = round(time.time() - t0, 1)
    cfg["codebook_path"] = f"codebook_{feature}_K{K}.npy"  # json 里存相对路径，便于打包迁移
    cfg_path = os.path.join(RESULTS, f"vlad_config_{feature}_K{K}.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print(f"  [{feature} K={K}] 完成：{idx_path}  ({cfg['build_seconds']}s)")
    return cfg_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--features", type=str, nargs="+", default=["ORB"],
                    choices=["ORB", "SIFT", "RootSIFT"])
    ap.add_argument("--no-filter", action="store_true",
                    help="跳过退化帧过滤（frames_meta.csv 已被首轮过滤时使用，避免不同特征过滤结果不一致）")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(META, "frames_meta.csv"), encoding="utf-8")))
    filenames = [r["filename"] for r in rows]
    region_ids = [r["region_id"] for r in rows]
    image_paths = [os.path.join(FRAMES, n) for n in filenames]

    for feat in args.features:
        # 同一特征只收集一次描述子，多个 K 复用
        dim = DESC_DIM[feat.upper()]
        base_cfg = {"feature": feat, "nfeatures": 2000, "descriptor_dim": dim,
                    "frame_size": [640, 480]}
        print(f"[{feat}] 提取全部地图帧描述子（供所有 K 共用）...")
        all_desc, kp_counts = collect_descriptors(image_paths, base_cfg)

        # ---- 退化帧过滤：关键点太少的帧不入索引，并从 frames_meta 移到 dropped ----
        if args.no_filter:
            keep = list(range(len(kp_counts)))
            drop = []
        else:
            keep = [i for i, c in enumerate(kp_counts) if c >= MIN_KEYPOINTS]
            drop = [i for i, c in enumerate(kp_counts) if c < MIN_KEYPOINTS]
        if drop:
            print(f"[{feat}] 过滤 {len(drop)} 张退化帧（关键点<{MIN_KEYPOINTS}）: "
                  f"{[filenames[i] for i in drop]}")
            dropped_csv = os.path.join(META, "frames_dropped.csv")
            with open(dropped_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for i in drop:
                    w.writerow([filenames[i], rows[i]["video"], rows[i]["time_sec"],
                                region_ids[i], "", "", f"few_keypoints({kp_counts[i]})"])
            # 重写 frames_meta.csv，只保留合格帧，保证索引行序与 meta 一致
            with open(os.path.join(META, "frames_meta.csv"), "w", newline="",
                      encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["filename", "video", "time_sec", "region_id"])
                w.writeheader()
                for i in keep:
                    w.writerow(rows[i])
        filenames_f = [filenames[i] for i in keep]
        region_ids_f = [region_ids[i] for i in keep]
        image_paths_f = [image_paths[i] for i in keep]
        print(f"[{feat}] 入索引地图帧: {len(image_paths_f)} 张")

        for K in args.ks:
            import faiss, time as _t
            t0 = _t.time()
            print(f"  [{feat} K={K}] 训练 KMeans ...")
            codebook = train_codebook(all_desc, K)
            cb_path = os.path.join(RESULTS, f"codebook_{feat.upper()}_K{K}.npy")
            np.save(cb_path, codebook)

            cfg = {
                "feature": feat.upper(), "nfeatures": 2000, "K": K,
                "descriptor_dim": dim, "vlad_dim": K * dim,
                "normalization": "power_law + L2", "frame_size": [640, 480],
                "codebook_path": cb_path,
            }
            print(f"  [{feat} K={K}] 批量编码 {len(image_paths_f)} 张地图帧 ...")
            mat = ve.encode_batch(image_paths_f, cfg)

            index = faiss.IndexFlatL2(cfg["vlad_dim"])
            index.add(mat.astype(np.float32))
            idx_path = os.path.join(RESULTS, f"map_{feat.upper()}_K{K}.index")
            faiss.write_index(index, idx_path)

            meta_path = os.path.join(RESULTS, f"map_{feat.upper()}_K{K}_meta.csv")
            with open(meta_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["filename", "region_id"])
                w.writerows(zip(filenames_f, region_ids_f))

            cfg["frame_count"] = len(image_paths_f)
            cfg["build_seconds"] = round(_t.time() - t0, 1)
            cfg["codebook_path"] = f"codebook_{feat.upper()}_K{K}.npy"
            with open(os.path.join(RESULTS, f"vlad_config_{feat.upper()}_K{K}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print(f"  [{feat} K={K}] 完成 ({cfg['build_seconds']}s)")


if __name__ == "__main__":
    main()
