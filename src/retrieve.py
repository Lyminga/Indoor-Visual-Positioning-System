# -*- coding: utf-8 -*-
"""
retrieve.py —— 在线检索封装（Gradio 界面与实验脚本共用）
==========================================================
Locator 在启动时一次性加载 码本 + FAISS 索引 + meta + 区域表，
之后每张查询图调用 locate() 即可，避免每次请求重复读盘。

用法：
    locator = Locator("results/vlad_config_ORB_K32.json")
    result = locator.locate("some_query.jpg", strategy="two_stage")
    # result: region / region_name / confidence / candidates / elapsed_ms
"""
import csv
import json
import os
import time
from collections import Counter

import faiss

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vlad_encoder as ve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Locator:
    def __init__(self, config_path):
        """config_path 形如 results/vlad_config_ORB_K32.json，索引与 meta 按同名规则推导"""
        self.cfg = ve.load_config(config_path)
        tag = f"{self.cfg['feature']}_K{self.cfg['K']}"
        results_dir = os.path.dirname(os.path.abspath(config_path))
        self.index = faiss.read_index(os.path.join(results_dir, f"map_{tag}.index"))
        self.meta = list(csv.DictReader(open(
            os.path.join(results_dir, f"map_{tag}_meta.csv"), encoding="utf-8")))
        assert self.index.ntotal == len(self.meta), "索引条数与 meta 行数不一致"

        self.frames_dir = os.path.join(ROOT, "data", "frames")
        # 区域 id -> 中文名，用于界面展示
        regions_path = os.path.join(ROOT, "data", "meta", "regions.csv")
        self.region_names = {}
        if os.path.exists(regions_path):
            for r in csv.DictReader(open(regions_path, encoding="utf-8")):
                self.region_names[r["region_id"]] = r.get("description", r["region_id"])

    def locate(self, image_path, strategy="two_stage", top_k=10):
        """
        strategy:
            single    纯 VLAD top1
            vote      top5 相似度加权投票
            two_stage VLAD 粗筛 + 几何验证精排（默认，抗相似场景混淆）
        返回 dict，含区域、置信度、候选列表、耗时。
        """
        t0 = time.perf_counter()
        if strategy == "two_stage":
            res = ve.locate_image(image_path, self.cfg, self.index, self.meta,
                                  self.frames_dir, top_k=top_k, use_verification=True)
        else:
            q = ve.encode_single_image(image_path, self.cfg).reshape(1, -1)
            D, I = self.index.search(q, top_k)
            cand = []
            for d, j in zip(D[0], I[0]):
                if not (0 <= j < len(self.meta)):
                    continue
                # faiss IndexFlatL2 返回平方 L2 距离 d；向量已 L2 归一化，
                # 余弦相似度 = 1 - d/2（不要再平方一次）
                cand.append({"filename": self.meta[j]["filename"],
                             "region_id": self.meta[j]["region_id"],
                             "similarity": float(1.0 - d / 2.0), "inliers": -1})
            if strategy == "single":
                res = {"region": cand[0]["region_id"] if cand else None,
                       "confidence": cand[0]["similarity"] if cand else 0.0,
                       "candidates": cand}
            elif strategy == "vote":
                top5 = cand[:5]
                score = {}
                for c in top5:
                    # 2*(1-sim) 即 faiss 的平方 L2 距离 D；沿用按 D^2 反比加权，
                    # 保证与修正相似度前的投票结果一致（低距离高权重）
                    d = 2.0 * (1.0 - c["similarity"])
                    w = 1.0 / (d * d + 1e-6)
                    score[c["region_id"]] = score.get(c["region_id"], 0.0) + w
                region = max(score, key=score.get) if score else None
                res = {"region": region,
                       "confidence": cand[0]["similarity"] if cand else 0.0,
                       "candidates": cand}
            else:
                raise ValueError(f"未知策略: {strategy}（single / vote / two_stage）")

        res["region_name"] = self.region_names.get(res["region"], res["region"])
        res["elapsed_ms"] = round(1000 * (time.perf_counter() - t0), 1)
        return res


if __name__ == "__main__":
    # 命令行自测：python src/retrieve.py <query图> [strategy]
    cfg_path = os.path.join(ROOT, "results", "vlad_config_ORB_K32.json")
    loc = Locator(cfg_path)
    q = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "query", "same", "R0_1.jpg")
    st = sys.argv[2] if len(sys.argv) > 2 else "two_stage"
    r = loc.locate(q, strategy=st)
    print(json.dumps({k: v for k, v in r.items() if k != "candidates"},
                     ensure_ascii=False, indent=2))
    for c in r["candidates"][:5]:
        print(f"  {c['filename']:20s} {c['region_id']} sim={c['similarity']:.3f} inliers={c['inliers']}")
