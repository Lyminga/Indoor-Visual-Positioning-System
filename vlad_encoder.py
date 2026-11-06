# -*- coding: utf-8 -*-
"""
vlad_encoder.py —— VLAD 编码模块（交付物 3，给 B 直接 import 调用）
====================================================================
对外只暴露两个接口：

    encode_single_image(image_path, config) -> np.ndarray  # shape=(vlad_dim,)
    encode_batch(image_paths, config)       -> np.ndarray  # shape=(N, vlad_dim)

config 可以是 dict，也可以是 vlad_config_Kxx.json 的路径。
所有参数（特征类型、nfeatures、K、码本路径、归一化方式、图像尺寸）
都从 config 读取，函数内部不硬编码任何 K 或路径。

VLAD 流程：
    读图 -> resize 640x480 -> 提局部描述子 -> 每个描述子分配到最近聚类中心
    -> 累加残差 (desc - center) -> 拼接成 K*desc_dim 维向量
    -> Power-law 归一化 (逐元素 sign(x)*sqrt(|x|)) -> L2 归一化

特征说明：
    ORB      : 64 维二进制描述子，转 float32 使用
    SIFT     : 128 维浮点描述子
    RootSIFT : 对 SIFT 描述子先做 L1 归一化再开方，检索通常优于原生 SIFT
"""
import json
import os

import cv2
import numpy as np

# 码本缓存：避免 encode_batch 里每张图都重复读一次 npy
_CODEBOOK_CACHE = {}


# ------------------------------------------------------------------
# config 读取
# ------------------------------------------------------------------
def load_config(config):
    """config 既支持 dict，也支持 json 文件路径；统一转成 dict 返回"""
    if isinstance(config, dict):
        cfg = dict(config)
    else:
        with open(config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    # 兼容相对路径：如果 codebook_path 是相对路径，则相对于 json 所在目录解析
    if isinstance(config, str) and not os.path.isabs(cfg.get("codebook_path", "")):
        base = os.path.dirname(os.path.abspath(config))
        cfg["codebook_path"] = os.path.join(base, cfg["codebook_path"])
    return cfg


def _get_codebook(cfg):
    """按路径缓存加载 K x desc_dim 的码本"""
    path = cfg["codebook_path"]
    if path not in _CODEBOOK_CACHE:
        cb = np.load(path).astype(np.float32)
        if cb.shape[0] != cfg["K"]:
            raise ValueError(f"码本聚类数 {cb.shape[0]} 与 config 的 K={cfg['K']} 不一致")
        _CODEBOOK_CACHE[path] = cb
    return _CODEBOOK_CACHE[path]


# ------------------------------------------------------------------
# 特征提取
# ------------------------------------------------------------------
def _make_detector(cfg):
    """根据 config 里的 feature 字段创建检测子"""
    feature = cfg["feature"].upper()
    nfeat = int(cfg.get("nfeatures", 2000))
    if feature == "ORB":
        # ORB 描述子是 64 维二进制串，这里仍按 0/1 浮点向量使用
        return cv2.ORB_create(nfeatures=nfeat)
    if feature in ("SIFT", "ROOTSIFT"):
        return cv2.SIFT_create(nfeatures=nfeat)
    raise ValueError(f"不支持的特征类型: {cfg['feature']}（可选 ORB / SIFT / RootSIFT）")


def _root_sift(desc):
    """RootSIFT：L1 行归一化后逐元素开方（Hellinger 核的等价变换）"""
    desc = desc.astype(np.float32)
    desc /= (desc.sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(desc)


def _extract_descriptors(image, cfg, detector=None):
    """
    对单张 BGR 图提取描述子矩阵。
    返回 (N, desc_dim) float32；无关键点时返回 (0, desc_dim)。
    """
    h, w = int(cfg["frame_size"][1]), int(cfg["frame_size"][0])
    if image.shape[0] != h or image.shape[1] != w:
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detector = detector or _make_detector(cfg)
    _, desc = detector.detectAndCompute(gray, None)

    # 真实维度以检测子输出为准（OpenCV 的 ORB 为 32 维 uint8，SIFT 为 128 维 float）
    dim = int(cfg.get("descriptor_dim") or 0) or (desc.shape[1] if desc is not None else 0)
    if desc is None:
        cb_dim = _get_codebook(cfg).shape[1] if cfg.get("codebook_path") else (dim or 1)
        return np.empty((0, cb_dim), dtype=np.float32)

    if cfg["feature"].upper() == "ROOTSIFT":
        desc = _root_sift(desc)
    return desc.astype(np.float32)


# ------------------------------------------------------------------
# 核心：VLAD 编码
# ------------------------------------------------------------------
def _assign_to_codebook(desc, codebook, chunk=4096):
    """
    把描述子分配到最近的聚类中心（暴力 L2，分块控制内存）。
    返回每个描述子的簇编号 (N,)。
    """
    cb = codebook.astype(np.float32)
    # 距离 = |x|^2 - 2 x·c + |c|^2，|x|^2 对 argmin 无影响，可省
    cb_sq = np.sum(cb ** 2, axis=1)  # (K,)
    labels = np.empty(desc.shape[0], dtype=np.int32)
    for start in range(0, desc.shape[0], chunk):
        block = desc[start:start + chunk]
        dist = cb_sq[None, :] - 2.0 * block @ cb.T
        labels[start:start + chunk] = np.argmin(dist, axis=1)
    return labels


def _vlad_from_descriptors(desc, cfg):
    """描述子矩阵 -> 归一化后的 VLAD 向量"""
    K = int(cfg["K"])
    codebook = _get_codebook(cfg)
    dim = codebook.shape[1]                      # 维度以码本为准，与 config 字段解耦
    vlad_dim = K * dim

    if desc.shape[0] == 0:
        # 极端情况：白墙等完全无纹理，返回零向量（检索时相似度自然很低）
        return np.zeros(vlad_dim, dtype=np.float32)

    labels = _assign_to_codebook(desc, codebook)
    resid = desc - codebook[labels]          # 每个描述子相对所属中心的残差
    # 按簇累加残差（np.add.at 对二维广播支持有问题，这里用循环，K 很小）
    vlad = np.zeros((K, dim), dtype=np.float32)
    for k in range(K):
        mask = labels == k
        if mask.any():
            vlad[k] = resid[mask].sum(axis=0)
    if cfg.get("intra_norm", False):
        # 簇内 L2 归一化：防止描述子多的簇主导整个向量（Arandjelovic & Zisserman 2013）
        norms = np.linalg.norm(vlad, axis=1, keepdims=True)
        vlad = vlad / (norms + 1e-9)
    vlad = vlad.reshape(-1)

    return _normalize(vlad, cfg)


def _normalize(vec, cfg):
    """
    VLAD 归一化：先 Power-law（逐元素开方，抑制大值主导），再全局 L2。
    这一步是经典 VLAD 的关键，漏掉会让检索性能明显下降。
    """
    mode = str(cfg.get("normalization", "power_law + L2")).lower()
    out = vec.astype(np.float32)
    if "power" in mode:
        out = np.sign(out) * np.sqrt(np.abs(out))
    norm = np.linalg.norm(out)
    if norm > 0:
        out = out / norm
    return out


# ------------------------------------------------------------------
# 对外接口 1：单图编码
# ------------------------------------------------------------------
def encode_single_image(image_path, config):
    """
    输入：一张图片路径 + config(dict 或 json 路径)
    输出：一维 VLAD 向量，shape=(vlad_dim,)，已做 Power-law + L2
    """
    cfg = load_config(config)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"读不到图片: {image_path}")
    desc = _extract_descriptors(image, cfg)
    return _vlad_from_descriptors(desc, cfg)


# ------------------------------------------------------------------
# 对外接口 2：批量编码
# ------------------------------------------------------------------
def encode_batch(image_paths, config, verbose=False):
    """
    输入：图片路径列表 + config
    输出：N x vlad_dim 矩阵（顺序与输入列表严格一致，索引与 meta 靠这个对齐）
    """
    cfg = load_config(config)
    detector = _make_detector(cfg)          # 复用检测子，省去重复构造开销
    rows = []
    for i, p in enumerate(image_paths):
        image = cv2.imread(p, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"读不到图片: {p}")
        desc = _extract_descriptors(image, cfg, detector=detector)
        rows.append(_vlad_from_descriptors(desc, cfg))
        if verbose and (i + 1) % 50 == 0:
            print(f"  encoded {i + 1}/{len(image_paths)}")
    if not rows:
        return np.empty((0, int(cfg["K"]) * int(cfg["descriptor_dim"])), dtype=np.float32)
    return np.vstack(rows)


# ------------------------------------------------------------------
# 对外接口 3：局部特征几何验证（VLAD 粗筛之后的精排用）
# ------------------------------------------------------------------
def verify_match(query_path, candidate_path, config, min_inliers=15):
    """
    对两张图做局部特征点匹配 + RANSAC 几何验证。

    用途：VLAD 是全局描述子，两个结构相似的楼梯间/休息区向量很接近，
    仅靠 VLAD 距离无法区分"看起来相似"和"确实是同一位置"。
    本函数逐点比较局部纹理，并用 RANSAC 检验匹配点是否满足同一空间变换，
    内点越多说明两图越可能拍摄自同一位置。

    输入：
        query_path     查询图路径
        candidate_path 候选地图帧路径
        config         dict 或 json 路径（只用其中的 feature / nfeatures / frame_size）
        min_inliers    判定"验证通过"的内点阈值
    输出：
        (inliers, passed)  inliers 为 RANSAC 内点数，passed 为是否通过
    """
    cfg = load_config(config)
    feature = cfg["feature"].upper()
    nfeat = int(cfg.get("nfeatures", 2000))
    w, h = int(cfg["frame_size"][0]), int(cfg["frame_size"][1])

    # SIFT 用浮点描述子 + L2 距离，ORB 用二进制描述子 + Hamming 距离
    if feature in ("SIFT", "ROOTSIFT"):
        detector = cv2.SIFT_create(nfeatures=nfeat)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=nfeat)
        norm = cv2.NORM_HAMMING

    kp_des = []
    for p in (query_path, candidate_path):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            return 0, False
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = detector.detectAndCompute(gray, None)
        if des is None or len(kp) < 2:
            return 0, False
        kp_des.append((kp, des))

    (kp1, des1), (kp2, des2) = kp_des
    bf = cv2.BFMatcher(norm, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test：保留最近邻明显优于次近邻的匹配，去掉歧义匹配
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
    if len(good) < 10:  # 匹配点太少，无法做稳定的几何估计
        return 0, False

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # RANSAC 求单应矩阵：只有符合同一空间变换的匹配才算内点
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return inliers, inliers > min_inliers


# ------------------------------------------------------------------
# 对外接口 4：两阶段定位（VLAD 粗筛 + 几何验证精排）
# ------------------------------------------------------------------
def locate_image(image_path, config, index, meta, frames_dir,
                 top_k=10, min_inliers=15, use_verification=True):
    """
    完整的单图定位流程，供检索端 / Gradio 界面直接调用。

    流程：
        1. encode_single_image 得到 query VLAD 向量
        2. FAISS 检索 top_k 候选地图帧
        3.（可选）对每个候选做 verify_match，按 RANSAC 内点数重排
           —— 用于区分 VLAD 距离相近的相似场景（如两段楼梯、两个休息区）
        4. 返回最终 top 结果

    输入：
        image_path   查询图路径
        config       dict 或 json 路径
        index        已加载的 faiss 索引（调用方负责加载，避免每次重复读盘）
        meta         与索引行序一致的 meta 列表，每行至少含 filename / region_id
        frames_dir   地图帧图片所在目录（几何验证需要读原图）
        top_k        粗筛候选数
        min_inliers  几何验证内点阈值
        use_verification  False 时退化为纯 VLAD 单阶段检索
    输出：
        dict:
          region      最终判定区域（重排后第 1 名的 region_id）
          confidence  top1 相似度，1 - D/2（D 为 faiss 返回的平方 L2 距离，范围 -1~1）
          candidates  列表，每项含 filename / region_id / similarity / inliers
    """
    import faiss  # 局部导入：只用编码功能时不必安装 faiss

    cfg = load_config(config)
    q = encode_single_image(image_path, cfg).reshape(1, -1)
    D, I = index.search(q, top_k)

    candidates = []
    for dist, j in zip(D[0], I[0]):
        if j < 0 or j >= len(meta):
            continue
        candidates.append({
            "filename": meta[j]["filename"],
            "region_id": meta[j]["region_id"],
            # faiss 的 IndexFlatL2 返回的是平方 L2 距离 D；向量已 L2 归一化，
            # 因此余弦相似度 = 1 - D/2（不要再平方一次）
            "similarity": float(1.0 - dist / 2.0),
            "inliers": -1,
        })

    if use_verification and candidates:
        for c in candidates:
            c["inliers"], _ = verify_match(
                image_path, os.path.join(frames_dir, c["filename"]), cfg,
                min_inliers=min_inliers)
        # 内点数降序；内点相同则保持 VLAD 距离序（稳定排序）
        candidates.sort(key=lambda c: c["inliers"], reverse=True)

    return {
        "region": candidates[0]["region_id"] if candidates else None,
        "confidence": candidates[0]["similarity"] if candidates else 0.0,
        "candidates": candidates,
    }


if __name__ == "__main__":
    # 简单自测：给一张图和一个 config，打印向量形状与范数
    import sys
    if len(sys.argv) >= 3:
        v = encode_single_image(sys.argv[1], sys.argv[2])
        print("vlad_dim =", v.shape[0], " L2 =", float(np.linalg.norm(v)))
