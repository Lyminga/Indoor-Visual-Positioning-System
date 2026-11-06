# -*- coding: utf-8 -*-
"""
视频抽帧脚本（A 阶段 3.1）
------------------------------------------------------------
输入 : data/video/*.mp4
输出 : data/frames/*.jpg          统一 resize 到 640x480
       data/meta/frames_meta.csv  每帧对应的 视频名 / 时间戳 / region_id
       data/meta/frames_dropped.csv 被过滤掉的坏帧（模糊 / 过曝）

规则（与组内约定一致）:
  * 固定间隔 0.5s 抽一帧（2fps）
  * 每 10 秒对应一个 region，按 regions.csv 的 R0..R7 顺序
  * 视频共 81s，最后 1s 不计入（只保留 t < 80s 的帧）
  * 严重运动模糊 / 严重过曝的帧不入库，但记录在 dropped csv 里，便于报告说明
"""
import csv
import os
import sys

import cv2
import numpy as np

# ------------------------- 可配置参数 -------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(ROOT, "data", "video")
FRAME_DIR = os.path.join(ROOT, "data", "frames")
META_DIR = os.path.join(ROOT, "data", "meta")

SAMPLE_INTERVAL = 0.5        # 抽帧间隔（秒）
REGION_DURATION = 10.0       # 每个区域占用时长（秒）
MAX_TIME_SEC = 80.0          # 81s 视频丢掉最后 1s
FRAME_SIZE = (640, 480)      # 统一尺寸 (宽, 高)

BLUR_THRESHOLD = 12.0        # 拉普拉斯方差低于该值视为严重模糊
OVEREXPOSE_RATIO = 0.75      # 亮度>250 的像素占比超过该值视为严重过曝

# 两段视频的区域遍历顺序（经组员现场确认）。
# E1_1 从 R0 正向出发，沿 R0->R7 走一圈；
# E1_2 起点在 R7（东北休息处），沿反方向走完整条环线，终点回到 R0。
VIDEO_REGION_ORDER = {
    "E1_1.mp4": ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"],
    "E1_2.mp4": ["R7", "R6", "R5", "R4", "R3", "R2", "R1", "R0"],
}


def read_regions():
    """读取 regions.csv，拿到合法的 region_id 集合，用于后续一致性校验"""
    path = os.path.join(META_DIR, "regions.csv")
    with open(path, "r", encoding="utf-8-sig") as f:
        return [row["region_id"] for row in csv.DictReader(f)]


def region_of_time(t, order):
    """根据时间戳 t 推断所属区域：每 REGION_DURATION 秒一个区域"""
    idx = int(t // REGION_DURATION)
    if idx >= len(order):
        return None  # 超出 8 个区域范围（应已被 MAX_TIME_SEC 拦住）
    return order[idx]


def quality(img_bgr):
    """返回 (是否合格, 模糊分数, 过曝比例)，用于过滤坏帧"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())  # 越大越清晰
    over_ratio = float(np.mean(gray > 250))                    # 过曝像素占比
    ok = (blur_score >= BLUR_THRESHOLD) and (over_ratio <= OVEREXPOSE_RATIO)
    return ok, blur_score, over_ratio


def extract_one(video_path, order, kept_rows, dropped_rows):
    """对单个视频抽帧，写 jpg 并累计 meta 行"""
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] 无法打开视频:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps if fps > 0 else 0
    print(f"\n[VIDEO] {video_name}  fps={fps:.2f}  frames={total}  duration={duration:.2f}s")

    next_target = 0.0      # 下一个要抽帧的时间点
    frame_idx = 0          # 当前读取到的帧号
    seq = 0                # 该视频内成功抽帧的连续序号
    kept = dropped = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / fps
        if t + 1e-6 < next_target:
            frame_idx += 1
            continue

        # 到达抽帧时间点（容差约半个采样间隔，避免帧率不是整数导致跳过）
        if t >= MAX_TIME_SEC:
            break

        seq += 1
        name = f"{video_name.replace('.mp4', '')}_f{seq:04d}.jpg"
        region_id = region_of_time(t, order)

        frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        ok, blur_score, over_ratio = quality(frame)

        if ok:
            cv2.imwrite(os.path.join(FRAME_DIR, name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            kept_rows.append({
                "filename": name,
                "video": video_name,
                "time_sec": round(t, 2),
                "region_id": region_id,
            })
            kept += 1
        else:
            dropped_rows.append({
                "filename": name,
                "video": video_name,
                "time_sec": round(t, 2),
                "region_id": region_id,
                "blur_score": round(blur_score, 1),
                "overexpose_ratio": round(over_ratio, 3),
                "reason": "blur" if blur_score < BLUR_THRESHOLD else "overexposed",
            })
            dropped += 1

        next_target += SAMPLE_INTERVAL
        frame_idx += 1

    cap.release()
    print(f"        保留 {kept} 帧，丢弃 {dropped} 帧")


def main():
    os.makedirs(FRAME_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)
    valid_regions = set(read_regions())

    videos = sorted(v for v in os.listdir(VIDEO_DIR) if v.lower().endswith((".mp4", ".mov")))
    if not videos:
        print("[ERROR] data/video 下没有视频文件")
        sys.exit(1)

    kept_rows, dropped_rows = [], []
    for v in videos:
        order = VIDEO_REGION_ORDER.get(v)
        if order is None:
            print(f"[WARN] 未配置 {v} 的区域顺序，默认按正向 R0..R7 处理")
            order = ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]
        extract_one(os.path.join(VIDEO_DIR, v), order, kept_rows, dropped_rows)

    # 硬性校验：region_id 必须非空且属于 regions.csv
    for row in kept_rows + dropped_rows:
        assert row["region_id"] in valid_regions, f"非法 region_id: {row}"

    kept_csv = os.path.join(META_DIR, "frames_meta.csv")
    with open(kept_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "video", "time_sec", "region_id"])
        w.writeheader()
        w.writerows(kept_rows)

    dropped_csv = os.path.join(META_DIR, "frames_dropped.csv")
    with open(dropped_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["filename", "video", "time_sec", "region_id",
                        "blur_score", "overexpose_ratio", "reason"],
        )
        w.writeheader()
        w.writerows(dropped_rows)

    print(f"\n[DONE] 地图帧共 {len(kept_rows)} 张 -> {kept_csv}")
    print(f"[DONE] 丢弃帧共 {len(dropped_rows)} 张 -> {dropped_csv}")

    # 每区域帧数统计，方便检查视频覆盖是否均匀
    from collections import Counter
    cnt = Counter(r["region_id"] for r in kept_rows)
    print("[STAT] 每区域帧数:", dict(sorted(cnt.items())))


if __name__ == "__main__":
    main()
