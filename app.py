# -*- coding: utf-8 -*-
"""
app.py —— 室内视觉定位系统 Gradio 界面
========================================
上传一张照片，返回：区域判定 + top-k 候选帧缩略图 + 相似度 + 几何验证内点数。

启动：
    conda activate vpr
    python app.py --config results/vlad_config_ORB_K32.json --strategy two_stage
然后浏览器打开 http://127.0.0.1:7860

索引/码本/区域表在启动时只加载一次，之后每次上传即拍即算。
"""
import argparse
import os

import gradio as gr

from src.retrieve import Locator, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "results", "vlad_config_ORB_K32.json"))
    parser.add_argument("--strategy", default="two_stage", choices=["single", "vote", "two_stage"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    locator = Locator(args.config)
    strategy_state = gr.State(args.strategy)

    def predict(image, strategy):
        if image is None:
            return "请先上传照片", "", None, ""
        # gr.Image(type="filepath") 直接给临时文件路径
        res = locator.locate(image, strategy=strategy, top_k=args.top_k)
        region_text = f"## 判定区域：{res['region']}（{res['region_name']}）"
        info = (f"- 策略：{strategy}\n"
                f"- top1 相似度：{res['confidence']:.3f}\n"
                f"- 耗时：{res['elapsed_ms']} ms")
        # 候选缩略图 + 标注
        gallery = []
        for c in res["candidates"][:5]:
            p = os.path.join(locator.frames_dir, c["filename"])
            if os.path.exists(p):
                caption = f"{c['region_id']}  sim={c['similarity']:.3f}"
                if c["inliers"] >= 0:
                    caption += f"  inliers={c['inliers']}"
                gallery.append((p, caption))
        table = "排名 | 帧文件 | 区域 | 相似度 | 内点\n---|---|---|---|---\n"
        for i, c in enumerate(res["candidates"][:5], 1):
            table += (f"{i} | {c['filename']} | {c['region_id']} | "
                      f"{c['similarity']:.3f} | {c['inliers']}\n")
        return region_text, info, gallery, table

    with gr.Blocks(title="室内视觉定位系统") as demo:
        gr.Markdown("# 室内视觉定位系统\n"
                    "上传一张教室/走廊照片，系统基于 VLAD 全局检索 + ORB 几何验证，"
                    "判定你所在的区域（R0~R7）。")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="filepath", label="上传查询照片")
                strategy = gr.Radio(["two_stage", "single", "vote"], value=args.strategy,
                                    label="检索策略（two_stage=VLAD+几何验证，single=纯VLAD，vote=加权投票）")
                btn = gr.Button("开始定位", variant="primary")
                region_out = gr.Markdown()
                info_out = gr.Markdown()
            with gr.Column(scale=1):
                gallery_out = gr.Gallery(label="Top-5 候选地图帧", columns=5, height=260)
                table_out = gr.Markdown()
        btn.click(predict, inputs=[inp, strategy], outputs=[region_out, info_out, gallery_out, table_out])

    demo.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
