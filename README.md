# 室内视觉定位系统（Visual Place Recognition）

本项目实现“只用一张照片判断当前位于教学楼哪个区域”。系统不使用 GPS、里程计或任何外部定位信号，而是先把探索视频转换成可检索的视觉地图，再把查询照片与地图帧进行匹配，输出区域标签、候选帧和相似度。

项目当前包含两种检索方式：

- 单阶段：ORB/SIFT/RootSIFT 局部描述子 → VLAD 全局向量 → FAISS 最近邻检索。
- 两阶段：在 VLAD 粗筛之后，对候选帧做局部特征点匹配与 RANSAC 几何验证，再按内点数重排。该阶段主要用于区分结构相似的楼梯、门口和休息区。

## 1. 快速使用

### 1.1 环境

项目使用 Anaconda 环境 `vpr`，主要依赖：

- Python 3.10
- OpenCV（`cv2`）：特征提取、图像读取、特征匹配、RANSAC
- NumPy：VLAD 向量与矩阵计算
- FAISS（`faiss-cpu`）：地图向量索引与检索
- scikit-learn：K-Means 码本训练
- Matplotlib：消融曲线与混淆矩阵
- Gradio：网页交互界面

### 1.2 启动网页系统

在项目根目录执行：

```powershell
conda activate vpr
python app.py --config results/vlad_config_ORB_K32.json --strategy two_stage
```

也可以在 base 环境中直接执行：

```powershell
conda run -n vpr python app.py --strategy two_stage
```

或双击 `启动系统.bat`。启动后打开：

```text
http://127.0.0.1:7860
```

服务进程必须保持运行；关闭终端或按 `Ctrl+C` 后网页也会停止。索引、码本和区域表只在启动时加载一次。

### 1.3 命令行定位一张图片

```powershell
python src/retrieve.py data/query/same/R0_1.jpg two_stage
```

可选策略：

- `single`：直接使用 VLAD 检索的 top1。
- `vote`：对 top5 候选进行距离加权投票。
- `two_stage`：VLAD 粗筛 top10，再用几何验证重排，默认推荐。

### 1.4 复现流水线

```powershell
# 视频抽帧与预处理
python src/extract_frames.py

# 训练码本、编码地图帧、构建 FAISS 索引
python src/build_index.py

# 第一阶段：自检、K 值消融、特征消融、混淆矩阵
python src/run_experiments.py

# 第二阶段：201 张查询图上的 single/vote/two_stage 对比
python src/run_stage2.py
```

注意：重建索引会覆盖 `results/` 中的同名产物；地图帧、码本、索引和 meta 必须来自同一轮构建。

## 2. 区域定义

| ID | 英文名称 | 含义 |
|---|---|---|
| R0 | entrance_stairs | 入口楼梯 |
| R1 | library_right_door | 图书室右门 |
| R2 | west_corridor | 西侧走廊 |
| R3 | library_left_door | 图书室左门 |
| R4 | exit_stairs | 出口楼梯 |
| R5 | southeast_lounge | 东南休息处 |
| R6 | east_corridor | 东侧走廊 |
| R7 | northeast_lounge | 东北休息处 |

区域 ID 以 `data/meta/regions.csv` 为唯一标准。`frames_meta.csv`、查询 meta、索引 meta 中的 ID 必须完全一致。

## 3. 目录结构与文件作用

```text
hw_data/
├─ app.py                         Gradio 网页系统入口
├─ vlad_encoder.py                核心算法模块：编码、几何验证、两阶段定位
├─ 启动系统.bat                   Windows 一键启动网页服务
├─ README.md                      项目说明与文件导航（本文件）
├─ 技术文档.md                    算法原理及与目标检测项目的区别
├─ 建图交接验收报告.txt            第一阶段建图、验收和消融实验记录
├─ 实验补充报告_第二阶段.txt       两阶段检索与 201 张查询图实验记录
├─ 交接验收报告_汇报解读.md        第一阶段结果的汇报提纲
│
├─ data/
│  ├─ video/                      原始探索视频 E1_1.mp4、E1_2.mp4
│  ├─ frames/                     建图帧，统一为 640×480，共 301 张入库
│  ├─ query/
│  │  └─ same/                    原始基准查询图 24 张，每区域 3 张
│  ├─ query_new/                  后续补拍查询图，共 177 张，按 R0~R7 分目录
│  │  ├─ R0/ ... R7/              命名如 R0_001.jpg；R6 没有补拍图
│  │  └─ query_new_meta.csv       补拍图的 region_id、category 和原始文件名
│  └─ meta/
│     ├─ regions.csv              8 个区域的标准定义
│     ├─ frames_meta.csv          地图帧与视频、时间戳、区域标签的对应关系
│     └─ query_meta.csv           原 24 张查询图的标注
│
├─ src/
│  ├─ extract_frames.py           视频按 0.5 秒抽帧、缩放、记录时间戳
│  ├─ build_index.py              收集地图描述子、训练 K-Means、编码并建索引
│  ├─ run_experiments.py          第一阶段自检、K 消融、特征消融与绘图
│  ├─ retrieve.py                 Locator 在线检索封装，供命令行和 Gradio 调用
│  ├─ run_stage2.py               第二阶段三策略实验、混淆矩阵和错误清单
│  ├─ label_queries.py            查询图标注辅助脚本
│  ├─ _try_intra.py               簇内归一化探索实验，不是主流程依赖
│  ├─ _try_orb256.py              ORB 参数探索实验，不是主流程依赖
│  └─ _final_check.py             索引/meta/config 一致性检查
│
├─ results/
│  ├─ codebook_ORB_K16.npy        K=16 的 ORB 码本
│  ├─ codebook_ORB_K32.npy        K=32 的 ORB 码本
│  ├─ codebook_ORB_K64.npy        K=64 的 ORB 码本
│  ├─ codebook_ORB_K128.npy       K=128 的 ORB 码本
│  ├─ codebook_SIFT_K32.npy       SIFT 对照码本
│  ├─ codebook_ROOTSIFT_K32.npy   RootSIFT 对照码本
│  ├─ map_ORB_K16.index 等        对应配置的 FAISS FlatL2 索引
│  ├─ map_ORB_K16_meta.csv 等     索引行号到地图帧/区域的映射，顺序不可打乱
│  ├─ vlad_config_ORB_K16.json 等 特征、K、维度、码本路径和归一化配置
│  ├─ selfcheck.csv               跨区域查询的 top1 自检结果
│  ├─ exp_K_ablation.csv/.png     第一阶段 K 值消融数据与曲线
│  ├─ exp_feature_ablation.*      ORB/SIFT/RootSIFT 对照数据与图
│  ├─ exp_confusion_matrix.*      第一阶段区域混淆矩阵
│  ├─ exp_stage2_strategy.csv     第二阶段三种策略的精度与耗时
│  ├─ exp_stage2_confusion.*      两阶段策略混淆矩阵与热力图
│  └─ exp_stage2_errors.csv       两阶段错误样本及 top1 候选、内点数
│
├─ 补拍/                          补拍时的原始手机照片，保留作溯源
└─ 补拍_group01.jpg ~ group18.jpg  人工标注时使用的缩略图拼图，不参与算法
```

`__pycache__/` 是 Python 自动生成的缓存，可删除，不影响运行。

## 4. 核心代码接口

### 4.1 VLAD 编码

```python
import vlad_encoder as ve

cfg = "results/vlad_config_ORB_K32.json"
v = ve.encode_single_image("any.jpg", cfg)       # shape: (1024,)
vs = ve.encode_batch(["a.jpg", "b.jpg"], cfg)   # shape: (2, 1024)
```

`encode_single_image` 和 `encode_batch` 只负责把图片转换为 VLAD 向量，不负责加载 FAISS 索引。

### 4.2 几何验证

```python
inliers, passed = ve.verify_match(
    "query.jpg", "data/frames/E1_1_f0002.jpg", cfg
)
```

返回 RANSAC 内点数和是否超过阈值。它比较的是两张图的局部纹理与空间一致性，而不是图像类别。

### 4.3 在线定位

推荐使用 `src/retrieve.py` 中的 `Locator`，避免每次查询都加载索引：

```python
from src.retrieve import Locator

locator = Locator("results/vlad_config_ORB_K32.json")
result = locator.locate("query.jpg", strategy="two_stage")
print(result["region"], result["region_name"])
print(result["candidates"][:5])
```

## 5. 当前实测结果

第一阶段在 24 张基准查询图上，ORB 的 K 值消融结果为：

| K | Recall@1 | Recall@5 |
|---|---|---|
| 16 | 0.500 | 0.708 |
| 32 | 0.500 | 0.792 |
| 64 | 0.458 | 0.750 |
| 128 | 0.458 | 0.833 |

第二阶段在 201 张查询图上，使用 ORB、K=32：

| 策略 | Recall@1 | 粗筛 Recall@5 | 中位耗时 |
|---|---|---|---|
| single | 0.413 | 0.756 | 87 ms |
| vote | 0.353 | 0.756 | 87 ms |
| two_stage | 0.557 | 0.756 | 1150 ms |

两阶段把 Recall@1 提高约 14.4 个百分点，但需要逐候选做特征匹配，速度明显慢于纯 VLAD。详细限制和错误分析见 `实验补充报告_第二阶段.txt`。

## 6. 数据与实验注意事项

1. 查询照片必须独立拍摄，不能从建图视频抽帧，否则会产生数据泄露。
2. K-Means 码本只能使用地图帧描述子训练，不能加入查询图。
3. 索引向量顺序必须与 `map_*_meta.csv` 行序一致。
4. 所有图像进入特征提取前统一缩放到 640×480。
5. VLAD 必须进行 Power-law 与 L2 归一化。
6. R6 目前只有原基准集的 3 张查询图，补拍集没有 R6，相关统计不充分。
7. 光照和视角专项分组尚未形成完整的 controlled query 集，当前 201 张结果不能直接当作光照/视角消融结论。
