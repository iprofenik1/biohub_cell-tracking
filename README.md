# 第 3 部分：完整推理管线

这里是最终提交所用的 Kaggle 推理 notebook，教学版逐段加了中文原理注释。

| 文件 | 说明 |
|---|---|
| `biohub_final_inference.ipynb` | 直接上传到 Kaggle 运行的 notebook：1 个 Markdown 说明单元 + 1 个代码单元 |
| `biohub_final_inference.py` | 与 notebook 代码单元逐字相同的脚本形式；第 1、2 部分的训练数据导出脚本读取的就是这个文件 |
| `kernel-metadata.example.json` | 用 Kaggle API 推送时的元数据模板：复制为同目录下的 `kernel-metadata.json`，把 `<你的 Kaggle 用户名>` 换成自己的，再运行 `kaggle kernels push -p part3_inference_pipeline` |

---

## 1. 流程总览

```
测试影像 (T, Z, Y, X) = (100, 64, 256, 256)
   │
   ├─ 第 0–3 段  配置、配置守卫、路径常量、离线安装依赖、物化推理代码与权重、SHA256 校验
   │
   ├─ 第 4 段    给支持包里的推理脚本打补丁，然后在 GPU 子进程里逐段影像运行：
   │               1) 检测：时序 3D U-Net（8 视角 TTA + 双模型融合）→ 中心热图 → 峰值（阈值 0.965）
   │               2) 坐标修正：10 折坐标头给每个峰预测亚体素位移（10 个头取平均）
   │               3) 关联：节点 Transformer 给相邻帧细胞对打连接概率（正反双向调和）
   │               4) ILP：整段影像全局求解，决定保留哪些节点与边
   │             每段影像输出一个图 (.geff)，同时缓存低分峰与全部候选对的连接概率
   │
   ├─ 第 5 段    定义后处理 filter_output_graph：
   │               边过滤 → 重链接 → 找回 → 再重链接 → 缺口闭合 / 两帧缺口 / 低分峰补缺
   │               → 安全分裂 + 分裂补全打分器 → 删除孤立节点与短轨迹 → 线拟合平滑
   │             段末先用“新模块全部关闭”的配置跑第一遍，写出参照结果
   │
   ├─ 第 6–10 段 第一遍输出的审计；本地验证器与参数扫描（最终版已关闭，只定义函数）
   │
   └─ 第 11–12 段 打开全部新模块跑最终一遍，覆盖 submission.csv；打印实际生效的运行清单
```

文件开头的注释列出了我们在公开基线之上的全部改动（候选概率重链接、全局位移估计、跳变感知平滑、
关联特征修正、找回概率打分、分裂补全打分器、两个参数、10 折坐标头、修复截止时间），以及每个改动的原理。

## 2. 在 Kaggle 上运行

1. 新建 notebook，用 *File → Import Notebook* 导入 `biohub_final_inference.ipynb`（或用 `kaggle kernels push`，
   元数据参考模板）。
2. 加速器选 **GPU T4 ×2**，**关闭网络**（代码竞赛要求离线；依赖从挂载的数据集里离线安装）。
3. 挂载输入（*Add Input*）：
   - 竞赛数据 `biohub-cell-tracking-during-development`（需要先在竞赛页面加入比赛）
   - `pilkwang/biohub-tracking-support-pack-50ep-v1`：主模型权重、推理代码、离线 wheel 包
   - `pilkwang/biohub-temporal-unet3d-seed314159-v1`：副模型（另一个随机种子）权重
   - `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`：DeepCenter 中心先验模型
   - 你自己的数据集 `biohub-cv10-coord-head`：第 1 部分训练出的 `fold0.pt … fold9.pt`
     （第 4 段按 `biohub-cv10-coord-head/fold*.pt` 查找，必须恰好 10 个，所以数据集名不要改；也可以改挂第 1 部分
     step3 那个 notebook 的输出，见第 1 部分 README 第 4 节，但两者只能挂一个）
4. *Save Version → Save & Run All*。保存运行只处理 4 段可见测试影像（GPU 推理部分约 10 分钟），用来确认整条管线能跑通；
   日志里应能看到 `mode = candidate`（坐标修正已启用）。
5. 提交：在这个 notebook 版本的 *Output* 里选中 `submission.csv` 点 *Submit*（或在竞赛页 *Submit Prediction*
   里选这个 notebook 和版本）。Kaggle 会把测试目录换成隐藏测试集，重新运行整个 notebook 后计分。
   比赛已经结束，这时的提交是 *Late Submission*，仍会给出公榜和私榜分数。

**还没有自己的坐标头时（只为先跑通流程）：** 在 Kaggle 编辑器里的 notebook 副本中，把第 4 段的
`if len(_myhead) != 10:` 改成 `if False:`，把 `os.environ['V1284_MODE']='candidate'` 改成
`os.environ['V1284_MODE']='zero'`。坐标修正头这时不读头文件、不修正坐标（连公开坐标头那一步修正也没有），分数一般会低一些。
这两处正是第 2 部分 step1 加 `--coord-head-mode zero` 时打的补丁。只改 Kaggle 上的副本，不要改仓库里的
`biohub_final_inference.py`：两个导出脚本按整行原文定位补丁（见第 4 节）。

三个 `pilkwang/...` 是社区公开的数据集。权重和推理代码都有 SHA256 校验，版本不对会在第 3 段直接报错，
而不是悄悄跑出一个不同的结果——隐藏测试集上的重跑我们看不到日志，硬失败比静默出错更安全。

**运行环境版本。** 第 3 段从支持包里离线安装的是固定版本的 wheel，但 torch / CUDA 等来自 Kaggle 的 notebook
镜像，而镜像会不定期更新。我们在比赛期间（2026 年 9 月）的镜像上运行。如果在新镜像上第 3 段安装失败或推理
报错，先看报错是哪个包；能跑通之后，可以在 notebook 的 *Settings → Environment* 里选择固定环境
（Pin to original environment），以免之后的镜像更新再改变结果。

可见的 4 段测试影像其实也是训练影像，所以保存运行（Save & Run All）时会多跑一遍诊断；提交后在隐藏测试集上
重跑时自动跳过。整个隐藏集的运行在 12 小时限制内完成（修复截止 11.5 小时）。

## 3. 读代码的建议顺序

| 想了解 | 看哪里 |
|---|---|
| 所有超参数及其含义 | 第 0 段（`BIOHUB_*` 环境变量，每个都有中文注释） |
| 检测 TTA、双模型融合、坐标头插入点、候选概率缓存 | 第 4 段（注意：补丁字符串里的文本会写进推理脚本，注释只能写在字符串外面） |
| 坐标头本身 | 第 4 段内嵌的模块源码 = `../part1_cv10_coord_head/coord_head_module.py` |
| 后处理每一步的原理 | 第 5 段，从 `filter_output_graph` 开始按调用顺序读 |
| 分裂补全打分器 | 第 5 段内嵌的 `_D1_SOURCE` = `../part2_d1_division_scorer/d1_module.py` |
| 两遍后处理、诊断与输出 | 第 11 段 |

## 4. 与第 1、2 部分的关系

- 第 1 部分（10 折坐标头）训练出的 `fold*.pt` 通过数据集 `biohub-cv10-coord-head` 被第 4 段加载。
- 第 2 部分（分裂补全打分器）训练出的 `MODEL = {...}` 是 `_D1_SOURCE` 里的一行；`step2_train_d1.py`
  打印的那一行可以整行替换它（见第 2 部分 README 第 7 节）。
- 两部分的训练数据导出脚本（各自的 `step1_*.py`）都通过 `common/pipeline_runner.py` 读取本目录的
  `biohub_final_inference.py`，只改几行（把测试目录指向训练影像、切换坐标头模式等）就在训练影像上
  复用整条管线，保证训练数据与推理时看到的数据来自同一套代码。所以请不要修改本目录代码中的代码行
  （注释可以随意改）；导出脚本按“整行原文”定位补丁位置，原文变了会直接报错提示。唯一的例外是第 2 部分
  README 第 7 节的整行替换 `MODEL = {...}`（以及同步的 `D1_THRESHOLD`、`_D1_SHA`），这几行都不是补丁锚点。

## 5. 教学版与最终运行版本的关系

教学版只添加了注释和说明文字，代码行为不变：

- 用 AST（抽象语法树）逐节点比较：除下面列出的字面量外，教学版与最终运行的 notebook 完全相同；
- 在 4 段可见影像上分别运行两个版本，输出的 `submission.csv` 逐字节相同。

字面量的改动只有：坐标头数据集名（改为 `biohub-cv10-coord-head`）、坐标头数量检查的报错信息（改为中文）、
一个报告字段名、最终一遍的运行标签（`final_all_modules`，只写进 run_stats.csv），以及两段内嵌模块源码（坐标头、分裂补全打分器）换成了带中文注释、语法树等价的版本
（分裂补全打分器源码的哈希 `_D1_SHA` 随之更新，它只写进运行报告）。

代码里保留了一些历史命名（`x138_*` 文件名、`_G1X1_ACTIVE` 等开关、`d1_pw10c_final` 等标签）。
改名会牵动补丁锚点和报告格式，所以没有改，含义见仓库根目录 README 的术语表。
`print` 里出现的 “public …” 等字样是公开流水线作者留下的历史标签，与本方案的成绩无关。
