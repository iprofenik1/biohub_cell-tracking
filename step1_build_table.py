"""第 2 部分 · 第 1 步（Kaggle GPU）：在训练影片上跑完整推理管线，导出分裂补全打分器的候选与特征，并用 GT 打标签。

用法（Kaggle notebook，加速器选 GPU T4 ×2；挂载与推理 notebook 相同的数据集；把本仓库放到 /kaggle/working
下并 cd 到仓库根目录）::

    !python part2_d1_division_scorer/step1_build_table.py --shard 0 --num-shards 4
    !python part2_d1_division_scorer/step1_build_table.py --shard 0 --num-shards 4 --dry-run   # 只检查补丁

每个 Kaggle 会话跑一个分片（--shard 0 … 3），把 ``/kaggle/working/d1_table`` 保存为 notebook 输出（或数据集），
step2 再把所有分片的 ``final/`` 目录一起读进来。

为什么要在“完整管线”上导出，而且是在“最终一遍”里导出
======================================================
分裂补全打分器的特征几乎全部取决于它在管线里**看到的那张图**：候选 = “只有一个子节点的母细胞 P” ×
“下一帧没有父亲的轨迹起点 B”，而哪些节点是起点、母细胞前后的轨迹有多长、局部运动场是多少，都由前面的
重链接、找回、缺口填补和安全分裂决定。训练表里的图只要和推理时差一点（少一个找回节点、多一条重链接的边），
特征分布就会偏。所以这里不另写“提特征”的代码，而是**一字不差地执行第三部分的推理脚本**，只做最小的改动：

1. ``TEST_DIR`` 指向训练影片子集（符号链接目录），管线把训练影片当作测试影片来推理；
2. 只执行第 0–5 段：第 0–4 段完成检测、坐标修正（10 折坐标头）、关联和 ILP，产出原始图、低分峰 dump
   和候选概率缓存；第 5 段定义全部后处理函数、加载 DeepCenter、安装分裂补全打分器；
3. 第 5 段最后一句 ``write_test_submission("base")``（立刻跑“第一遍”后处理并写 submission.csv）换成 ``pass``——
   制作训练表不需要第一遍，训练影片的 submission.csv 更不能留下来；
4. 然后在同一个命名空间里按第 11 段**最终一遍**的方式打开 5 个模块开关：候选概率重链接、全局位移估计 +
   跳变感知平滑、关联特征修正、找回概率打分、分裂补全打分器（``_DR_ACTIVE`` / ``_G1X1_ACTIVE`` /
   ``_D4_ACTIVE`` / ``_RR_ACTIVE`` / ``_D1_ACTIVE`` 全部为 True）。最终一遍里的 ``pp_apply(selected_config)``
   在最终版本中是空操作（本地验证器关闭，``selected_config = {}``），第 6–10 段也没有改动后处理用到的任何
   函数或常量，所以“第 0–5 段 + 5 个开关”就是最终一遍的后处理状态；
5. 逐部影片直接调用 ``filter_output_graph``（后处理总链），而**不**调用 ``write_test_submission``：后者要求
   原始图数量等于影片数、超过截止时间会静默关闭修复阶段、并把任何异常吞掉改写成兜底图——这三点都会
   悄悄污染训练表。

导出（dump）模式：只记录，不改图
==============================
推理模块（d1_module.py）在全局变量 ``D1_DUMP`` 是一个列表时进入导出模式：在安全分裂之后的图上生成**全部**
候选、计算**全部**特征（不用惰性 DeepCenter 上界、不受 30 秒时间预算限制，所以每一行都完整），把它们追加进
列表，然后把安全分裂阶段的边**原样返回**。因此：

- 训练和推理用的是**同一个特征函数、同一个管线位置**——不会出现“训练时这样算、推理时那样算”的偏差；
- 导出模式下最终一遍的输出 = 关掉分裂补全打分器时的输出（唯一的区别就是打分器补的分叉没有加上；
  DeepCenter 热图多查了一些帧，只影响缓存和耗时）。``--check-dump-neutral`` 会在本次待处理的第一部影片上
  实际跑两遍核对这一点；
- 模块开关 ``_D1_ACTIVE`` 必须为 True 才会导出（门控在导出判断之前）。

标签怎么打（与比赛时制作训练表的规则完全相同，见 ``label_rows``）
==================================================================
1. **快照时机**：用一个透明包装器套在分裂阶段外面，记下分裂阶段刚结束时的节点和边（安全分裂已加、短轨迹
   过滤和跳变感知平滑还没做）。候选就是在这张图上生成的，标签必须用同一张图。
2. **节点匹配**：把快照里的节点坐标四舍五入成整数（与写 submission.csv 时一样，评测看到的就是取整坐标），
   与 GT 逐帧做官方指标的一对一最优匹配（7 µm 内，``tracksdata.metrics.DistanceMatching``），得到
   ``预测节点 → GT 节点``。
3. **只留指标看得见的行**：母细胞 P 必须匹配到一个**至少有一个 GT 子节点**的 GT 节点 g。GT 只标注了约 2.8% 的
   细胞，没匹配上的母细胞无从判断对错，进训练表只会是噪声；这些候选只计入 ``n_all``（全部候选数）。
4. 每行记录：``div`` = g 在 GT 中分裂（≥ 2 个子节点）；``label`` = B 匹配到 g 的某个 GT 子节点；``near_div`` =
   g 是分裂节点本身、某个分裂节点的子节点或父节点（官方分裂指标有 ±1 帧容忍，这些位置的标签含糊）；``a_ok`` = A 也
   匹配到 g 的子节点；``b_annotated`` = B 匹配到任一 GT 节点。step2 用 ``label and div`` 作正例，
   ``near_div and not 正例`` 作含糊行（不参与拟合；“g 确实分裂、但 B 不是它的子节点”的行也在其中），其余为负例。
5. 特征保留 4 位小数写盘（比赛时的训练表就是这样存的；要逐位复现发布的系数，必须保留这一步）。

产物（``--out-root``，默认 ``/kaggle/working/d1_table``）
========================================================
- ``<out-root>/<table-name>/<影片>.json``（默认表名 ``final``）：
  ``{"movie", "metrics"?, "rows": [...], "n_all", "gt_divisions", "stats", "seconds", "labelling"}``，
  每行 ``{"P", "A", "B", "t", "div", "near_div", "label", "a_ok", "b_annotated", "g", "Q", "q_annotated",
  "q_true_parent", "feat": {特征名: 值}}``——与比赛时的训练表格式相同，step2 两种表都能读。
  ``metrics`` 只有在环境里装了官方评测包 ``tracking_cellmot`` 并加 ``--official-metric`` 时才写（官方指标在
  “不加分叉”的最终图上的边 / 分裂计数，只用于 step2 读数表里的 divJ 推算列）。
- ``<out-root>/summary_shard{i}of{n}.json``：每部影片的行数、正例数、候选数、耗时与失败列表。

算力与耗时（粗估）
==================
GPU 推理与第 1 部分相同：两张 T4 并行，平均每部影片约 2.5 分钟（单卡每部检测 + 关联约 4–4.5 分钟，ILP 再加
5–75 秒）。之后逐部影片做
后处理（CPU 为主）；导出模式还要给**每个**候选查 DeepCenter 热图（推理时惰性上界只查约 4%），DeepCenter 在
T4 上每帧约 0.2–0.3 秒、一部影片 100 帧，所以每部再多约半分钟。199 部影片合计粗估 12–20 小时，超过单个会话 12 小时
的上限，建议分 4 片（每片约 50 部，约 3–5 小时）。正例很少（发布模型的训练表：199 部影片 6,760 行里只有
46 个正例，其中 42 个来自 6bba 胚胎），所以**不要**只跑一部分影片——少跑一片就可能少掉一大块正例。

关于发布模型的训练表
====================
推理模块里发布的 MODEL 是在**较早的管线**上导出的训练表拟合的：当时用公开坐标头（还没有 10 折坐标头）、
没有关联特征修正和找回概率打分、找回阈值 0.965（现在 0.94）、ILP 分裂权重也是公开基线的取值。用本脚本
（默认 = 最终管线）重建的表会有些不同，拟合出的系数也会略有差异——两者都是合理的选择：发布模型的优势是
经过了官方指标 A/B 的检验，重建表的优势是训练/推理的分布完全一致。``--set`` 可以覆盖后处理的全局变量
（例如 ``--set _D4_ACTIVE=False --set _RR_ACTIVE=False --set READMIT_MIN_SCORE=0.965``）来接近当时的后处理，
但原始图（坐标头、ILP 分裂权重）仍来自当前的检测管线，所以只能复现一半。``--set`` 在第 0–5 段执行完之后才生效，
只影响后处理在调用时读取的全局变量；第 4 段 GPU 推理用到的设置（例如 ILP 分裂权重）改不了。

另外要记住：这些训练影片本身参与过检测器和坐标头的训练，在它们上面得到的图比测试集上的更“干净”（假
轨迹起点更少），属于样本内数据。比赛时在碎片化的留出图上，打分器的收益明显变小甚至为负。
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import io
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping

# 让 ``from common import ...`` 在“从仓库根目录运行本脚本”时可用（也让验证脚本可以直接 import 本文件）。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import pipeline_runner as pr  # noqa: E402

# 训练表的默认输出根目录与表名。必须在 /kaggle/working/tracking_repo 之外（第 3 段会删除并重建那个目录）。
DEFAULT_OUT_ROOT = pr.WORKING_DIR / "d1_table"
DEFAULT_TABLE_NAME = "final"
# 只执行到第 5 段（后处理函数定义 + DeepCenter 加载 + 分裂补全打分器安装）。
LAST_SECTION = 5
# 第 11 段最终一遍打开的 5 个模块开关（顺序与推理脚本中相同）：候选概率重链接、全局位移估计 + 跳变感知平滑、
# 关联特征修正、找回概率打分、分裂补全打分器。
FINAL_PASS_FLAGS = ("_DR_ACTIVE", "_G1X1_ACTIVE", "_D4_ACTIVE", "_RR_ACTIVE", "_D1_ACTIVE")
# 训练表里特征保留的小数位数（比赛时的训练表就是这样存的）。
FEATURE_DECIMALS = 4
# 官方指标的节点匹配半径（µm）。
MATCH_RADIUS_UM = 7.0
# 官方指标的计数字段（只在 --official-metric 时写入 metrics）。
METRIC_KEYS = ("edge_tp", "edge_fp", "edge_fn", "division_tp", "division_fp", "division_fn", "num_pred_nodes")
# 推理管线在 /kaggle/working 下生成、只对推理有用的大目录（--cleanup 时删除）。
PIPELINE_SCRATCH_DIRS = ("tracking_repo", "edge_cache", "x138_candidate_prob_cache", "secondary_seed_weights")


# =============================================================================================
# 标签（纯函数，不依赖管线和 tracksdata，可以单独测试，也可以拿到别的环境里与旧训练表逐行核对）
# =============================================================================================
def gt_division_sets(gt_edges: Iterable[tuple[int, int]]):
    """由 GT 边 (母 ID, 子 ID) 得到 (子节点表, 父节点表, 分裂节点集合, 分裂邻域集合)。

    分裂邻域 = 分裂节点本身 + 它的子节点 + 它的父节点。官方分裂指标对时间有 ±1 帧的容忍
    （母细胞一侧认“分裂节点或它的前一帧”，子细胞一侧认“子节点或它的后一帧”），落在这些 GT 节点上的
    母细胞，其候选的标签是含糊的。
    """
    gt_children: dict[int, list[int]] = {}
    gt_parent: dict[int, int] = {}
    for u, v in gt_edges:
        gt_children.setdefault(int(u), []).append(int(v))
        gt_parent[int(v)] = int(u)
    dividers = {g for g, c in gt_children.items() if len(c) >= 2}
    near = set()
    for g in dividers:
        near.add(g)
        near.update(gt_children[g])
        if g in gt_parent:
            near.add(gt_parent[g])
    return gt_children, gt_parent, dividers, near


def label_rows(dump: Iterable[Mapping[str, Any]], node_gt: Mapping[int, int], gt_edges: Iterable[tuple[int, int]],
               decimals: int = FEATURE_DECIMALS) -> list[dict]:
    """给导出的候选打 GT 标签，返回训练表的行（只含“母细胞匹配到有 GT 子节点的 GT 节点”的候选）。

    - ``dump``：推理模块导出模式追加的候选列表，每个元素 ``{"P", "A", "B", "t", "feat"}``
      （P = 母细胞、A = 它现有的子节点、B = 下一帧的无父起点、t = P 的帧号、feat = 特征字典）；
    - ``node_gt``：``{预测节点 ID: GT 节点 ID}``，在分裂阶段刚结束时的图上、按官方规则逐帧匹配得到；
    - ``gt_edges``：GT 边 ``(母 ID, 子 ID)``。

    行的字段与顺序和比赛时的训练表相同。``Q`` / ``q_*`` 是另一种候选生成方式（第二子细胞已被别的母细胞
    占用）才有的字段，本模块的候选没有 Q，按原格式写 None / 0。
    """
    gt_children, gt_parent, dividers, near = gt_division_sets(gt_edges)
    rows = []
    for c in dump:
        g = node_gt.get(c["P"])
        # 母细胞没匹配上 GT，或匹配到的 GT 节点没有子节点（谱系片段的末端）：指标看不到，不进训练表。
        if g is None or not gt_children.get(g):
            continue
        b_gt = node_gt.get(c["B"])
        a_gt = node_gt.get(c["A"])
        kids = gt_children[g]
        q_gt = node_gt.get(c["Q"]) if "Q" in c else None
        rows.append({"P": c["P"], "A": c["A"], "B": c["B"], "t": c["t"], "div": int(g in dividers), "near_div": int(g in near),
                     "label": int(b_gt is not None and b_gt in kids), "a_ok": int(a_gt is not None and a_gt in kids),
                     "b_annotated": int(b_gt is not None), "g": g, "Q": c.get("Q"),
                     "q_annotated": int(q_gt is not None),
                     "q_true_parent": int(q_gt is not None and b_gt is not None and gt_parent.get(b_gt) == q_gt),
                     "feat": {k: round(float(v), decimals) for k, v in c["feat"].items()}})
    return rows


# =============================================================================================
# 选影片、打补丁
# =============================================================================================
def select_movies(train_dir: Path, movies_arg: str | None, shard: int, num_shards: int) -> list[str]:
    """本分片要处理的影片：全部训练影片（或 ``--movies`` 给出的列表，逗号分隔或 ``@文件``）按名字排序后
    取 ``movies[shard::num_shards]``——交错切分让每个分片都含两个胚胎、耗时也更均匀。"""
    if movies_arg:
        if movies_arg.startswith("@"):
            lines = Path(movies_arg[1:]).read_text(encoding="utf-8").splitlines()
            movies = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
        else:
            movies = [m.strip() for m in movies_arg.split(",") if m.strip()]
        available = set(pr.list_train_movies(train_dir))
        unknown = [m for m in movies if m not in available]
        if unknown:
            raise SystemExit(f"这些影片在 {train_dir} 中没有 .zarr + .geff：{unknown[:5]}")
    else:
        movies = pr.list_train_movies(train_dir)
    chosen = pr.shard_movies(movies, shard, num_shards)
    if not chosen:
        raise SystemExit(f"分片 {shard}/{num_shards} 没有影片（共 {len(movies)} 部）")
    return chosen


def table_patches(subset_dir: str | Path, coord_head_mode: str = "candidate") -> list[tuple[str, str, str]]:
    """制作训练表所需的补丁，每处都锚定推理脚本中恰好出现一次的整行原文（见 common/pipeline_runner.py）。"""
    patches = [
        # 1) 第 2 段：TEST_DIR = COMP_DIR / "test"  →  TEST_DIR = Path('<训练影片子集目录>')。
        #    第 4 段列影片、推理子进程读影像，第 5 段 read_test_frame 读原始帧（亮度特征、DeepCenter 热图）都经由它。
        pr.redirect_test_dir_patch(subset_dir),
        # 2) 第 5 段最后一句：不跑第一遍后处理、不写 submission.csv。
        pr.line_patch("跳过第一遍后处理与 submission.csv", pr.ANCHORS["first_pass_call"], "pass"),
    ]
    if coord_head_mode == "zero":
        # 还没有自己的 10 折坐标头时：跳过“必须恰好 10 个头”的检查，坐标修正头不修正坐标。能跑通，但原始图的
        # 节点坐标与最终管线不同，训练表的分布会略有偏差。
        patches += [
            pr.line_patch("坐标头：没有头文件时跳过 10 个头的挂载检查", pr.ANCHORS["coord_head_count_guard"], "if False:"),
            pr.line_patch("坐标头：zero 模式（不修正坐标）", pr.ANCHORS["coord_head_mode"], "os.environ['V1284_MODE']='zero'"),
        ]
    elif coord_head_mode != "candidate":
        raise ValueError(f"未知的坐标头模式：{coord_head_mode!r}（只能是 candidate 或 zero）")
    return patches


def build_table_source(subset_dir: str | Path, coord_head_mode: str = "candidate",
                       pipeline_path: str | Path | None = None) -> str:
    """读取第三部分推理脚本，打补丁，只保留第 0–5 段，返回可直接执行的源码。

    先在完整源码上打补丁（锚点唯一性在全文范围检查），再切片；切片后再核对几个关键行，确保
    “第 5 段完整、第一遍被跳过、第 6 段以后不在其中”。
    """
    source = pr.load_pipeline_source(pipeline_path)
    source = pr.apply_patches(source, table_patches(subset_dir, coord_head_mode))
    sliced = pr.section_slice(source, 0, LAST_SECTION)
    counts = pr.check_anchors(sliced, {
        "deepcenter_load": pr.ANCHORS["deepcenter_load"],
        "d1_switch_default": pr.ANCHORS["d1_switch_default"],
        "first_pass_call": pr.ANCHORS["first_pass_call"],
        "final_pass_pp_apply": pr.ANCHORS["final_pass_pp_apply"],
    })
    expected = {"deepcenter_load": 1, "d1_switch_default": 1, "first_pass_call": 0, "final_pass_pp_apply": 0}
    if counts != expected:
        raise RuntimeError(f"第 0–{LAST_SECTION} 段切片的关键行计数 {counts} 与预期 {expected} 不符，推理脚本的分段可能被改动了")
    return sliced


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """``--set 名字=值``：值按 Python 字面量解析（True / False / 0.965 / 'abc'）。"""
    result: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set 需要 名字=值 的形式：{item!r}")
        name, text = item.split("=", 1)
        try:
            result[name.strip()] = ast.literal_eval(text.strip())
        except (ValueError, SyntaxError):
            raise SystemExit(f"--set {name}：值 {text!r} 不是 Python 字面量") from None
    return result


# =============================================================================================
# 在管线命名空间里逐部影片运行最终一遍后处理（导出模式）
# =============================================================================================
def set_final_pass_state(ns: dict, overrides: Mapping[str, Any] | None = None) -> None:
    """按第 11 段最终一遍的方式打开 5 个模块开关，再应用 --set 覆盖（覆盖的名字必须已在命名空间里，防止拼错）。"""
    for flag in FINAL_PASS_FLAGS:
        if flag not in ns:
            raise RuntimeError(f"命名空间里没有模块开关 {flag}：推理脚本第 5 段是否完整执行？")
        ns[flag] = True
    for name, value in (overrides or {}).items():
        if name not in ns:
            raise SystemExit(f"--set {name}：推理脚本里没有这个全局变量")
        print(f"覆盖全局变量 {name} = {value!r}（原值 {ns[name]!r}）")
        ns[name] = value


def install_snapshot(ns: dict) -> dict:
    """在分裂阶段（已被分裂补全打分器包装过的 add_safe_divisions_postlink）外面再套一层透明包装：
    原样调用并返回结果，同时把“分裂阶段刚结束时”的节点与边复制一份存进返回的字典 ``snap``。

    为什么要复制：后面的短轨迹过滤、跳变感知平滑会删节点、原地改坐标；标签必须打在候选生成时的那张图上。
    ``filter_output_graph`` 在调用时按全局名字查找 ``add_safe_divisions_postlink``，所以替换命名空间里的
    这个名字就足够了。
    """
    inner = ns["add_safe_divisions_postlink"]
    if getattr(inner, "_d1_table_capture", False):
        raise RuntimeError("快照包装器已经安装过了")
    if getattr(inner, "__name__", "") != "stage":
        raise RuntimeError("分裂阶段不是分裂补全打分器的包装函数：推理脚本第 5 段的模块安装没有执行？")
    snap: dict = {}

    def capture(nodes_by_id, edges, stats, **kw):
        result = inner(nodes_by_id, edges, stats, **kw)
        snap["dataset"] = kw.get("dataset")
        snap["nodes"] = {int(k): dict(v) for k, v in nodes_by_id.items()}
        snap["edges"] = [dict(e) for e in result]
        snap["calls"] = snap.get("calls", 0) + 1
        return result

    capture._d1_table_capture = True
    ns["add_safe_divisions_postlink"] = capture
    return snap


def find_raw_geffs(ns: dict, movies: list[str]) -> dict[str, Path]:
    """第 4 段推理子进程写出的原始 ILP 图（与 write_test_submission 用同一个 glob）。"""
    geffs = sorted((Path(ns["REPO_DIR"]) / "predictions").glob(f"*/{ns['METHOD']}/split_0/*.geff"))
    by_movie = {p.stem: p for p in geffs}
    missing = [m for m in movies if m not in by_movie]
    if missing:
        raise RuntimeError(f"{len(missing)} 部影片没有原始图（GPU 推理失败？）：{missing[:5]}")
    return {m: by_movie[m] for m in movies}


def load_raw_graph(ns: dict, geff_path: Path) -> tuple[dict[int, dict], list[dict]]:
    """读原始 ILP 图，转成后处理用的节点字典与边列表——逐字段照搬 write_test_submission 的读法。"""
    graph = ns["graph_from_geff"](geff_path)
    nodes_by_id: dict[int, dict] = {}
    for row in graph.node_attrs().iter_rows(named=True):
        node_id = int(row["node_id"])
        nodes_by_id[node_id] = {
            "node_id": node_id,
            "t": int(row["t"]),
            "z": float(row["z"]),
            "y": float(row["y"]),
            "x": float(row["x"]),
        }
    raw_edges: list[dict] = []
    for row in graph.edge_attrs().iter_rows(named=True):
        edge_prob = row.get("edge_prob") if hasattr(row, "get") else None
        raw_edges.append({
            "source_id": int(row["source_id"]),
            "target_id": int(row["target_id"]),
            "edge_prob": None if edge_prob is None else float(edge_prob),
        })
    return nodes_by_id, raw_edges


def fresh_copy(nodes: Mapping[int, Mapping], edges: Iterable[Mapping]) -> tuple[dict[int, dict], list[dict]]:
    """后处理会原地修改节点字典和边，每次调用都给它一份新拷贝。"""
    return {int(k): dict(v) for k, v in nodes.items()}, [dict(e) for e in edges]


def run_postprocess(ns: dict, snap: dict, movie: str, raw_nodes, raw_edges, bundle, *, dump_mode: bool = True,
                    verbose: bool = False):
    """对一部影片跑一遍最终一遍后处理。dump_mode=True 时打分器进入导出模式，返回 (候选列表, 最终节点, 最终边, 统计)。

    后处理的逐阶段打印默认收进缓冲区（每部影片几十行）；出错时把末尾打印出来。
    """
    nodes, edges = fresh_copy(raw_nodes, raw_edges)
    dump: list = []
    snap.clear()
    buffer = io.StringIO()
    ns["D1_DUMP"] = dump if dump_mode else None
    try:
        redirect = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(buffer)
        with redirect:
            nodes_f, edges_f, stats = ns["filter_output_graph"](nodes, edges, dataset=movie, deepcenter_bundle=bundle)
    except Exception:
        print(buffer.getvalue()[-4000:])
        raise
    finally:
        # 恢复为“没有导出列表”（等价于推理时的打分模式），避免下一次调用误用上一部影片的列表。
        ns.pop("D1_DUMP", None)
    if snap.get("calls") != 1 or snap.get("dataset") != movie:
        raise RuntimeError(f"{movie}：分裂阶段被调用 {snap.get('calls')} 次（数据集 {snap.get('dataset')!r}），快照不可信")
    if stats.get("m_d1_errors", 0):
        # 打分器出错时会退回安全分裂的结果并只记一次错误——对推理是保险，对训练表却意味着整部影片的候选丢了。
        print(buffer.getvalue()[-4000:])
        raise RuntimeError(f"{movie}：分裂补全打分器报错 {stats['m_d1_errors']} 次（见上面的输出）")
    return dump, nodes_f, edges_f, stats


def rounded_graph(nodes: Mapping[int, Mapping], edges: Iterable[Mapping]):
    """把后处理结果建成 tracksdata 图（坐标四舍五入为非负整数，与 submission.csv 相同），供官方评测使用。"""
    import polars as pl
    import tracksdata as td

    graph = td.graph.IndexedRXGraph()
    for axis in ("z", "y", "x"):
        graph.add_node_attr_key(axis, pl.Int64, 0)
    ordered = sorted(nodes)
    graph_ids = graph.bulk_add_nodes([
        {"t": int(nodes[n]["t"]), **{a: max(0, int(round(float(nodes[n][a])))) for a in ("z", "y", "x")}}
        for n in ordered
    ])
    mapping = dict(zip(ordered, graph_ids))
    edge_rows = [{"source_id": mapping[int(e["source_id"])], "target_id": mapping[int(e["target_id"])]} for e in edges]
    if edge_rows:
        graph.bulk_add_edges(edge_rows)
    return graph


def official_metrics(nodes_f, edges_f, train_dir: Path, movie: str) -> tuple[dict | None, str | None]:
    """可选：用官方评测包算最终图（未加分叉）的边 / 分裂计数，写进训练表的 metrics，供 step2 读数表推算 divJ。

    官方评测包 ``tracking_cellmot`` 不在 Kaggle 默认环境里；没有它时返回 (None, 原因)，训练表照常生成
    （metrics 只影响 step2 读数表的 divJ 两列，不影响拟合）。
    """
    try:
        from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics
    except ImportError:
        return None, "未安装官方评测包 tracking_cellmot"
    from common import gt_io

    try:
        # 重新读一份 GT：评测会在图上写匹配属性，不要与打标签用的那份混用。
        gt = gt_io.load_train_gt(train_dir, movie, backend="tracksdata")
        graph = rounded_graph(nodes_f, edges_f)
        result = evaluate(graph, gt.graph, scale=tuple(float(v) for v in gt.voxel_um), max_distance=MATCH_RADIUS_UM)
        recall = node_recall(graph, gt.graph) if graph.num_nodes() > 0 and graph.num_edges() > 0 else 0.0
        row = per_sample_metrics(result, gt.estimated_number_of_nodes, recall)
        return {k: row[k] for k in METRIC_KEYS}, None
    except Exception as exc:  # 只影响读数表，不让它中断训练表的制作
        return None, f"官方评测失败：{type(exc).__name__}: {exc}"


def process_movie(ns: dict, snap: dict, movie: str, raw_nodes, raw_edges, gt, bundle, *, match_backend: str = "tracksdata",
                  verbose: bool = False, train_dir: Path | None = None, with_metrics: bool = False) -> dict:
    """一部影片：导出模式跑后处理 → 快照上与 GT 做官方匹配 → label_rows → 训练表记录（字典）。"""
    from common import gt_io

    t0 = time.time()
    dump, nodes_f, edges_f, stats = run_postprocess(ns, snap, movie, raw_nodes, raw_edges, bundle, verbose=verbose)
    node_gt = gt_io.match_nodes_official(snap["nodes"], gt, pred_edges=snap["edges"], max_distance_um=MATCH_RADIUS_UM,
                                         round_coords=True, backend=match_backend)
    gt_edges = gt.edge_list()
    rows = label_rows(dump, node_gt, gt_edges)
    record: dict[str, Any] = {"movie": movie}
    if with_metrics and train_dir is not None:
        metrics, why = official_metrics(nodes_f, edges_f, train_dir, movie)
        if metrics is not None:
            record["metrics"] = metrics
        else:
            print(f"  [{movie}] 不写 metrics：{why}")
    record.update({
        "rows": rows,
        "n_all": len(dump),
        "gt_divisions": len(gt_division_sets(gt_edges)[2]),
        "stats": {k: v for k, v in stats.items() if k.startswith("m_") or k == "safe_divisions_added"},
        "seconds": round(time.time() - t0, 1),
        "labelling": {"snapshot_nodes": len(snap["nodes"]), "snapshot_edges": len(snap["edges"]),
                      "matched_snapshot_nodes": len(node_gt), "gt_nodes": gt.num_nodes,
                      "gt_backend": gt.backend, "match_backend": match_backend,
                      "match_radius_um": MATCH_RADIUS_UM, "feature_decimals": FEATURE_DECIMALS},
    })
    return record


def check_dump_neutral(ns: dict, snap: dict, movie: str, raw_nodes, raw_edges, bundle) -> dict:
    """核对“导出模式不改最终输出”：同一部影片分别以 (导出模式) 和 (关闭分裂补全打分器) 各跑一遍，比较最终节点与边。"""
    _, nodes_a, edges_a, _ = run_postprocess(ns, snap, movie, raw_nodes, raw_edges, bundle, dump_mode=True)
    saved = ns["_D1_ACTIVE"]
    ns["_D1_ACTIVE"] = False
    try:
        nodes, edges = fresh_copy(raw_nodes, raw_edges)
        with contextlib.redirect_stdout(io.StringIO()):
            nodes_b, edges_b, _ = ns["filter_output_graph"](nodes, edges, dataset=movie, deepcenter_bundle=bundle)
    finally:
        ns["_D1_ACTIVE"] = saved
    differing_nodes = sum(1 for k in set(nodes_a) | set(nodes_b) if nodes_a.get(k) != nodes_b.get(k))
    key = lambda e: (int(e["source_id"]), int(e["target_id"]))  # noqa: E731
    differing_edges = len({key(e) for e in edges_a} ^ {key(e) for e in edges_b})
    return {"movie": movie, "identical": nodes_a == nodes_b and edges_a == edges_b,
            "differing_nodes": differing_nodes, "differing_edge_pairs": differing_edges}


def write_json_atomic(path: Path, payload: dict) -> None:
    """先写临时文件再改名：会话中途被杀时不会留下半个 JSON（续跑时会被当作“已完成”跳过）。"""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, default=float), encoding="utf-8")
    os.replace(tmp, path)


def cleanup_pipeline_outputs(keep: Path, subset_dir: Path) -> None:
    """删除推理管线在 /kaggle/working 下生成的大目录，只保留训练表（便于把输出保存为数据集）。"""
    protected = keep.resolve()
    for target in [pr.WORKING_DIR / name for name in PIPELINE_SCRATCH_DIRS] + [subset_dir]:
        target = Path(os.path.abspath(target))
        if not target.exists():
            continue
        if protected == target or target in protected.parents or protected in target.parents:
            print(f"跳过（与训练表目录重叠）：{target}")
            continue
        if target.is_symlink():
            target.unlink()
        else:
            shutil.rmtree(target)
        print(f"已删除：{target}")


# =============================================================================================
# 主流程
# =============================================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在训练影片上跑完整推理管线（第 0–5 段 + 最终一遍开关），导出分裂补全打分器的训练表。")
    parser.add_argument("--train-dir", type=Path, default=None, help="竞赛训练集目录；默认 Kaggle 竞赛数据的 train/")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="输出根目录，默认 /kaggle/working/d1_table")
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME,
                        help="表名 = 子目录名（step2 用它作 MODEL['tables'] 的记录），各分片请保持一致；默认 final")
    parser.add_argument("--shard", type=int, default=0, help="本次运行的分片编号（从 0 开始）")
    parser.add_argument("--num-shards", type=int, default=1, help="分片总数（199 部影片建议 4）")
    parser.add_argument("--movies", default=None, help="可选：显式的影片列表，逗号分隔，或 @文件（每行一个）；之后仍按分片切分")
    parser.add_argument("--subset-dir", type=Path, default=pr.DEFAULT_SUBSET_DIR,
                        help="训练影片子集（符号链接）目录，默认 /kaggle/working/train_subset")
    parser.add_argument("--pipeline", type=Path, default=None,
                        help="推理脚本路径（.py 或 .ipynb）；默认 part3_inference_pipeline/biohub_final_inference.py")
    parser.add_argument("--coord-head-mode", choices=("candidate", "zero"), default="candidate",
                        help="candidate = 最终管线（需挂载 10 折坐标头数据集）；zero = 没有头时不修正坐标")
    parser.add_argument("--gt-backend", choices=("tracksdata", "raw", "auto"), default="tracksdata",
                        help="读 GT 的方式；默认与官方评测相同的 tracksdata（推理脚本第 3 段已安装）")
    parser.add_argument("--match-backend", choices=("tracksdata", "numpy", "auto"), default="tracksdata",
                        help="节点匹配；默认官方的 DistanceMatching，numpy 是等价思路的替代实现")
    parser.add_argument("--official-metric", action="store_true",
                        help="若环境里有官方评测包 tracking_cellmot，就把最终图的边/分裂计数写进 metrics（只用于读数表）")
    parser.add_argument("--check-dump-neutral", action="store_true",
                        help="在本次待处理的第一部影片上多跑一遍，核对导出模式的最终输出 = 关闭打分器时的输出")
    parser.add_argument("--set", action="append", default=None, metavar="名字=值",
                        help="在打开最终一遍开关之后覆盖推理脚本的全局变量（可重复；只影响后处理），例如 READMIT_MIN_SCORE=0.965")
    parser.add_argument("--no-skip-existing", action="store_true", help="默认跳过已有 JSON 的影片（续跑）；加上则全部重做")
    parser.add_argument("--verbose", action="store_true", help="显示后处理逐阶段的打印")
    parser.add_argument("--dry-run", action="store_true", help="只建子集、打补丁、编译检查，不运行推理")
    parser.add_argument("--cleanup", action="store_true", help="完成后删除推理管线的中间目录，只留训练表")
    args = parser.parse_args(argv)

    train_dir = (args.train_dir or (pr.competition_dir() / "train")).resolve()
    out_root = Path(os.path.abspath(args.out_root))
    if "tracking_repo" in out_root.parts:
        raise SystemExit("输出目录不能放在 tracking_repo 里面（推理管线第 3 段会删除它）")
    table_dir = out_root / args.table_name
    movies = select_movies(train_dir, args.movies, args.shard, args.num_shards)
    todo = movies if args.no_skip_existing else [m for m in movies if not (table_dir / f"{m}.json").exists()]
    print(f"分片 {args.shard}/{args.num_shards}：{len(movies)} 部影片，待处理 {len(todo)} 部，例如 {todo[:3]}", flush=True)
    if not todo:
        print("本分片的训练表已经全部存在，无事可做。")
        return 0
    overrides = parse_overrides(args.set)

    subset_dir = pr.make_train_subset(todo, train_dir, args.subset_dir)
    source = build_table_source(subset_dir, args.coord_head_mode, args.pipeline)
    if args.dry_run:
        compile(source, "biohub_d1_table.py", "exec", dont_inherit=True)
        for line in source.splitlines():
            if line.startswith(("TEST_DIR = ", "os.environ['V1284_MODE']", "if False:")) or line == "pass":
                print("补丁后：", line)
        print(f"dry-run 通过：第 0–{LAST_SECTION} 段共 {len(source.splitlines())} 行，可以编译；子集目录 {subset_dir}")
        if overrides:
            print(f"将覆盖的全局变量：{overrides}")
        return 0

    table_dir.mkdir(parents=True, exist_ok=True)
    # 后处理阶段还要从子集目录读原始影像（DeepCenter 与亮度特征），所以符号链接在 finally 里才删除。
    try:
        started = time.time()
        # 像 notebook 一样执行第 0–5 段：GPU 推理（第 4 段末尾的子进程）→ 后处理函数定义 → DeepCenter → 打分器安装。
        ns = pr.run_source(source, "biohub_d1_table.py")
        predict_seconds = time.time() - started
        from common import gt_io  # 第 3 段已安装 tracksdata，这时再导入

        bundle = ns.get("DEEPCENTER_VETO_DETECTOR")
        if bundle is None:
            # 没有 DeepCenter 时 3 个中心先验特征全为 0，缺口与安全分裂的否决也会关闭——这样的表与推理不一致。
            raise SystemExit("DeepCenter 没有加载（DEEPCENTER_VETO_DETECTOR 为 None）：检查 DeepCenter 数据集是否挂载")
        set_final_pass_state(ns, overrides)
        snap = install_snapshot(ns)
        geffs = find_raw_geffs(ns, todo)

        per_movie, failures, neutral = {}, [], None
        for index, movie in enumerate(todo):
            try:
                raw_nodes, raw_edges = load_raw_graph(ns, geffs[movie])
                if args.check_dump_neutral and index == 0:
                    neutral = check_dump_neutral(ns, snap, movie, raw_nodes, raw_edges, bundle)
                    print(f"导出模式中性检查：{neutral}", flush=True)
                gt = gt_io.load_train_gt(train_dir, movie, backend=args.gt_backend)
                record = process_movie(ns, snap, movie, raw_nodes, raw_edges, gt, bundle, match_backend=args.match_backend,
                                       verbose=args.verbose, train_dir=train_dir, with_metrics=args.official_metric)
                write_json_atomic(table_dir / f"{movie}.json", record)
                positives = sum(1 for r in record["rows"] if r["label"] and r["div"])
                ambiguous = sum(1 for r in record["rows"] if r["near_div"] and not (r["label"] and r["div"]))
                per_movie[movie] = {"n_all": record["n_all"], "rows": len(record["rows"]), "positives": positives,
                                    "ambiguous": ambiguous, "gt_divisions": record["gt_divisions"], "seconds": record["seconds"]}
                print(f"OK {movie}：候选 {record['n_all']}，进表 {len(record['rows'])} 行，正例 {positives}，含糊 {ambiguous}，"
                      f"GT 分裂 {record['gt_divisions']}，{record['seconds']:.0f}s", flush=True)
            except Exception:
                failures.append(movie)
                print(f"FAIL {movie}\n{traceback.format_exc()}", flush=True)

        totals = {k: sum(v[k] for v in per_movie.values()) for k in ("n_all", "rows", "positives", "ambiguous", "gt_divisions")}
        summary = {"shard": args.shard, "num_shards": args.num_shards, "table_dir": str(table_dir),
                   "coord_head_mode": args.coord_head_mode, "overrides": {k: repr(v) for k, v in overrides.items()},
                   "predict_seconds": round(predict_seconds, 1), "total_seconds": round(time.time() - started, 1),
                   "totals": totals, "movies": per_movie, "failures": failures, "dump_neutral_check": neutral}
        summary_path = out_root / f"summary_shard{args.shard}of{args.num_shards}.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"完成：{len(per_movie)} 部影片，{totals['rows']} 行，正例 {totals['positives']}，含糊 {totals['ambiguous']}；"
              f"汇总 {summary_path}")
        # 中性检查不通过也要让退出码非 0：后台 Save & Run All 时，只有退出码能让人注意到问题。
        neutral_bad = neutral is not None and not neutral["identical"]
        if neutral_bad:
            print("警告：导出模式的最终输出与关闭打分器时不同，请检查（见上面的中性检查结果）")
        if failures:
            print(f"{len(failures)} 部影片失败：{failures[:10]}（修好后重跑同一分片，已完成的影片会被跳过）")
            return 1
        if neutral_bad:
            return 2
        if args.cleanup:
            cleanup_pipeline_outputs(table_dir, subset_dir)
        return 0
    finally:
        pr.remove_train_subset(subset_dir)


if __name__ == "__main__":
    raise SystemExit(main())
