"""读取竞赛训练集真值（GT，``train/<影片>.geff``）与逐帧匹配工具（两个训练数据导出脚本共用）。

GT 的格式（竞赛数据说明）
=========================
每部训练影片配一个同名的 ``<影片>.geff`` 目录。GEFF 是基于 Zarr v3 的图交换格式（本赛题由
``tracksdata`` 导出，geff v1.1），目录结构：

=================================  ===========================================================
路径                                含义
=================================  ===========================================================
``nodes/ids``                      节点 ID，形状 (N,)
``nodes/props/{t,z,y,x}/values``   每个节点的帧号与质心坐标（**体素单位**的整数）
``edges/ids``                      边，形状 (E, 2)，两列为 (源 = 母/前一帧, 目标 = 子/后一帧)
根 ``zarr.json`` 的 geff 元数据    ``extra.estimated_number_of_nodes``：主办方估计的该影片真实细胞总数
=================================  ===========================================================

所有数组用 zstd 压缩。读 GT 时要记住四件事：

1. **坐标是体素下标，不是微米。** ``nodes/props/{t,z,y,x}`` 与影像数组 ``image[t, z, y, x]`` 一一对应
   （无翻转、无平移）。体素尺寸 z = 1.625 µm、y = x = 0.40625 µm（``VOXEL_UM``），z 与 xy 之比 4:1。
   任何距离都必须先乘体素尺寸换成 µm 再算：z 方向差 4 个体素是 6.5 µm，y 方向差 4 个体素只有 1.6 µm，
   直接在体素坐标上算距离会把 z 方向低估 4 倍。
2. **GT 是稀疏标注。** 只标注了约 2.8% 的细胞（每帧平均约 7 个），标注的是谱系片段。没被标注的细胞
   不等于“这里没有细胞”——所以官方指标只把“与 GT 冲突”的预测边算作 FP，导出训练数据时也只能
   在有 GT 的位置打标签。
3. **分裂 = 一个节点有两条出边。** 199 部训练影片里总共只有约 151 个 GT 分裂，正样本极少。
4. ``estimated_number_of_nodes`` 进入指标的节点数调整项（预测节点数相对它多报会扣分）；
   本模块把它读出来备用（目前两个导出脚本都不需要）。

体素尺寸由谁决定：官方评测代码从影像 ``<影片>.zarr`` 根属性的 OME ``multiscales`` 变换中读取
（``read_zarr_voxel_scale`` 复刻了同一逻辑），199 部训练影片全部是 (1.625, 0.40625, 0.40625)；
推理脚本第 5 段写死的 ``VOXEL_SCALE_UM`` 也是这个值。

读取后端的选择（``backend`` 参数）
==================================
- ``"tracksdata"``：``tracksdata.graph.IndexedRXGraph.from_geff``——官方评测代码和推理脚本
  （第 5 段 ``graph_from_geff``）用的就是它，原始训练脚本读 GT 也用它。用它读，节点的**行顺序**
  与当初制作训练数据时完全相同，这是逐位复现训练对（第 1 部分 step2）的前提。推理 notebook 的
  第 3 段会从离线 wheel 安装 tracksdata，所以在跑过管线的 Kaggle 会话里总是可用。
- ``"raw"``：不依赖任何图库，直接按 Zarr v3 规范解析（读 ``zarr.json`` 元数据，逐块 zstd 解压，
  按小端字节序还原数组）。为什么不用 ``zarr`` 包：Kaggle 默认镜像里的 zarr 可能是 2.x，读不了 v3；
  而 GEFF 的结构很简单（几个一维/二维数组），直接解析反而最稳。推理脚本第 5 段 ``read_test_frame``
  读影像时也是绕过 zarr、直接解压数据块。行顺序 = 文件中的存储顺序。
- ``"auto"``（默认）：能 import tracksdata 就用它，否则用 raw。实际使用的后端记录在
  ``GroundTruth.backend`` 里。两种后端读出的数值相同；如果 tracksdata 内部重排了节点，行顺序可能
  不同——这只影响“距离完全相等”时谁先被匹配（极少见）。要逐位复现原始训练对，请用 tracksdata。

逐帧匹配工具
============
- ``greedy_match_frame``：贪心一对一最近邻（第 1 部分制作坐标头训练对，半径 3.5 µm）。
- ``optimal_match_frame`` / ``match_nodes_official``：与官方指标相同的逐帧最优二分匹配（7 µm），
  第 2 部分用它给分裂候选打标签。有 tracksdata 时直接调用官方使用的
  ``tracksdata.metrics.DistanceMatching``，没有时用等价思路的 numpy/scipy 实现。

本模块只依赖 numpy；scipy、tracksdata、zstd 解压库都在用到时才 import。
"""
from __future__ import annotations

import gzip
import itertools
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "MATCH_RADIUS_UM",
    "VOXEL_UM",
    "GroundTruth",
    "greedy_match_frame",
    "load_train_gt",
    "match_nodes_official",
    "optimal_match_frame",
    "read_estimated_number_of_nodes",
    "read_geff_metadata",
    "read_gt_geff",
    "read_zarr_v3_array",
    "read_zarr_voxel_scale",
]

# (z, y, x) 体素尺寸，单位 µm：官方指标计算匹配距离时用的就是这组值。
VOXEL_UM: tuple[float, float, float] = (1.625, 0.40625, 0.40625)
# 官方指标的节点匹配半径（µm）：同一帧内，预测点与 GT 点距离超过 7 µm 就不能配对。
MATCH_RADIUS_UM = 7.0


# =============================================================================================
# 数据结构
# =============================================================================================
@dataclass
class GroundTruth:
    """一部影片的 GT 图。所有数组按同一行顺序对齐（第 i 行 = 第 i 个节点）。

    字段：
    - ``node_ids``：(N,) int64，GT 节点 ID（边用它来引用节点）；
    - ``t``：(N,) int64，帧号；
    - ``zyx``：(N, 3) float64，**体素单位**的 (z, y, x)（GT 本身是整数，转成浮点方便计算）；
    - ``edges``：(E, 2) int64，每行 (母节点 ID, 子节点 ID)，t(子) = t(母) + 1；
    - ``voxel_um``：(z, y, x) 体素尺寸（µm）；``load_train_gt`` 会用影像元数据中的值覆盖默认值；
    - ``estimated_number_of_nodes``：主办方估计的真实细胞总数（可能为 None）；
    - ``backend``：实际使用的读取后端（"tracksdata" 或 "raw"）；
    - ``graph``：tracksdata 后端时保留原始图对象（官方匹配要用），否则为 None。
    """

    movie: str
    node_ids: np.ndarray
    t: np.ndarray
    zyx: np.ndarray
    edges: np.ndarray
    voxel_um: tuple[float, float, float] = VOXEL_UM
    estimated_number_of_nodes: float | None = None
    backend: str = ""
    path: Path | None = None
    metadata: dict = field(default_factory=dict, repr=False)
    graph: Any = field(default=None, repr=False)

    @property
    def num_nodes(self) -> int:
        return int(len(self.node_ids))

    @property
    def num_edges(self) -> int:
        return int(len(self.edges))

    def zyx_um(self) -> np.ndarray:
        """(N, 3) 物理坐标（µm）= 体素坐标 × 体素尺寸。所有距离都应在这个空间里算。"""
        return self.zyx * np.asarray(self.voxel_um, dtype=np.float64)

    def frames(self) -> np.ndarray:
        """有 GT 节点的帧号（升序）。稀疏标注下很多帧可能一个 GT 都没有。"""
        return np.unique(self.t)

    def rows_in_frame(self, frame: int) -> np.ndarray:
        """第 ``frame`` 帧的 GT 节点所在行（升序）。"""
        return np.flatnonzero(self.t == int(frame))

    def rows_by_frame(self) -> dict[int, np.ndarray]:
        """``{帧号: 该帧 GT 节点所在行}``，帧号升序。"""
        return {int(frame): np.flatnonzero(self.t == frame) for frame in np.unique(self.t)}

    def index_of(self) -> dict[int, int]:
        """``{节点 ID: 行号}``。"""
        return {int(node_id): row for row, node_id in enumerate(self.node_ids.tolist())}

    def edge_list(self) -> list[tuple[int, int]]:
        """边列表 ``[(母 ID, 子 ID), ...]``（文件中的顺序）。"""
        return [(int(src), int(dst)) for src, dst in self.edges.tolist()]

    def children_of(self) -> dict[int, list[int]]:
        """``{母节点 ID: [子节点 ID, ...]}``。有两个子节点的母节点就是一次分裂。"""
        children: dict[int, list[int]] = {}
        for src, dst in self.edges.tolist():
            children.setdefault(int(src), []).append(int(dst))
        return children

    def parent_of(self) -> dict[int, int]:
        """``{子节点 ID: 母节点 ID}``。细胞只有一个母亲，GT 中每个节点至多一条入边。"""
        parents: dict[int, int] = {}
        for src, dst in self.edges.tolist():
            parents.setdefault(int(dst), int(src))
        return parents

    def dividing_nodes(self, min_children: int = 2) -> list[int]:
        """出边数 ≥ ``min_children`` 的节点 ID（升序），即 GT 中的分裂母细胞。"""
        return sorted(node for node, kids in self.children_of().items() if len(kids) >= min_children)

    def as_node_dict(self) -> dict[int, tuple[int, float, float, float]]:
        """``{节点 ID: (t, z, y, x)}``，与推理脚本第 8 段 ``graph_to_plain`` 的节点格式相同。"""
        return {
            int(node_id): (int(frame), float(z), float(y), float(x))
            for node_id, frame, (z, y, x) in zip(self.node_ids.tolist(), self.t.tolist(), self.zyx.tolist())
        }


# =============================================================================================
# 对外的读取函数
# =============================================================================================
def read_gt_geff(geff_path: str | Path, *, backend: str = "auto") -> GroundTruth:
    """读取一个 GT ``.geff`` 目录，返回 ``GroundTruth``（体素尺寸取默认值 ``VOXEL_UM``）。

    ``backend``：``"auto"``（默认，能用 tracksdata 就用）、``"tracksdata"`` 或 ``"raw"``，见模块说明。
    读完会做两项一致性检查：节点 ID 不重复、每条边的两端都是已知节点——GT 文件损坏或读错列时
    立刻报错，而不是悄悄产生错误的训练标签。
    """
    path = Path(geff_path)
    if not path.is_dir():
        raise FileNotFoundError(f"GT 文件不存在：{path}")
    if backend not in ("auto", "tracksdata", "raw"):
        raise ValueError(f"未知的读取后端：{backend!r}")

    loaded = None
    if backend in ("auto", "tracksdata"):
        try:
            import tracksdata  # noqa: F401  （只检查是否可用）
        except ImportError:
            if backend == "tracksdata":
                raise
        else:
            loaded = _read_with_tracksdata(path) + ("tracksdata",)
    if loaded is None:
        loaded = _read_raw(path) + (None, "raw")
    node_ids, frames, zyx, edges, graph, used = loaded

    if len(np.unique(node_ids)) != len(node_ids):
        raise ValueError(f"{path}：节点 ID 有重复")
    if len(edges) and not np.isin(edges, node_ids).all():
        raise ValueError(f"{path}：有边引用了不存在的节点")

    metadata = read_geff_metadata(path)
    movie = path.name[: -len(".geff")] if path.name.endswith(".geff") else path.name
    return GroundTruth(
        movie=movie,
        node_ids=node_ids,
        t=frames,
        zyx=zyx,
        edges=edges,
        voxel_um=VOXEL_UM,
        estimated_number_of_nodes=read_estimated_number_of_nodes(path),
        backend=used,
        path=path,
        metadata=metadata,
        graph=graph,
    )


def load_train_gt(
    train_dir: str | Path,
    movie: str,
    *,
    backend: str = "auto",
    check_scale: bool = True,
) -> GroundTruth:
    """读取 ``<train_dir>/<movie>.geff``，并从 ``<train_dir>/<movie>.zarr`` 的元数据读体素尺寸。

    ``check_scale=True`` 时要求体素尺寸等于 ``VOXEL_UM``：推理管线、坐标头的训练目标（µm）和官方
    指标都默认这组尺寸，万一某部影片不同，宁可报错也不要产生单位错误的训练数据。
    """
    root = Path(train_dir)
    gt = read_gt_geff(root / f"{movie}.geff", backend=backend)
    zarr_path = root / f"{movie}.zarr"
    if zarr_path.exists():
        gt.voxel_um = read_zarr_voxel_scale(zarr_path)
    if check_scale and not np.allclose(np.asarray(gt.voxel_um, dtype=np.float64), np.asarray(VOXEL_UM)):
        raise ValueError(f"{movie}：体素尺寸 {gt.voxel_um} 与约定的 {VOXEL_UM} 不同")
    return gt


def read_zarr_voxel_scale(zarr_path: str | Path) -> tuple[float, float, float]:
    """从影像 ``.zarr`` 根属性读 (z, y, x) 体素尺寸，逻辑与官方 ``tracking_cellmot.io._parse_scale`` 相同：

    有 ``multiscales`` 时取 ``multiscales[0].datasets[0].coordinateTransformations[0].scale`` 的最后 3 个值
    （前面是 t 轴），否则返回默认值 ``VOXEL_UM``。
    """
    attrs = _read_group_attributes(Path(zarr_path))
    if "multiscales" in attrs:
        transform = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        if transform.get("type") != "scale":
            raise ValueError(f"坐标变换类型不是 scale：{transform}")
        z, y, x = (float(value) for value in transform["scale"][-3:])
        return (z, y, x)
    return VOXEL_UM


def read_geff_metadata(geff_path: str | Path) -> dict:
    """返回 GEFF 根属性中的 ``geff`` 元数据字典（找不到时返回空字典）。"""
    attrs = _read_group_attributes(Path(geff_path))
    meta = attrs.get("geff")
    return dict(meta) if isinstance(meta, dict) else {}


def read_estimated_number_of_nodes(geff_path: str | Path, *, required: bool = False) -> float | None:
    """读 ``estimated_number_of_nodes``（主办方估计的该影片真实细胞总数）。

    官方评测代码读的是 ``GeffMetadata.read(path).extra["estimated_number_of_nodes"]``，即根
    ``zarr.json`` 中 ``attributes.geff.extra`` 下的同名键；这里先按这个位置读，找不到时再像推理脚本
    第 8 段 ``read_estimated_true_node_count`` 那样在全部根属性中递归查找。值缺失或不是正数时返回 None；
    ``required=True`` 时改为抛出 ``ValueError``（与官方行为一致）。
    """
    path = Path(geff_path)
    attrs = _read_group_attributes(path)
    meta = attrs.get("geff") if isinstance(attrs.get("geff"), dict) else {}
    extra = meta.get("extra") if isinstance(meta.get("extra"), dict) else {}
    value = extra.get("estimated_number_of_nodes")
    if value is None:
        value = _find_key_recursive(attrs, "estimated_number_of_nodes")
    try:
        count = float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        count = float("nan")
    if not math.isfinite(count) or count <= 0:
        if required:
            raise ValueError(f"{path} 中没有有效的 estimated_number_of_nodes")
        return None
    return count


# =============================================================================================
# 逐帧匹配
# =============================================================================================
def greedy_match_frame(det_um: np.ndarray, gt_um: np.ndarray, radius_um: float) -> list[tuple[int, int]]:
    """同一帧内的贪心一对一最近邻匹配，返回 ``[(检测下标 i, GT 下标 j), ...]``（按匹配先后顺序）。

    第 1 部分（10 折坐标头）用它制作训练对，半径 3.5 µm，训练目标 = GT 坐标 − 检测坐标（µm）。
    为什么这样做：
    - 一对一：一个 GT 细胞只能“认领”一个检测点，否则同一个目标会被重复计入、把学习目标拉偏；
    - 贪心按距离从小到大配对：最近的那对最可能是同一个细胞；
    - 3.5 µm：检测与 GT 的平均偏差约 1.6 µm（主要来自 z 方向 1.625 µm 的层间量化），坐标头的输出
      也被限制在 2 µm 以内；在 3.5 µm 内约 92% 的 GT 节点能配到检测点，这个半径又明显小于相邻细胞核
      之间的典型距离，错配很少；
    - 平局按 (距离, i, j) 的固定顺序打破：结果可逐位复现（这正是 10 折坐标头能被逐位重训的前提）。

    输入都是 µm 坐标（(n, 3) 的 z, y, x）。实现与原始训练脚本逐行一致，不要改动。
    """
    from scipy.spatial import cKDTree

    if not len(det_um) or not len(gt_um):
        return []
    pairs = cKDTree(det_um).sparse_distance_matrix(cKDTree(gt_um), radius_um, output_type="ndarray")
    used_det, used_gt, out = set(), set(), []
    for k in np.lexsort((pairs["j"], pairs["i"], pairs["v"])):
        i, j = int(pairs["i"][k]), int(pairs["j"][k])
        if i in used_det or j in used_gt:
            continue
        used_det.add(i)
        used_gt.add(j)
        out.append((i, j))
    return out


def optimal_match_frame(
    reference_um: np.ndarray,
    predicted_um: np.ndarray,
    max_distance_um: float = MATCH_RADIUS_UM,
) -> list[tuple[int, int]]:
    """同一帧内的最优二分匹配（官方指标的节点匹配思路），返回 ``[(参考下标, 预测下标), ...]``，按下标排序。

    官方指标逐帧把预测点与 GT 点做**一一对应的最优分配**：只允许距离 ≤ 7 µm 的点配对，在此前提下
    让整体匹配最好（每对的权重为 1/(1+距离)，求权重和最大）。与贪心不同，最优分配不会因为先配了
    一对近的点、而让另一个点失去唯一的匹配对象。

    这里的实现：先用 KD 树找出 7 µm 内的全部候选对，再按“候选对连成的连通块”分别做匈牙利算法
    （``scipy.optimize.linear_sum_assignment``）——不同块之间互不影响，分块后矩阵很小、速度快。
    权重用 float32 保存，与 tracksdata 的实现思路一致。仅在没有 tracksdata 时作为替代；
    距离恰好相等的平局可能与官方实现的打破方式不同（极少见）。
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    reference = _as_points(reference_um)
    predicted = _as_points(predicted_um)
    gate = float(max_distance_um)
    if len(reference) == 0 or len(predicted) == 0:
        return []

    neighborhoods = cKDTree(predicted).query_ball_point(reference, r=np.nextafter(gate, np.inf), return_sorted=True)

    # 并查集：参考点占 0..R-1，预测点占 R..R+P-1；有候选边相连的点属于同一个连通块。
    parent = list(range(len(reference) + len(predicted)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    offset = len(reference)
    candidates: list[tuple[int, int, float]] = []
    for ref_index, neighbor_indices in enumerate(neighborhoods):
        if not neighbor_indices:
            continue
        neighbors = np.asarray(neighbor_indices, dtype=np.int64)
        distances = np.linalg.norm(predicted[neighbors] - reference[ref_index], axis=1)
        for pred_index, distance in zip(neighbors.tolist(), distances.tolist()):
            if distance > gate:
                continue
            candidates.append((ref_index, pred_index, distance))
            root_a, root_b = find(ref_index), find(offset + pred_index)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)
    if not candidates:
        return []

    components: dict[int, list[tuple[int, int, float]]] = {}
    for candidate in candidates:
        components.setdefault(find(candidate[0]), []).append(candidate)

    matches: list[tuple[int, int]] = []
    for block in components.values():
        refs = sorted({ref for ref, _, _ in block})
        preds = sorted({pred for _, pred, _ in block})
        row_of = {ref: row for row, ref in enumerate(refs)}
        col_of = {pred: col for col, pred in enumerate(preds)}
        weights = np.full((len(refs), len(preds)), -1.0, dtype=np.float32)
        for ref, pred, distance in block:
            weights[row_of[ref], col_of[pred]] = np.float32(1.0 / (1.0 + distance))
        # 不允许的配对设为 -inf，迫使分配只在 7 µm 内进行；若因此无可行完全分配则退回带 -1 的矩阵。
        try:
            rows, cols = linear_sum_assignment(np.where(weights > 0, weights, -np.inf), maximize=True)
        except ValueError:
            rows, cols = linear_sum_assignment(weights, maximize=True)
        for row, col in zip(rows.tolist(), cols.tolist()):
            if weights[row, col] > 0:
                matches.append((refs[row], preds[col]))
    matches.sort()
    return matches


def match_nodes_official(
    pred_nodes: Mapping[int, Any],
    gt: GroundTruth,
    *,
    pred_edges: Iterable[Mapping[str, Any] | Sequence[int]] | None = None,
    max_distance_um: float = MATCH_RADIUS_UM,
    round_coords: bool = True,
    backend: str = "auto",
) -> dict[int, int]:
    """把预测节点与 GT 节点按官方规则逐帧匹配，返回 ``{预测节点 ID: GT 节点 ID}``（未匹配的不出现）。

    - ``pred_nodes``：推理脚本后处理里的节点字典 ``{id: {"t", "z", "y", "x", ...}}``，或 ``{id: (t, z, y, x)}``；
      坐标为原分辨率体素单位。
    - ``round_coords=True``：坐标先四舍五入为整数并截到 ≥ 0，与写 submission.csv 时一样——评测
      看到的就是取整后的坐标，打标签时也应该用它。
    - ``pred_edges``：可选，只在 tracksdata 后端时一并加进预测图（与官方调用方式保持一致），不影响节点匹配。
    - ``backend``：``"auto"``（有 tracksdata 就用官方的 ``DistanceMatching``）、``"tracksdata"`` 或
      ``"numpy"``（用 ``optimal_match_frame`` 逐帧计算）。
    """
    if backend not in ("auto", "tracksdata", "numpy"):
        raise ValueError(f"未知的匹配后端：{backend!r}")
    ordered = sorted(int(node_id) for node_id in pred_nodes)
    if not ordered or gt.num_nodes == 0:
        return {}
    table = np.asarray([_node_tzyx(pred_nodes[node_id]) for node_id in ordered], dtype=np.float64)
    if round_coords:
        # Python 的 round 是“四舍六入五成双”，与写提交文件时的取整方式相同。
        table[:, 1:] = np.asarray(
            [[max(0, int(round(value))) for value in row] for row in table[:, 1:].tolist()], dtype=np.float64
        )

    use_tracksdata = backend == "tracksdata"
    if backend == "auto":
        try:
            import tracksdata  # noqa: F401
            use_tracksdata = True
        except ImportError:
            use_tracksdata = False
    if use_tracksdata:
        return _match_with_tracksdata(ordered, table, pred_edges, gt, max_distance_um)

    scale = np.asarray(gt.voxel_um, dtype=np.float64)
    pred_frames = table[:, 0].astype(np.int64)
    pred_um = table[:, 1:] * scale
    gt_um = gt.zyx_um()
    result: dict[int, int] = {}
    for frame in np.unique(pred_frames).tolist():
        pred_rows = np.flatnonzero(pred_frames == frame)
        gt_rows = gt.rows_in_frame(frame)
        if not len(gt_rows):
            continue
        for gt_index, pred_index in optimal_match_frame(gt_um[gt_rows], pred_um[pred_rows], max_distance_um):
            result[ordered[int(pred_rows[pred_index])]] = int(gt.node_ids[gt_rows[gt_index]])
    return result


# =============================================================================================
# 内部实现：tracksdata 后端
# =============================================================================================
def _read_with_tracksdata(path: Path):
    """用 tracksdata 读 GEFF（与官方评测、推理脚本第 5 段 ``graph_from_geff`` 相同的调用）。"""
    import tracksdata as td

    loaded = td.graph.IndexedRXGraph.from_geff(path)
    graph = loaded[0] if isinstance(loaded, tuple) else loaded
    keys = td.DEFAULT_ATTR_KEYS
    if graph.num_nodes() == 0:
        # 空图往返 GEFF 时坐标列可能缺失；直接返回空数组，不去访问不存在的列。
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int64), graph
    nodes = graph.node_attrs()
    node_ids = np.asarray(nodes[keys.NODE_ID].to_numpy()).astype(np.int64)
    frames = np.asarray(nodes["t"].to_numpy()).astype(np.int64)
    zyx = np.stack([np.asarray(nodes[axis].to_numpy()) for axis in ("z", "y", "x")], axis=1).astype(np.float64)
    if graph.num_edges() > 0:
        edge_table = graph.edge_attrs()
        edges = np.stack(
            [
                np.asarray(edge_table[keys.EDGE_SOURCE].to_numpy()),
                np.asarray(edge_table[keys.EDGE_TARGET].to_numpy()),
            ],
            axis=1,
        ).astype(np.int64)
    else:
        edges = np.empty((0, 2), dtype=np.int64)
    return node_ids, frames, zyx, edges, graph


def _match_with_tracksdata(ordered, table, pred_edges, gt: GroundTruth, max_distance_um: float) -> dict[int, int]:
    """官方匹配：把预测节点建成 tracksdata 图，调用 ``graph.match(gt_graph, DistanceMatching(...))``。

    官方 ``_evaluate`` 内部做的就是这一步；匹配结果写在预测图节点的 ``MATCHED_NODE_ID`` 属性上
    （未匹配为 -1）。
    """
    import polars as pl
    import tracksdata as td
    from tracksdata.metrics import DistanceMatching

    keys = td.DEFAULT_ATTR_KEYS
    gt_graph = gt.graph
    if gt_graph is None:
        if gt.path is None:
            raise ValueError("GT 不是用 tracksdata 读的，也没有文件路径，无法做官方匹配")
        gt_graph = _read_with_tracksdata(Path(gt.path))[4]

    graph = td.graph.IndexedRXGraph()
    for axis in ("z", "y", "x"):
        graph.add_node_attr_key(axis, pl.Int64, 0)
    graph_ids = graph.bulk_add_nodes(
        [
            {"t": int(row[0]), "z": int(row[1]), "y": int(row[2]), "x": int(row[3])}
            for row in table.tolist()
        ]
    )
    to_graph = dict(zip(ordered, graph_ids))
    to_pred = {graph_id: node_id for node_id, graph_id in to_graph.items()}
    if pred_edges is not None:
        edge_rows = []
        for edge in pred_edges:
            if isinstance(edge, Mapping):
                src, dst = int(edge["source_id"]), int(edge["target_id"])
            else:
                src, dst = int(edge[0]), int(edge[1])
            if src in to_graph and dst in to_graph:
                edge_rows.append({"source_id": to_graph[src], "target_id": to_graph[dst]})
        if edge_rows:
            graph.bulk_add_edges(edge_rows)

    # 关掉 tracksdata 的进度条（与官方评测代码相同的做法），结束后恢复原设置。
    restore_progress = None
    try:
        from tracksdata.options import get_options, set_options

        previous = get_options().show_progress
        set_options(show_progress=False)
        restore_progress = lambda: set_options(show_progress=previous)  # noqa: E731
    except Exception:
        restore_progress = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            graph.match(gt_graph, matching=DistanceMatching(max_distance=float(max_distance_um), scale=tuple(gt.voxel_um)))
    finally:
        if restore_progress is not None:
            restore_progress()

    matched = graph.node_attrs(attr_keys=[keys.NODE_ID, keys.MATCHED_NODE_ID])
    result: dict[int, int] = {}
    for graph_id, gt_id in zip(matched[keys.NODE_ID].to_list(), matched[keys.MATCHED_NODE_ID].to_list()):
        if gt_id is not None and gt_id != -1:
            result[to_pred[graph_id]] = int(gt_id)
    return result


# =============================================================================================
# 内部实现：直接解析 Zarr v3（raw 后端）
# =============================================================================================
def _read_raw(path: Path):
    """不依赖图库，直接读 GEFF 的几个数组。行顺序 = 文件中的存储顺序。"""
    node_ids = read_zarr_v3_array(path / "nodes" / "ids").astype(np.int64).reshape(-1)
    count = len(node_ids)
    columns = {}
    for axis in ("t", "z", "y", "x"):
        prop_dir = path / "nodes" / "props" / axis
        if not (prop_dir / "values" / "zarr.json").is_file():
            if count == 0:
                columns[axis] = np.empty(0)
                continue
            raise ValueError(f"{path}：缺少节点属性 {axis}")
        values = read_zarr_v3_array(prop_dir / "values").reshape(-1)
        # GEFF 允许属性有缺失值（另存一个布尔数组 missing）；GT 坐标必须完整，缺失就报错。
        if (prop_dir / "missing" / "zarr.json").is_file():
            if read_zarr_v3_array(prop_dir / "missing").astype(bool).any():
                raise ValueError(f"{path}：节点属性 {axis} 有缺失值")
        if len(values) != count:
            raise ValueError(f"{path}：属性 {axis} 的长度 {len(values)} 与节点数 {count} 不一致")
        columns[axis] = values
    frames = np.asarray(columns["t"]).astype(np.int64)
    zyx = np.stack([np.asarray(columns[axis]) for axis in ("z", "y", "x")], axis=1).astype(np.float64)
    zyx = zyx.reshape(count, 3)
    edge_dir = path / "edges" / "ids"
    if (edge_dir / "zarr.json").is_file():
        edges = read_zarr_v3_array(edge_dir).astype(np.int64).reshape(-1, 2)
    else:
        edges = np.empty((0, 2), dtype=np.int64)
    return node_ids, frames, zyx, edges


def read_zarr_v3_array(array_dir: str | Path) -> np.ndarray:
    """按 Zarr v3 规范读出一个数组（只用 numpy + 解压库），返回内存中的 ndarray。

    Zarr v3 的数组 = 一个 ``zarr.json``（形状、数据类型、分块形状、编解码器链）+ 若干数据块文件。
    读取步骤：按分块网格逐块找到文件 → 按编解码器链**倒序**解码（先解压 zstd 等字节压缩，再按字节序
    还原数组，最后撤销转置）→ 放回整体数组中对应的位置。缺失的数据块等于填充值。
    支持的编解码器：bytes、transpose、zstd、gzip、blosc、crc32c；不支持分片（sharding）——GEFF 用不到。
    """
    array_dir = Path(array_dir)
    meta = json.loads((array_dir / "zarr.json").read_text(encoding="utf-8"))
    if meta.get("zarr_format") != 3 or meta.get("node_type") != "array":
        raise ValueError(f"{array_dir} 不是 Zarr v3 数组")
    shape = tuple(int(size) for size in meta["shape"])
    data_type = meta["data_type"]
    if not isinstance(data_type, str):
        raise NotImplementedError(f"{array_dir}：不支持的数据类型 {data_type!r}")
    dtype = np.dtype(data_type)
    grid = meta.get("chunk_grid", {})
    if grid.get("name") != "regular":
        raise NotImplementedError(f"{array_dir}：不支持的分块网格 {grid!r}")
    chunk_shape = tuple(int(size) for size in grid["configuration"]["chunk_shape"])
    key_encoding = meta.get("chunk_key_encoding", {"name": "default"})
    encoding_name = key_encoding.get("name", "default")
    separator = (key_encoding.get("configuration") or {}).get("separator", "/" if encoding_name == "default" else ".")
    codecs = meta.get("codecs", [])

    out = np.full(shape, _fill_value(meta.get("fill_value"), dtype), dtype=dtype)
    if any(size == 0 for size in shape):
        return out
    grid_counts = [math.ceil(size / chunk) for size, chunk in zip(shape, chunk_shape)]
    for chunk_index in itertools.product(*(range(count) for count in grid_counts)):
        chunk_file = array_dir / _chunk_key(chunk_index, encoding_name, separator)
        if not chunk_file.is_file():
            continue
        chunk = _decode_chunk(chunk_file.read_bytes(), codecs, dtype, chunk_shape)
        target = tuple(
            slice(index * chunk, min((index + 1) * chunk, size))
            for index, chunk, size in zip(chunk_index, chunk_shape, shape)
        )
        # 边缘处的数据块按完整的分块形状存储，只取落在数组范围内的部分。
        out[target] = chunk[tuple(slice(0, part.stop - part.start) for part in target)]
    return out


def _chunk_key(chunk_index: tuple[int, ...], encoding_name: str, separator: str) -> str:
    """数据块文件的相对路径。default 编码：``c/0/0``；v2 编码：``0.0``。"""
    if encoding_name == "default":
        return separator.join(["c", *(str(index) for index in chunk_index)])
    if encoding_name == "v2":
        return separator.join(str(index) for index in chunk_index) if chunk_index else "0"
    raise NotImplementedError(f"不支持的块键编码：{encoding_name!r}")


def _decode_chunk(raw: bytes, codecs: list, dtype: np.dtype, chunk_shape: tuple[int, ...]) -> np.ndarray:
    """按编解码器链倒序解码一个数据块。"""
    array_codecs, bytes_codec, byte_codecs = [], None, []
    for codec in codecs:
        name = codec["name"] if isinstance(codec, dict) else str(codec)
        config = (codec.get("configuration") if isinstance(codec, dict) else None) or {}
        if name == "sharding_indexed":
            raise NotImplementedError("不支持分片（sharding）数组")
        if bytes_codec is None and name == "transpose":
            array_codecs.append(config)
        elif bytes_codec is None and name == "bytes":
            bytes_codec = config
        elif bytes_codec is None:
            raise NotImplementedError(f"不支持的数组编解码器：{name!r}")
        else:
            byte_codecs.append((name, config))

    expected_nbytes = int(np.prod(chunk_shape, dtype=np.int64)) * dtype.itemsize
    data = raw
    for name, config in reversed(byte_codecs):
        data = _decode_bytes(name, data, expected_nbytes)

    endian = (bytes_codec or {}).get("endian", "little")
    stored_dtype = dtype.newbyteorder("<" if endian == "little" else ">") if dtype.itemsize > 1 else dtype
    stored_shape = list(chunk_shape)
    for config in array_codecs:
        order = [int(axis) for axis in config["order"]]
        stored_shape = [stored_shape[axis] for axis in order]
    array = np.frombuffer(data, dtype=stored_dtype)
    if array.size != int(np.prod(stored_shape, dtype=np.int64)):
        raise ValueError(f"数据块大小 {array.size} 与分块形状 {chunk_shape} 不符")
    array = array.reshape(stored_shape)
    for config in reversed(array_codecs):
        array = np.transpose(array, np.argsort([int(axis) for axis in config["order"]]))
    return array.astype(dtype.newbyteorder("="), copy=False)


def _decode_bytes(name: str, data: bytes, expected_nbytes: int) -> bytes:
    """解一层字节编解码器。"""
    if name == "zstd":
        return _zstd_decompress(data)
    if name == "gzip":
        return gzip.decompress(data)
    if name == "crc32c":
        # 末尾 4 字节是校验和；这里只读不校验。
        return data[:-4]
    if name == "blosc":
        try:
            import blosc2

            return bytes(blosc2.decompress(data))
        except ImportError:
            import numcodecs

            return bytes(numcodecs.Blosc().decode(data))
    raise NotImplementedError(f"不支持的字节编解码器：{name!r}（期望输出 {expected_nbytes} 字节）")


def _zstd_decompress(data: bytes) -> bytes:
    """zstd 解压：依次尝试 Python 3.14 标准库、zstandard、numcodecs，用第一个可用的。"""
    try:
        from compression import zstd as stdlib_zstd  # Python ≥ 3.14

        return stdlib_zstd.decompress(data)
    except ImportError:
        pass
    try:
        import zstandard

        # 流式解压：即使数据帧头里没有写原始长度也能正确解出。
        return zstandard.ZstdDecompressor().decompressobj().decompress(data)
    except ImportError:
        pass
    try:
        import numcodecs

        return bytes(numcodecs.Zstd().decode(data))
    except ImportError as exc:
        raise ImportError("读取 zstd 压缩的 GEFF 需要 zstandard 或 numcodecs（或 Python ≥ 3.14）") from exc


def _fill_value(value: Any, dtype: np.dtype):
    """把 zarr.json 里的 fill_value 转成 numpy 标量（支持 "NaN"、"Infinity" 等写法）。"""
    if value is None:
        return np.zeros((), dtype=dtype)[()]
    if isinstance(value, str):
        special = {"NaN": np.nan, "Infinity": np.inf, "-Infinity": -np.inf}
        if value in special:
            return special[value]
        if value.startswith("0x"):
            raw = int(value, 16).to_bytes(dtype.itemsize, "big")
            return np.frombuffer(raw, dtype=dtype.newbyteorder(">"))[0]
    return np.asarray(value).astype(dtype)[()]


def _read_group_attributes(path: Path) -> dict:
    """读一个 Zarr 组的根属性：v3 在 ``zarr.json`` 的 ``attributes`` 里，v2 在 ``.zattrs`` 里。"""
    v3 = path / "zarr.json"
    if v3.is_file():
        attrs = json.loads(v3.read_text(encoding="utf-8")).get("attributes", {})
        return attrs if isinstance(attrs, dict) else {}
    v2 = path / ".zattrs"
    if v2.is_file():
        attrs = json.loads(v2.read_text(encoding="utf-8"))
        return attrs if isinstance(attrs, dict) else {}
    return {}


def _find_key_recursive(obj: Any, key: str) -> Any:
    """在嵌套的 dict / list 中查找第一个名为 ``key`` 的值（与推理脚本第 8 段的同名函数相同）。"""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find_key_recursive(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_key_recursive(item, key)
            if found is not None:
                return found
    return None


def _as_points(values: Any) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return points.reshape(-1, 3)


def _node_tzyx(node: Any) -> tuple[int, float, float, float]:
    """从节点字典或 (t, z, y, x) 元组中取出帧号和体素坐标。"""
    if isinstance(node, Mapping):
        return int(node["t"]), float(node["z"]), float(node["y"]), float(node["x"])
    frame, z, y, x = node[:4]
    return int(frame), float(z), float(y), float(x)
