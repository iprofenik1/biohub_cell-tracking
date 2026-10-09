"""第 1 部分 · 第 2 步（CPU）：把 step1 抓取的检测特征与 GT 逐帧配对，生成坐标头的训练对 pairs.npz。

用法（在仓库根目录下运行；Kaggle CPU notebook 或本地均可）::

    # Kaggle：把 step1 各分片的 notebook 输出都挂载为输入，--capture 默认 auto 会自动找到它们
    python part1_cv10_coord_head/step2_build_pairs.py --out /kaggle/working/coord_pairs/pairs.npz

    # 也可以显式给出抓取目录与训练集目录（例如在本地运行）
    python part1_cv10_coord_head/step2_build_pairs.py \
        --capture <分片0>/coord_capture <分片1>/coord_capture <分片2>/coord_capture <分片3>/coord_capture \
        --train-dir <竞赛数据>/train --out coord_pairs/pairs.npz

为什么需要“配对”
================
坐标头要学的是“检测点离真实细胞中心差多少”。step1 只导出了每个检测点的位置和 224 维特征，
并不知道它对应哪个真实细胞；GT（``train/<影片>.geff``）只标注了约 2.8% 的细胞。所以：

1. 逐帧把检测点与 GT 节点做**贪心一对一最近邻**匹配，半径 3.5 µm（按距离从小到大，已被用过的
   检测点或 GT 不再参与；距离相同时依次按检测下标、GT 下标决定先后，结果完全确定）；
2. 只有配上的检测点才有监督信号，回归目标 = GT 坐标 − 检测坐标（单位 µm，顺序 z, y, x）；
3. 没配上的检测点（绝大多数）不进训练集。但推理时头会移动**每一个**检测点，所以另外从每部影片
   的全部检测里固定抽 400 个“背景样本”（bg_*），step3 用它们统计头在一般检测点上的平均位移。

单位（最容易出错的地方）
========================
- 检测坐标在降采样网格上：z 不变、y/x 每 4 个像素取 1 个，三个方向格距都是 1.625 µm，
  所以“检测 µm = 网格坐标 × 1.625”；
- GT 坐标是原始体素下标：z 1.625 µm、y/x 0.40625 µm，所以“GT µm = 体素坐标 × (1.625, 0.40625, 0.40625)”；
- 两者都换成 µm 之后才能比较距离、相减得到目标。直接拿网格坐标和体素坐标相减，y/x 会错 4 倍。

为什么目标不截断到 2 µm
=======================
头的有界输出模长恒小于 2 µm（见 coord_head_module.bounded），而配对半径是 3.5 µm，少数目标会超过 2 µm
（例如 z 差两层 = 3.25 µm，或 z 差一层且 y/x 合计再偏 1.2 µm 以上）。这里**不**截断目标，原因有二：

1. 损失是**逐轴**计算的（Huber 分别作用在 z、y、x 三个分量上）。如果按模长把目标缩到 2 µm，同一行的
   y、x 分量也会被一起缩小，等于给它们一个错误的监督值；不截断时，每个轴的目标都是真实的 GT − 检测。
2. 够不到的那部分落在 Huber 损失的线性段：误差超过 1 µm 后梯度大小恒定、不再随误差增大，这些样本只会
   把输出推向正确方向（饱和），不会主导训练。

输出 pairs.npz 的键（每行一个训练对；行顺序 = 影片顺序 → 每部影片内按 GT 帧号升序 → 帧内按匹配先后）
====================================================================================================
- ``movies``      (M,) 字符串：影片名，顺序即 movie_idx 的编号；
- ``movie_idx``   (n,) int16：该行属于第几部影片；
- ``rows``        (n,) int32：该检测点在本影片抓取数据（按帧号拼接）中的行号；
- ``coords``      (n, 4) int16：检测点的网格坐标 [t, z, y, x]；
- ``features``    (n, 224) float32：坐标头的输入特征（与推理时完全相同的那一份）；
- ``target``      (n, 3) float32：GT − 检测（µm，z, y, x），**不截断**；
- ``gt_voxel``    (n, 3) float32：配上的 GT 体素坐标（便于检查）；
- ``bg_movie_idx`` / ``bg_rows`` / ``bg_coords`` / ``bg_features``：每部影片随机 400 个检测点（背景样本）。

同目录还会写 ``pairs_summary.json``：每个胚胎的配对数、平均偏差、z 方向“恰好 0 层 / 1 层”的比例等，
可以直观看到 z 方向的层间量化误差有多大。

影片顺序（决定行顺序，从而决定 step3 的训练结果）
================================================
默认：capture 目录里找到的、同时有 GT 的全部影片，**按名字排序**。step3 的分折与影片顺序无关
（按影片名哈希分折），但 mini-batch 的打乱是对“行下标”做随机排列，行顺序不同，训练出的权重就会
有微小差别。要与某个已有的 pairs.npz 逐位一致，用 ``--movie-order`` 指定同一顺序（可以直接传那个
pairs.npz，脚本会读其中的 ``movies``）。

抓取数据的两种目录布局（都支持）
================================
- 推理模块 capture 模式的布局（step1 产物）：``<capture>/<影片>/<帧号:04d>.npz``，键 ``coords``、``features``；
- 按影片合并的布局：``<capture>/<影片>.npz``（同样的键，按帧号升序拼接）。文件里多出的键（例如
  ``heat27``）会被忽略。两种布局拼出来的行顺序相同：每帧只在第一次出现的滑窗里检测一次，帧号升序。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

# 让 ``from common import gt_io`` 在“从仓库根目录运行本脚本”时可用。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import gt_io  # noqa: E402

# 检测网格的格距（µm）：三个方向都是 1.625 µm。
GRID_UM = 1.625
# GT 体素尺寸（µm，z, y, x）。写成常量而不是用每部影片读出的值：两者相等（读 GT 时已校验），
# 用同一个常量保证浮点计算与原始训练数据逐位一致。
GT_VOXEL_UM = np.array([1.625, 0.40625, 0.40625])
# 配对半径（µm）：检测与 GT 的平均偏差约 1.6 µm（z 层间量化为主），在 3.5 µm 内约 92% 的 GT 节点
# 能配到检测点；这个半径又明显小于相邻细胞核的间距，错配很少。
MATCH_UM = 3.5
# 每部影片抽取的背景样本数（只用于统计头在全部检测点上的位移，不参与训练）。
BG_PER_MOVIE = 400
# 坐标头的输入维度：7 个采样位置 × 32 个通道。
FEATURE_DIMS = 224
# 检测网格每个轴 64 个格点（0..63）。
GRID_SIZE = 64

# 逐帧贪心一对一最近邻匹配。实现放在 common/gt_io.py（与原始训练数据制作代码逐行一致）：
#   1) 用 KD 树找出所有距离 ≤ 半径的 (检测 i, GT j) 对；
#   2) 按 (距离, i, j) 升序排序——np.lexsort 以最后一个键为主键，所以写成 lexsort((j, i, 距离))；
#   3) 依次取对，检测或 GT 已被占用就跳过。
# 为什么用贪心而不是最优二分匹配：这里要的是“可靠的近邻对”而不是“配上的总数最多”；贪心从最近的
# 对开始，每个 GT 只认领离它最近的那个检测点，既简单又可完全复现。（官方指标的 7 µm 节点匹配才是
# 最优二分匹配，见第 2 部分。）
match_frame = gt_io.greedy_match_frame


# =============================================================================================
# 读取抓取数据
# =============================================================================================
def _movie_capture_path(capture_dirs: list[Path], movie: str) -> Path | None:
    """在若干个抓取目录（每个分片一个）中找这部影片：先找按帧布局的目录，再找按影片合并的 .npz。"""
    for root in capture_dirs:
        frame_dir = root / movie
        if frame_dir.is_dir():
            return frame_dir
        movie_file = root / f"{movie}.npz"
        if movie_file.is_file():
            return movie_file
    return None


def list_captured_movies(capture_dirs: list[Path]) -> list[str]:
    """抓取目录里出现过的全部影片名（两种布局都算），排序后返回。"""
    found = set()
    for root in capture_dirs:
        if not root.is_dir():
            raise FileNotFoundError(f"抓取目录不存在：{root}")
        for entry in root.iterdir():
            if entry.is_dir() and any(p.suffix == ".npz" and p.stem.isdigit() for p in entry.iterdir()):
                found.add(entry.name)
            elif entry.is_file() and entry.suffix == ".npz" and not entry.name.startswith("."):
                found.add(entry.stem)
    return sorted(found)


def _check_arrays(coords: np.ndarray, features: np.ndarray, where: str) -> tuple[np.ndarray, np.ndarray]:
    """抓取数据的格式检查：坐标是 (N, 4) 整数网格坐标，特征是 (N, 224)。"""
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError(f"{where}：coords 形状应为 (N, 4)，实际 {coords.shape}")
    if not np.issubdtype(coords.dtype, np.integer):
        # capture 模式在坐标修正“之前”保存，坐标一定是整数网格点；出现浮点说明抓错了阶段。
        raise ValueError(f"{where}：coords 应为整数网格坐标，实际 dtype {coords.dtype}")
    if features.shape != (len(coords), FEATURE_DIMS):
        raise ValueError(f"{where}：features 形状应为 ({len(coords)}, {FEATURE_DIMS})，实际 {features.shape}")
    if len(coords) and (coords[:, 1:].min() < 0 or coords[:, 1:].max() >= GRID_SIZE):
        raise ValueError(f"{where}：网格坐标超出 [0, {GRID_SIZE - 1}]")
    # int16 / float32 与推理模块保存的类型一致；已经是这个类型时不复制。
    return coords.astype(np.int16, copy=False), features.astype(np.float32, copy=False)


def load_capture(capture_dirs: list[Path], movie: str) -> tuple[np.ndarray, np.ndarray]:
    """读一部影片的全部抓取数据，返回 (coords (N,4) int16, features (N,224) float32)，按帧号升序拼接。

    只读取 ``coords`` 和 ``features`` 两个键，其它键（如 ``heat27``）忽略。
    没有检测点的帧，推理模块不会写文件（refine 遇到空帧直接返回），所以帧号可以不连续。
    """
    path = _movie_capture_path(capture_dirs, movie)
    if path is None:
        raise FileNotFoundError(f"{movie}：在 {[str(p) for p in capture_dirs]} 中找不到抓取数据")
    if path.is_file():
        with np.load(path) as data:
            return _check_arrays(data["coords"], data["features"], str(path))
    frames = sorted((int(p.stem), p) for p in path.iterdir() if p.suffix == ".npz" and p.stem.isdigit())
    coords_parts, feature_parts = [], []
    for t, frame_path in frames:
        with np.load(frame_path) as data:
            coords, features = _check_arrays(data["coords"], data["features"], str(frame_path))
        if len(coords) and not np.all(coords[:, 0] == t):
            raise ValueError(f"{frame_path}：coords 第 0 列（帧号）与文件名 {t} 不一致")
        coords_parts.append(coords)
        feature_parts.append(features)
    if not coords_parts:
        return np.empty((0, 4), np.int16), np.empty((0, FEATURE_DIMS), np.float32)
    return np.concatenate(coords_parts), np.concatenate(feature_parts)


# =============================================================================================
# 一部影片的配对
# =============================================================================================
def pair_movie(job: tuple) -> dict:
    """一部影片：读抓取数据与 GT，逐帧配对，返回该影片的训练对与背景样本。

    ``job`` = (抓取目录列表, 训练数据目录, 影片名, 配对半径 µm, GT 读取后端, 每部影片背景样本数)。
    写成单个元组参数，方便交给进程池的 map（map 会保持输入顺序，所以并行不改变行顺序）。
    """
    capture_dirs, train_dir, movie, radius, gt_backend, bg_per_movie = job
    # 读 GT，并核对影像的体素尺寸确实是 (1.625, 0.40625, 0.40625) µm；不一致就报错，
    # 宁可停下来也不要产生单位错误的训练目标。
    gt = gt_io.load_train_gt(train_dir, movie, backend=gt_backend, check_scale=True)
    coords, features = load_capture([Path(p) for p in capture_dirs], movie)

    gt_t = gt.t.astype(np.int64)
    gt_vox = gt.zyx.astype(np.float64)
    rows, gts = [], []
    frames_with_gt = 0
    # 只遍历有 GT 的帧（稀疏标注：很多帧没有 GT），帧号升序。
    for t in np.unique(gt_t):
        det_rows = np.flatnonzero(coords[:, 0] == t)
        g = np.flatnonzero(gt_t == t)
        if len(det_rows):
            frames_with_gt += 1
        # 两边都先换成 µm 再匹配：检测 = 网格 × 1.625；GT = 体素 × (1.625, 0.40625, 0.40625)。
        for i, j in match_frame(coords[det_rows, 1:].astype(np.float64) * GRID_UM,
                                gt_vox[g] * GT_VOXEL_UM, radius):
            rows.append(det_rows[i])
            gts.append(g[j])
    rows = np.asarray(rows, dtype=np.int64)
    gts = np.asarray(gts, dtype=np.int64)
    det_um = coords[rows, 1:].astype(np.float64) * GRID_UM
    # 回归目标：GT − 检测（µm，z, y, x）。先用 float64 计算再转 float32，与原始训练数据逐位一致。
    target = gt_vox[gts] * GT_VOXEL_UM - det_um

    # 背景样本：随机种子取自影片名的 SHA-1（前 4 字节，小端），所以每次运行、每台机器抽到的都一样。
    seed = int.from_bytes(hashlib.sha1(movie.encode()).digest()[:4], "little")
    bg = np.sort(np.random.default_rng(seed).choice(len(coords), size=min(bg_per_movie, len(coords)),
                                                    replace=False))
    return {
        "movie": movie,
        "rows": rows.astype(np.int32),
        "coords": coords[rows],
        "features": features[rows],
        "target": target.astype(np.float32),
        "gt_voxel": gt_vox[gts].astype(np.float32),
        "bg_rows": bg.astype(np.int32),
        "bg_coords": coords[bg],
        "bg_features": features[bg],
        "n_det": int(len(coords)),
        "n_gt": int(gt.num_nodes),
        "frames_with_gt": frames_with_gt,
        "gt_backend": gt.backend,
    }


# =============================================================================================
# 影片顺序
# =============================================================================================
def read_movie_order(path: Path) -> list[str]:
    """读取显式的影片顺序：.npz（取其中的 ``movies``，例如一个已有的 pairs.npz）、.json（字符串列表）
    或文本文件（每行一个影片名，忽略空行和以 # 开头的行）。"""
    if path.suffix == ".npz":
        with np.load(path) as data:
            return [str(m) for m in data["movies"]]
    if path.suffix == ".json":
        movies = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(movies, list):
            raise ValueError(f"{path}：JSON 必须是影片名列表")
        return [str(m) for m in movies]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


# =============================================================================================
# 汇总
# =============================================================================================
def summarize(results: list[dict], merged: dict, movies: list[str], match_um: float) -> dict:
    """按胚胎统计配对数与偏差，帮助理解坐标头要修正的误差是什么样的。"""
    target = merged["target"]
    dist = np.linalg.norm(target, axis=1)
    embryo = np.array([movies[i].split("_")[0] for i in merged["movie_idx"]])
    summary = {
        "match_um": match_um, "movies": len(movies), "pairs": int(len(target)),
        "gt_nodes": int(sum(r["n_gt"] for r in results)),
        "detections": int(sum(r["n_det"] for r in results)),
        "bg_rows": int(len(merged["bg_rows"])),
        "gt_backend": sorted({r["gt_backend"] for r in results}),
        "by_embryo": {},
    }
    for e in sorted(set(embryo)):
        m = embryo == e
        r = target[m]
        summary["by_embryo"][e] = {
            "pairs": int(m.sum()), "mean_um": float(dist[m].mean()), "median_um": float(np.median(dist[m])),
            "p90_um": float(np.percentile(dist[m], 90)),
            "mean_abs_zyx": np.abs(r).mean(axis=0).round(4).tolist(),
            "bias_zyx": r.mean(axis=0).round(4).tolist(),
            # z 方向的层间量化：dz 恰好为 0（同一层）或恰好 ±1 层（1.625 µm）的比例。
            # GT 的 z 也是整数层号，所以 z 误差绝大多数取这两种值（其余是 ±2 层）——网格量化误差主要就在这里。
            "dz0": float(np.mean(np.isclose(r[:, 0], 0))),
            "dz1": float(np.mean(np.isclose(np.abs(r[:, 0]), GRID_UM, atol=0.01))),
            # 超过 2 µm 的目标：有界输出够不到，只能尽量靠近（不截断目标，见文件头说明）。
            "over_2um": float(np.mean(dist[m] > 2.0)),
        }
    summary["per_movie"] = {r["movie"]: {"pairs": int(len(r["rows"])), "gt": r["n_gt"], "det": r["n_det"],
                                         "frames_with_gt": r["frames_with_gt"]} for r in results}
    return summary


# =============================================================================================
# 主流程
# =============================================================================================
def default_train_dir() -> Path:
    """竞赛训练集目录（Kaggle 上的两种挂载位置取存在的那个）。"""
    from common import pipeline_runner

    return pipeline_runner.competition_dir() / "train"


def discover_capture_dirs() -> list[Path]:
    """``--capture auto``：找出 Kaggle 上所有名为 ``coord_capture`` 的目录。

    step1 的每个分片各自是一个 notebook 输出，挂载到本 notebook 后位于 ``/kaggle/input/<...>/coord_capture``
    （层级随挂载方式不同）。这里只在固定的几层深度上找（不递归整个 /kaggle/input：竞赛数据的 zarr
    目录里有大量小文件，全量递归很慢），再加上本会话自己的 ``/kaggle/working/coord_capture``。
    """
    found = []
    working = Path("/kaggle/working/coord_capture")
    if working.is_dir():
        found.append(working)
    root = Path("/kaggle/input")
    if root.is_dir():
        for depth in range(1, 5):
            pattern = "/".join(["*"] * depth) + "/coord_capture"
            found.extend(sorted(p for p in root.glob(pattern) if p.is_dir()))
    return list(dict.fromkeys(p.resolve() for p in found))


def build_pairs(capture_dirs: list[Path], train_dir: Path, movies: list[str], *, match_um: float = MATCH_UM,
                gt_backend: str = "auto", bg_per_movie: int = BG_PER_MOVIE, workers: int = 1) -> tuple[dict, list[dict]]:
    """对 ``movies``（按给定顺序）逐部配对并合并，返回 (合并后的数组字典, 每部影片的结果列表)。"""
    jobs = [(tuple(str(p) for p in capture_dirs), str(train_dir), m, match_um, gt_backend, bg_per_movie)
            for m in movies]
    if workers > 1:
        # 进程池的 map 按输入顺序返回结果，所以并行与串行得到的行顺序完全相同。
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(pair_movie, jobs, chunksize=2))
    else:
        results = [pair_movie(job) for job in jobs]
    arrays: dict[str, list] = {}
    for index, result in enumerate(results):
        n, nb = len(result["rows"]), len(result["bg_rows"])
        arrays.setdefault("movie_idx", []).append(np.full(n, index, np.int16))
        arrays.setdefault("bg_movie_idx", []).append(np.full(nb, index, np.int16))
        for key in ("rows", "coords", "features", "target", "gt_voxel",
                    "bg_rows", "bg_coords", "bg_features"):
            arrays.setdefault(key, []).append(result[key])
    merged = {key: np.concatenate(value) for key, value in arrays.items()}
    merged["movies"] = np.asarray(movies)
    return merged, results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把检测特征与 GT 逐帧配对，生成 10 折坐标头的训练对。")
    parser.add_argument("--capture", nargs="+", default=["auto"],
                        help="step1 的抓取目录，可以给多个（每个分片一个）；也接受按影片合并的 <影片>.npz 布局。"
                             "默认 auto：自动查找 /kaggle/working 与 /kaggle/input 下所有 coord_capture 目录")
    parser.add_argument("--train-dir", type=Path, default=None,
                        help="竞赛训练集目录（含 <影片>.geff 与 <影片>.zarr）；默认 Kaggle 竞赛数据的 train/")
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/coord_pairs/pairs.npz"),
                        help="输出的 pairs.npz 路径；同目录写 pairs_summary.json")
    parser.add_argument("--movie-order", type=Path, default=None,
                        help="显式的影片顺序：文本（每行一个）、JSON 列表，或含 movies 键的 .npz（如已有的 pairs.npz）")
    parser.add_argument("--match-um", type=float, default=MATCH_UM, help="逐帧配对半径（µm），默认 3.5")
    parser.add_argument("--bg-per-movie", type=int, default=BG_PER_MOVIE, help="每部影片的背景样本数，默认 400")
    parser.add_argument("--gt-backend", choices=("auto", "tracksdata", "raw"), default="auto",
                        help="GT 读取方式；要与原始训练数据逐位一致请用 tracksdata（节点行顺序相同）")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="并行进程数")
    parser.add_argument("--allow-missing", action="store_true",
                        help="--movie-order 中有影片缺抓取数据时跳过它，检测点数与 step1 清单不一致时只警告（默认都直接报错）")
    args = parser.parse_args(argv)

    started = time.time()
    if args.capture == ["auto"]:
        capture_dirs = discover_capture_dirs()
        if not capture_dirs:
            raise SystemExit("--capture auto 没有找到任何 coord_capture 目录；请把 step1 各分片的输出挂载为输入，或显式传入路径")
        print("自动找到的抓取目录：" + ", ".join(str(p) for p in capture_dirs), flush=True)
    else:
        capture_dirs = [Path(p).resolve() for p in args.capture]
    train_dir = (args.train_dir or default_train_dir()).resolve()
    if args.movie_order is not None:
        movies = read_movie_order(args.movie_order)
        if len(set(movies)) != len(movies):
            raise SystemExit("--movie-order 中有重复的影片名")
    else:
        # 默认顺序：抓取目录里找到的影片按名字排序（只保留有 GT 的训练影片）。
        movies = [m for m in list_captured_movies(capture_dirs) if (train_dir / f"{m}.geff").is_dir()]
    missing = [m for m in movies if _movie_capture_path(capture_dirs, m) is None]
    if missing:
        if not args.allow_missing:
            raise SystemExit(f"{len(missing)} 部影片缺抓取数据，例如 {missing[:3]}（确认所有分片的输出都已传入 --capture）")
        movies = [m for m in movies if m not in set(missing)]
    no_gt = [m for m in movies if not (train_dir / f"{m}.geff").is_dir()]
    if no_gt:
        raise SystemExit(f"{len(no_gt)} 部影片没有 GT（{train_dir}/<影片>.geff），例如 {no_gt[:3]}")
    if not movies:
        raise SystemExit("没有可用的影片")
    print(f"{len(movies)} 部影片，配对半径 {args.match_um} µm，{args.workers} 个进程", flush=True)

    merged, results = build_pairs(capture_dirs, train_dir, movies, match_um=args.match_um,
                                  gt_backend=args.gt_backend, bg_per_movie=args.bg_per_movie,
                                  workers=args.workers)
    # 完整性检查：step1 每个分片都写了 capture_manifest_shard*.json（每部影片的检测点数）。
    # 如果某个分片的输出被截断或只拷了一部分，读到的检测点会变少，配对会悄悄变少而不报错，所以这里核对一遍。
    expected = {}
    for root in capture_dirs:
        for manifest_path in sorted(root.glob("capture_manifest_shard*.json")):
            for movie, info in json.loads(manifest_path.read_text(encoding="utf-8")).get("movies", {}).items():
                expected[movie] = int(info["detections"])
    mismatched = [(r["movie"], r["n_det"], expected[r["movie"]]) for r in results
                  if r["movie"] in expected and r["n_det"] != expected[r["movie"]]]
    if mismatched:
        message = f"{len(mismatched)} 部影片的检测点数与 step1 清单不一致（读到 / 清单）：{mismatched[:3]}"
        if not args.allow_missing:
            raise SystemExit(message + "；请检查各分片输出是否完整（或加 --allow-missing 忽略）")
        print("警告：" + message, flush=True)
    print(f"抓取清单核对：{len(expected)} 部影片有清单记录，{len(mismatched)} 部不一致", flush=True)

    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # 先写临时文件再改名：中途失败不会留下一个看起来完整、其实残缺的 pairs.npz。
    temporary = out.parent / f".{out.stem}.tmp.npz"
    np.savez(temporary, **merged)
    temporary.replace(out)

    summary = summarize(results, merged, movies, args.match_um)
    summary["seconds"] = round(time.time() - started, 1)
    (out.parent / "pairs_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_movie"}, indent=1, ensure_ascii=False))
    print(f"已写出 {out}（{len(merged['target']):,} 个训练对，{len(merged['bg_rows']):,} 个背景样本）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
