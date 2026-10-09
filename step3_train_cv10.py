"""第 1 部分 · 第 3 步（CPU）：10 折 × 5 个种子训练坐标头，输出 fold0.pt … fold9.pt（即 10 折坐标头）。

用法（在仓库根目录下运行）::

    python part1_cv10_coord_head/step3_train_cv10.py \
        --pairs /kaggle/working/coord_pairs/pairs.npz \
        --out /kaggle/working/biohub-cv10-coord-head

输出目录里的 10 个 ``fold*.pt`` 上传为 Kaggle 数据集 ``biohub-cv10-coord-head`` 后，第三部分的推理管线
（第 4 段）会找到它们，对 10 个头的有界位移取平均来修正每个检测点的坐标。

训练配方（与部署的 10 个头完全相同，逐行对应原始训练代码）
==========================================================
- 网络：``coord_head_module.make_head()``——Linear(224, 32) → SiLU → Linear(32, 3)，最后一层零初始化；
  输出经 ``coord_head_module.bounded()`` 变成模长 < 2 µm 的位移。直接 import 推理模块里的这两个函数，
  保证“训练的网络”和“推理加载的网络”是同一份代码。
- 输入标准化：每一折只用**该折的训练行**算均值和标准差（numpy float32，ddof=0，标准差下限 1e-3），
  与权重一起存进 .pt；推理时每个头用自己的 mean/scale。
- 损失：Huber（delta = 1 µm），作用在有界输出与目标（GT − 检测，µm）之间，对 n×3 个元素取平均。
- 优化：AdamW（lr 1e-3，weight_decay 1e-4，其余默认），batch 1024（最后不足一批的也用），最多 30 个 epoch；
  每个 epoch 结束后在验证折上算一次全量 Huber，比历史最好低 1e-6 以上才算进步，连续 5 个 epoch
  没有进步就停，保留最好的那个 epoch 的权重。
- 10 折：按影片分折、按胚胎分层（见 ``folds_of``）。第 k 个头用其余 9 折训练，第 k 折是它**唯一**的
  验证集：早停与“5 个种子（0–4）里选最好的”都只看第 k 折，不再在训练集内部切分。

逐位复现需要固定的东西
======================
训练是完全确定的：同样的 pairs.npz（同样的行顺序）、同样的 torch / numpy 版本、同样的 CPU 线程数、
同类 CPU（指令集决定矩阵乘法的实现），就能得到逐字节相同的 fold*.pt。随机性只有两个来源，而且都被固定：
1. 网络初始化：``torch.manual_seed(seed)`` 紧挨着 ``make_head()``——第一层的初值取决于全局随机数
   生成器当时的状态，中间多建一个层、多抽一个随机数，所有权重都会变；
2. 每个 epoch 的样本打乱：用一个**独立的** ``torch.Generator().manual_seed(seed)`` 生成 ``randperm``，
   与全局生成器互不干扰。
CPU 上矩阵乘法的求和顺序与线程数有关，所以 ``--threads`` 默认 32（部署的头就是 32 线程训练的）。
换线程数或换机器，每一步只差浮点舍入，但几千步训练会把它累积成权重上的微小差别（效果上等价）；
早停阈值 1e-6 还可能把它放大成“最佳 epoch 不同”。所以要逐位复现必须全部一致。只想自己训练一套
能用的头时，``--threads`` 设成 CPU 核数即可。

输出
====
- ``fold{k}.pt``：``{"state_dict": 网络权重, "mean": (224,) float32, "scale": (224,) float32}``，
  用 ``torch.save`` 默认参数保存（推理端用 ``torch.load(..., weights_only=True)`` 读取）；
- ``ensemble.json``：10 个头的文件名（折号顺序）；``head_map_oof.json``：影片 → 把它留作验证的那个头；
- ``metrics.json``：每折的验证 Huber、最佳 epoch、选中的种子、行数、验证影片；折外（OOF）残差统计；
  背景样本上的平均位移；
- ``head_meta.json``：配方、超参数、输入数据与模块的 sha256、每个头的 sha256、运行环境、耗时。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 直接导入推理期的坐标头模块（与本脚本同目录），训练与推理共用 make_head / bounded。
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from coord_head_module import bounded, make_head  # noqa: E402

# ---------------------------------------------------------------------------------------------
# 常量（单位都写清楚）
# ---------------------------------------------------------------------------------------------
GRID_UM = 1.625                                   # 检测网格格距（µm），三个方向相同
GT_VOXEL_UM = np.array([1.625, 0.40625, 0.40625])  # 原始体素尺寸（µm，z, y, x）
DOWNSAMPLE = np.array([1, 4, 4])                   # 网格 → 原始体素：z 不变，y/x 乘 4
GRID_MAX = 63                                     # 网格每轴 64 个格点，坐标范围 [0, 63]
HUBER_DELTA = 1.0                                 # Huber 损失的拐点（µm）
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1024
MAX_EPOCHS = 30
PATIENCE = 5                                      # 连续这么多个 epoch 没有进步就停
MIN_IMPROVEMENT = 1e-6                            # 验证损失至少要降这么多才算进步
SCALE_FLOOR = 1e-3                                # 标准差下限，防止常数特征除以 0
HEAD_NAME = "biohub-cv10-coord-head"              # 第三部分推理管线查找的数据集名


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# =============================================================================================
# 分折
# =============================================================================================
def folds_of(movies: list[str], k: int) -> dict[str, int]:
    """把影片分成 k 折：先按胚胎分组（影片名前缀 44b6 / 6bba），组内按影片名的 SHA-1 十六进制串排序，
    再按下标轮流发牌（第 i 部影片 → 第 i % k 折）。

    为什么按**影片**分折：同一部影片的相邻帧几乎一样，如果同一部影片的检测点同时出现在训练和验证里，
    验证损失会因为“见过几乎相同的样本”而虚低，早停和选种子就失去意义。按影片分折，验证的是
    “对没见过的影片能修正多少”——这正是测试时的情形。
    为什么按**胚胎**分层：两个胚胎的成像条件和细胞密度不同（6bba 的配对数是 44b6 的 5 倍多），
    分层后每一折里两个胚胎的比例都和整体接近，10 个头之间可比。
    为什么用哈希排序而不是随机打乱：完全确定、可复现，而且与影片在输入文件中的顺序无关。
    """
    out = {}
    for embryo in sorted({m.split("_")[0] for m in movies}):
        group = sorted((m for m in movies if m.startswith(embryo + "_")),
                       key=lambda m: hashlib.sha1(m.encode()).hexdigest())
        for index, movie in enumerate(group):
            out[movie] = index % k
    return out


# =============================================================================================
# 训练一个头（一折）
# =============================================================================================
def train_head(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray, seeds: list[int],
               epochs: int = MAX_EPOCHS, batch: int = BATCH_SIZE, verbose: bool = False):
    """用训练行 (x_tr, y_tr) 训练、只用验证行 (x_va, y_va) 早停和选种子，返回 (头, mean, scale, 信息)。

    计算顺序与部署的头完全相同；为了逐位复现，下面每一行的顺序、类型转换都不要改。
    """
    # 标准化参数只来自训练行：验证折（以及测试影片）的统计量不能泄漏进训练。
    # numpy 在 float32 上求均值 / 标准差（ddof=0），结果再存成 float32。
    mean = x_tr.mean(axis=0).astype(np.float32)
    scale = np.maximum(x_tr.std(axis=0), SCALE_FLOOR).astype(np.float32)
    # Huber 损失：|误差| < 1 µm 时是平方损失（像 L2，精细拟合常见的亚体素误差）；超过 1 µm 时变成线性
    # （像 L1）。配对半径 3.5 µm 难免混入错配或标注偏差，线性部分让这些离群样本的梯度有上限，
    # 不会主导训练。默认对 batch 内全部 n×3 个元素（每个坐标轴单独算）取平均。
    loss_fn = torch.nn.HuberLoss(delta=HUBER_DELTA)
    xt = torch.as_tensor((x_tr - mean) / scale, dtype=torch.float32)
    yt = torch.as_tensor(y_tr, dtype=torch.float32)
    xv = torch.as_tensor((x_va - mean) / scale, dtype=torch.float32)
    yv = torch.as_tensor(y_va, dtype=torch.float32)
    best = None
    for seed in seeds:
        # 种子紧挨着建网络：初始化只依赖这个种子。最后一层是零初始化（make_head 里），训练从
        # “输出恒为 0 = 完全相信检测器”出发，只学能降低损失的修正。
        torch.manual_seed(seed)
        head = make_head()
        optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # 打乱顺序用独立的生成器：与网络初始化互不影响，每个种子的打乱序列也固定。
        generator = torch.Generator().manual_seed(seed)
        seed_best = (float("inf"), None, -1)
        stale = 0
        for epoch in range(epochs):
            head.train()
            order = torch.randperm(len(xt), generator=generator)
            for start in range(0, len(order), batch):
                index = order[start:start + batch]
                optimizer.zero_grad()
                # 损失直接作用在有界输出上：训练与推理用同一个映射，网络学到的就是推理时真正施加的位移。
                # 目标超过 2 µm 时有界输出够不到，梯度会把它推向饱和方向——方向仍然是对的。
                loss = loss_fn(bounded(head, xt[index]), yt[index])
                loss.backward()
                optimizer.step()
            head.eval()
            with torch.no_grad():
                val = float(loss_fn(bounded(head, xv), yv))
            # 早停：验证损失比历史最好至少低 MIN_IMPROVEMENT 才算进步，并保存这一刻的权重副本。
            if val < seed_best[0] - MIN_IMPROVEMENT:
                seed_best = (val, {k: v.clone() for k, v in head.state_dict().items()}, epoch)
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        if verbose:
            print(f"    seed {seed}: best val_huber {seed_best[0]:.6f} at epoch {seed_best[2]} "
                  f"(ran {epoch + 1} epochs)", flush=True)
        # 选种子：验证损失严格更小才替换（平局时保留先训练的种子）。
        if best is None or seed_best[0] < best[0]:
            best = (seed_best[0], seed_best[1], seed_best[2], seed)
    head = make_head()
    head.load_state_dict(best[1])
    return head.eval(), mean, scale, {"val_huber": best[0], "best_epoch": best[2], "seed": best[3]}


def save_head(path: Path, head, mean: np.ndarray, scale: np.ndarray) -> None:
    """保存一个头：state_dict + 该折的 mean/scale（都是 float32 张量）。

    推理端用 ``torch.load(weights_only=True)`` 读取，它只允许张量和字典这类“纯数据”，所以 mean/scale
    必须存成 torch 张量而不是 numpy 数组。保存后立刻按推理端的方式读回一次做检查。
    文件名就是 fold{k}.pt：torch.save 会把文件名（不含扩展名）写进 zip 内部的目录名。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "mean": torch.as_tensor(mean, dtype=torch.float32),
                "scale": torch.as_tensor(scale, dtype=torch.float32)}, path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if set(saved) != {"state_dict", "mean", "scale"} or tuple(saved["mean"].shape) != (mean.shape[0],):
        raise RuntimeError(f"{path}：按推理端方式读回后，键或 mean 的形状与保存时不一致")


# =============================================================================================
# 折外评估（只用于报告，不影响训练）
# =============================================================================================
def predict(head, mean, scale, x) -> np.ndarray:
    """一个头对 x 的有界位移（µm，z, y, x）。"""
    with torch.no_grad():
        shift = bounded(head, torch.as_tensor((x - mean) / scale, dtype=torch.float32)).numpy()
    return shift.astype(np.float64)


def applied_shift(coords: np.ndarray, shift_um: np.ndarray) -> np.ndarray:
    """推理时真正施加的位移（µm）：位移换成网格单位加到坐标上，截断在 [0, 63] 内，再换回 µm。"""
    grid = coords[:, 1:].astype(np.float64)
    moved = np.clip(grid + shift_um / GRID_UM, 0, GRID_MAX)
    return (moved - grid) * GRID_UM


def centered(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """减去每部影片（groups 相同的行）的平均值。"""
    out = values.astype(np.float64).copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] -= out[m].mean(axis=0)
    return out


def centred_stats(target, shift_um, groups) -> dict:
    """去中心化残差：先减去每部影片的平均残差再算长度。

    为什么要看这个口径：把一部影片的所有检测点整体平移同一个向量，任意两点之间的距离都不变，
    关联和后处理（它们只看相对位置）也就完全不受影响。所以“整体偏移”修得再好也不加分，
    只有“逐细胞不同”的那部分修正才有用。一个只会输出常数位移的头，在这个口径上的改进恰好为 0。
    """
    before = np.linalg.norm(centered(target, groups), axis=1)
    after = np.linalg.norm(centered(target - shift_um, groups), axis=1)
    return {"centred_before_um": float(before.mean()), "centred_after_um": float(after.mean()),
            "centred_rel_change": float(after.mean() / before.mean() - 1),
            "centred_abs_zyx_before": np.abs(centered(target, groups)).mean(axis=0).round(4).tolist(),
            "centred_abs_zyx_after": np.abs(centered(target - shift_um, groups)).mean(axis=0).round(4).tolist()}


def residual_stats(coords, target, shift_um, groups=None) -> dict:
    """|GT − 修正后坐标|：原始值，以及按输出 CSV 的方式取整到原始体素之后的值；可选去中心化口径。"""
    grid = coords[:, 1:].astype(np.float64)
    gt_um = grid * GRID_UM + target
    new_grid = grid + shift_um / GRID_UM
    raw = gt_um - new_grid * GRID_UM
    # 提交文件里的坐标是原始体素的整数：网格坐标 × (1, 4, 4) 后四舍五入。z 方向一个体素就是 1.625 µm，
    # 所以小于半层（0.81 µm）的 z 修正在 CSV 里会被抹掉——但在此之前，关联和后处理已经用上了它。
    voxel = np.maximum(0, np.round(new_grid * DOWNSAMPLE))
    rounded = gt_um - voxel * GT_VOXEL_UM
    before = np.linalg.norm(target, axis=1)
    after = np.linalg.norm(raw, axis=1)
    after_r = np.linalg.norm(rounded, axis=1)
    return {
        "n": int(len(target)),
        "before_um": float(before.mean()),
        "after_um": float(after.mean()),
        "after_rounded_um": float(after_r.mean()),
        "rel_change": float(after.mean() / before.mean() - 1),
        "rel_change_rounded": float(after_r.mean() / before.mean() - 1),
        "frac_improved": float(np.mean(after < before - 1e-9)),
        "frac_worse": float(np.mean(after > before + 1e-9)),
        "before_abs_zyx": np.abs(target).mean(axis=0).round(4).tolist(),
        "after_abs_zyx": np.abs(raw).mean(axis=0).round(4).tolist(),
        "after_rounded_abs_zyx": np.abs(rounded).mean(axis=0).round(4).tolist(),
        "mean_shift_um": float(np.linalg.norm(shift_um, axis=1).mean()),
        **({} if groups is None else centred_stats(target, shift_um, groups)),
    }


# =============================================================================================
# 与参考头逐位对比（可选，用于验证复现）
# =============================================================================================
def compare_with_reference(paths: list[Path], reference_dir: Path) -> bool:
    """把新训练的 fold*.pt 与参考目录中的同名文件逐个比较：文件 sha256、每个张量是否完全相等。"""
    all_equal = True
    print(f"与参考头对比：{reference_dir}")
    for path in paths:
        ref = reference_dir / path.name
        if not ref.is_file():
            print(f"  {path.name}: 参考目录中没有这个文件")
            all_equal = False
            continue
        same_file = sha256(path) == sha256(ref)
        mine = torch.load(path, map_location="cpu", weights_only=True)
        theirs = torch.load(ref, map_location="cpu", weights_only=True)
        tensors = {"mean": (mine["mean"], theirs["mean"]), "scale": (mine["scale"], theirs["scale"])}
        same_keys = list(mine["state_dict"]) == list(theirs["state_dict"])
        for key in mine["state_dict"]:
            if key in theirs["state_dict"]:
                tensors[f"state_dict.{key}"] = (mine["state_dict"][key], theirs["state_dict"][key])
        equal = {name: bool(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b))
                 for name, (a, b) in tensors.items()}
        max_diff = max(float((a.double() - b.double()).abs().max()) if a.shape == b.shape else float("inf")
                       for a, b in tensors.values())
        ok = same_keys and all(equal.values())
        all_equal &= ok and same_file
        print(f"  {path.name}: sha256 {'相同' if same_file else '不同'} | 张量 {'全部相等' if ok else '有差异'} "
              f"| 最大绝对差 {max_diff:.3g}" + ("" if ok else f" | 不相等的张量 {[k for k, v in equal.items() if not v]}"))
    return all_equal


# =============================================================================================
# 主流程
# =============================================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="10 折 × 5 个种子训练坐标头（10 折坐标头）。")
    parser.add_argument("--pairs", type=Path, default=Path("/kaggle/working/coord_pairs/pairs.npz"),
                        help="step2 输出的 pairs.npz（含 movies/movie_idx/coords/features/target；多余的键忽略）")
    parser.add_argument("--out", type=Path, default=Path(f"/kaggle/working/{HEAD_NAME}"),
                        help="输出目录（之后整个上传为 Kaggle 数据集 biohub-cv10-coord-head）")
    parser.add_argument("--folds", type=int, default=10, help="折数；推理管线要求恰好 10 个头")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="每折尝试的随机种子，按验证损失选最好的")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="每个种子最多训练的 epoch 数")
    parser.add_argument("--threads", type=int, default=32,
                        help="torch CPU 线程数；部署的头用 32 线程训练，逐位复现需要相同取值")
    parser.add_argument("--overwrite", action="store_true", help="输出目录已有 fold*.pt 时覆盖")
    parser.add_argument("--reference-dir", type=Path, default=None,
                        help="可选：训练完后与该目录中的同名 fold*.pt 逐位对比")
    parser.add_argument("--kaggle-user", default=None,
                        help="可选：写出 dataset-metadata.json（id = <用户名>/biohub-cv10-coord-head），便于用 kaggle CLI 上传")
    parser.add_argument("--verbose", action="store_true", help="打印每个种子的早停情况")
    args = parser.parse_args(argv)

    # 线程数必须在任何张量计算之前设置。
    torch.set_num_threads(args.threads)
    started = time.time()
    seeds = [int(s) for s in args.seeds.split(",")]

    pairs_path = args.pairs.resolve()
    data = np.load(pairs_path)
    movies = [str(m) for m in data["movies"]]
    movie_of = np.array(movies)[data["movie_idx"]]
    # 输入 = 224 维特征（float32 副本）；目标 = GT − 检测（µm），float32。
    x = data["features"].astype(np.float32)
    y = data["target"].astype(np.float32)
    coords = data["coords"]
    # 背景样本（step2 写出的 bg_*）只用于报告，没有也能训练。
    has_bg = all(key in data.files for key in ("bg_features", "bg_movie_idx", "bg_coords"))
    if has_bg:
        bg_movie_of = np.array(movies)[data["bg_movie_idx"]]
        bg_x = data["bg_features"].astype(np.float32)
        bg_coords = data["bg_coords"]
    embryo = np.array([m.split("_")[0] for m in movie_of])
    if x.shape[1] != 224 or len(x) != len(y) or y.shape[1] != 3:
        raise SystemExit(f"pairs.npz 形状不对：features {x.shape}，target {y.shape}")

    folds = folds_of(movies, args.folds)
    fold_rows = np.array([folds.get(m, -1) for m in movie_of])
    if has_bg:
        bg_fold = np.array([folds.get(m, -1) for m in bg_movie_of])
    out = args.out.resolve()
    if out.exists() and any(out.glob("fold*.pt")) and not args.overwrite:
        raise SystemExit(f"{out} 中已有 fold*.pt；确认要覆盖请加 --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(y):,} 个训练对，{len(movies)} 部影片，{args.folds} 折 × 种子 {seeds}，"
          f"{args.threads} 线程，torch {torch.__version__}，numpy {np.__version__}", flush=True)

    oof = np.zeros((len(y), 3))
    if has_bg:
        bg_oof = np.zeros((len(bg_x), 3))
    infos, head_map, paths = {}, {}, []
    for k in range(args.folds):
        # 训练行与验证行都按原始行顺序排列（flatnonzero 返回升序下标）：randperm 打乱的是这些行的下标，
        # 行顺序变了，打乱后的 batch 组成就变了。
        tr, va = np.flatnonzero((fold_rows >= 0) & (fold_rows != k)), np.flatnonzero(fold_rows == k)
        head, mean, scale, info = train_head(x[tr], y[tr], x[va], y[va], seeds, epochs=args.epochs,
                                             verbose=args.verbose)
        path = out / f"fold{k}.pt"
        save_head(path, head, mean, scale)
        paths.append(path)
        # 折外预测：第 k 折的影片只由没见过它们的第 k 个头来修正，这是“对新影片”的单模型估计。
        oof[va] = applied_shift(coords[va], predict(head, mean, scale, x[va]))
        if has_bg:
            bt = np.flatnonzero(bg_fold == k)
            bg_oof[bt] = applied_shift(bg_coords[bt], predict(head, mean, scale, bg_x[bt]))
        held = sorted(m for m, f in folds.items() if f == k)
        for m in held:
            head_map[m] = path.name
        info.update({"n_train_rows": int(len(tr)), "n_val_rows": int(len(va)), "val_movies": held})
        infos[f"fold{k}"] = info
        print(f"fold {k}: val_huber {info['val_huber']:.5f} best_epoch {info['best_epoch']} seed {info['seed']} "
              f"({len(held)} movies, {len(va):,} val rows)", flush=True)

    (out / "ensemble.json").write_text(json.dumps([p.name for p in paths], indent=1) + "\n")
    (out / "head_map_oof.json").write_text(json.dumps(head_map, indent=1, sort_keys=True) + "\n")

    # 折外指标：每部影片由把它留作验证的那一个头打分。部署时用的是 10 个头的平均，它无法在训练影片上
    # 评估（每部影片都被其中 9 个头见过）；另外第 k 折同时用于早停和选种子，这个数字会略偏乐观。
    metrics = {"folds": infos, "oof": {}}
    for label, mask in [("all", np.ones(len(y), bool))] + [(e, embryo == e) for e in sorted(set(embryo))]:
        metrics["oof"][label] = residual_stats(coords[mask], y[mask], oof[mask], movie_of[mask])
    if has_bg:
        # 背景样本（任意检测点）上的平均位移：头会移动每一个检测点，而 GT 配对只占其中很小一部分，
        # 所以要看它在一般检测点上的行为是否与配对样本一致（没有异常大的系统性平移）。
        metrics["bg_shift"] = {"mean_um": float(np.linalg.norm(bg_oof, axis=1).mean()),
                               "mean_zyx": np.round(bg_oof.mean(axis=0), 4).tolist()}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1) + "\n")

    module_path = HERE / "coord_head_module.py"
    meta = {
        "name": HEAD_NAME, "arch": "mlp 224-32-3 (SiLU), zero-initialised last layer", "inputs": "feat (224)",
        "zero_z": False,
        "recipe": ("Huber(delta=1 um) on bounded(head(x)) vs (GT - detection) um; AdamW lr 1e-3 wd 1e-4; "
                   "batch 1024; <= 30 epochs, patience 5 (improvement > 1e-6); per-fold float32 mean/std "
                   "(ddof 0, floor 1e-3) from the training rows; K-fold by movie, stratified by embryo "
                   "(sha1 order, round robin); fold k is the only validation of model k (early stopping and "
                   "seed choice); deploy the mean of the K bounded shifts"),
        "pairs_sha256": sha256(pairs_path), "module_sha256": sha256(module_path),
        "folds": args.folds, "seeds": seeds, "epochs": args.epochs, "threads": args.threads,
        "n_pairs": int(len(y)), "n_movies": len(movies),
        "fold_sha256": [sha256(p) for p in paths],
        "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "seconds": round(time.time() - started, 1),
    }
    (out / "head_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    if args.kaggle_user:
        (out / "dataset-metadata.json").write_text(json.dumps({
            "title": HEAD_NAME, "id": f"{args.kaggle_user}/{HEAD_NAME}",
            "licenses": [{"name": "CC0-1.0"}]}, indent=1) + "\n")

    a = metrics["oof"]
    per_embryo = ", ".join(f"{e} {a[e]['rel_change']:+.2%}/c{a[e]['centred_rel_change']:+.2%}"
                           for e in sorted(set(embryo)))
    print(f"{HEAD_NAME}: single-model OOF |r| {a['all']['before_um']:.4f} -> {a['all']['after_um']:.4f} "
          f"({a['all']['rel_change']:+.2%}; CENTRED {a['all']['centred_rel_change']:+.2%}), "
          f"{per_embryo} | {meta['seconds']} s")
    if has_bg:
        print(f"背景样本平均位移 {metrics['bg_shift']['mean_um']:.3f} µm，平均 (z, y, x) = {metrics['bg_shift']['mean_zyx']} µm")
    for p, digest in zip(paths, meta["fold_sha256"]):
        print(f"  {p.name}  sha256 {digest}")
    if args.reference_dir is not None:
        if not compare_with_reference(paths, args.reference_dir.resolve()):
            print("与参考头不完全一致（检查 pairs.npz 行顺序、线程数、torch / numpy 版本）")
            return 1
        print("与参考头逐位一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
