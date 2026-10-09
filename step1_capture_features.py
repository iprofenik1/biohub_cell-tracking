"""第 1 部分 · 第 1 步（Kaggle GPU）：在训练影片上以 capture 模式运行推理管线，导出每帧检测点的 224 维特征。

用法（Kaggle notebook，加速器选 GPU T4 ×2；先把本仓库放到 /kaggle/working 下并 cd 到仓库根目录）::

    !python part1_cv10_coord_head/step1_capture_features.py --shard 0 --num-shards 4
    !python part1_cv10_coord_head/step1_capture_features.py --shard 0 --num-shards 4 --dry-run   # 只检查补丁

每个 Kaggle 会话跑一个分片（--shard 0 … 3），各自把 ``/kaggle/working/coord_capture`` 保存为 notebook 输出，
step2 再把所有分片的输出一起读进来。

为什么直接运行第三部分的推理管线，而不是另写一个“提特征”的小脚本
==============================================================
坐标头的输入是检测器 U-Net 在检测点周围的特征。推理时这份特征是这样来的：主模型 8 个视角 TTA 的
特征图取平均（其中一个视角与 x 翻转重复，这是公开方案的原样实现），检测峰来自主、副两个模型融合后的
热图（概率 > 0.965 且是 3×3×3 格点邻域内的最大值；参数 pool_kernel_um = 3.0 换算成每轴 3 个格点），
每帧只在它第一次出现的长度为 2 的滑窗里检测、取特征。任何一处不同
（少一个 TTA 视角、换一种检测方式、单帧编码……）都会让训练特征与推理特征的分布错开，头学到的修正
在推理时就不准了。最可靠的办法是**一字不差地运行推理管线本身**，只做三处补丁：

1. ``TEST_DIR`` 指向一个只含本分片训练影片（符号链接）的目录——管线把它们当作“测试影片”来推理；
2. 跳过“必须恰好挂载 10 个坐标头”的检查——capture 模式根本不读头文件（这时我们还没有头）；
3. 坐标头模式从 ``candidate``（10 个头取平均）改成 ``capture``：推理模块在每帧检测完成后，把
   ``(检测坐标, 224 维特征)`` 写到 ``<out>/<影片>/<帧号:04d>.npz``，坐标原样返回、不做修正。

只跑到第 4 段为止
=================
推理管线共 13 段（第 0–12 段）。检测、取特征、关联和 ILP 都在第 4 段末尾启动的 GPU 子进程里完成；
第 4 段的最后一行打印 ``Prediction completed ...`` 时，capture 文件已经全部写完。第 5 段起是后处理与
写提交文件，对训练数据没有用处，而且第 6 段会核对 submission.csv 等产物，在训练影片上没有意义。
所以本脚本用 ``section_slice(source, 0, 4)`` 只取第 0–4 段执行：后面的段根本不会被编译执行，
既省时间，也不会产生一份“训练影片的 submission.csv”（那绝不能提交）。

产物
====
- ``<out>/<影片>.npz``：键 ``coords``——(N, 4) int16，检测点的网格坐标 [t, z, y, x]（网格格距
  1.625 µm，乘 1.625 得 µm）；``features``——(N, 224) float32，坐标头的输入（7 个采样位置 × 32 通道）。
  推理模块先按帧写 ``<out>/<影片>/<帧号:04d>.npz``（没有检测点的帧不写文件），推理结束后本脚本把每部
  影片的逐帧文件按帧号升序合并成一个文件并删除逐帧目录：Kaggle 对 notebook 输出有约 500 个文件的上限，
  逐帧文件（每个分片约 5,000 个）会被截断。加 ``--keep-frames`` 可保留逐帧布局；step2 两种布局都能读。
- ``<out>/capture_manifest_shard{i}of{n}.json``：本分片每部影片的帧数、检测点数，以及整个分片的耗时。

算力与耗时
==========
实测（最终管线在 Kaggle 双 T4 上跑 4 部影片的日志）：每部影片在一张 T4 上检测 + 取特征 + 关联约
4–4.5 分钟，ILP 再加 5–75 秒；两张卡各处理一部影片、并行，平均每部约 2.5 分钟。199 部约 8–9 小时，
加上第 3 段的安装，离单个 Kaggle 会话 12 小时的上限太近，所以分 4 个分片（每片约 50 部，约 2–2.5 小时）。
全部特征约 4–5 GB（每个检测点 224 个 float32，约 500 万个检测点），每个分片约 1 GB。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# 让 ``from common import pipeline_runner`` 在“从仓库根目录运行本脚本”时可用。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import pipeline_runner as pr  # noqa: E402

# 抓取结果的默认输出目录。必须在 /kaggle/working/tracking_repo 之外：推理管线第 3 段会删除并重建那个目录。
DEFAULT_CAPTURE_DIR = pr.WORKING_DIR / "coord_capture"
# 只执行到第 4 段（GPU 推理结束）。
LAST_SECTION = 4
# 推理管线在 /kaggle/working 下生成、只对推理有用的大目录（--cleanup 时删除，只留下抓取结果）。
PIPELINE_SCRATCH_DIRS = ("tracking_repo", "edge_cache", "x138_candidate_prob_cache", "secondary_seed_weights")


# =============================================================================================
# 选影片
# =============================================================================================
def select_movies(train_dir: Path, movies_arg: str | None, shard: int, num_shards: int) -> list[str]:
    """本分片要处理的影片。

    ``movies_arg``：逗号分隔的影片名，或 ``@文件``（每行一个）；不给时取训练集全部影片（同时有 .zarr 与
    .geff，按名字排序）。然后按 ``movies[shard::num_shards]`` 交错切分：排序后同一胚胎连在一起，
    交错切分让每个分片都含两个胚胎、耗时也更均匀。
    """
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


# =============================================================================================
# 打补丁
# =============================================================================================
def capture_patches(subset_dir: str | Path, capture_dir: str | Path) -> list[tuple[str, str, str]]:
    """capture 所需的三处补丁，每处都锚定推理脚本中恰好出现一次的整行原文（见 common/pipeline_runner.py）。"""
    capture_dir = os.path.abspath(os.fspath(capture_dir))
    return [
        # 1) 第 2 段：TEST_DIR = COMP_DIR / "test"  →  TEST_DIR = Path('<训练影片子集目录>')
        pr.redirect_test_dir_patch(subset_dir),
        # 2) 第 4 段：“坐标头文件必须恰好 10 个”的检查 → if False:（下一行的 raise 永远不会执行）。
        #    capture 模式不读头文件，此时 V1284_HEAD 为空字符串也没关系。
        pr.line_patch("坐标头：capture 模式不需要头文件，跳过 10 个头的挂载检查",
                      pr.ANCHORS["coord_head_count_guard"], "if False:"),
        # 3) 第 4 段：坐标头模式 candidate → capture，并给出输出目录。环境变量在第 4 段末尾启动推理子进程
        #    之前设置，子进程继承 os.environ，推理模块（v1284_coordinate_refinement.py）在子进程里读取它们。
        pr.line_patch("坐标头：切换到 capture 模式",
                      pr.ANCHORS["coord_head_mode"],
                      "os.environ['V1284_MODE']='capture'\n"
                      f"os.environ['V1284_CAPTURE']={capture_dir!r}"),
    ]


def build_capture_source(subset_dir: str | Path, capture_dir: str | Path,
                         pipeline_path: str | Path | None = None) -> str:
    """读取第三部分的推理脚本，打上 capture 补丁，只保留第 0–4 段，返回可直接执行的源码。

    先在**完整**源码上打补丁（锚点唯一性在全文范围内检查，更严格），再切片。
    """
    source = pr.load_pipeline_source(pipeline_path)
    source = pr.apply_patches(source, capture_patches(subset_dir, capture_dir))
    sliced = pr.section_slice(source, 0, LAST_SECTION)
    # 切片必须以“GPU 推理结束”那一行收尾：说明第 4 段完整，且第 5 段及以后的后处理都不在其中。
    if pr.check_anchors(sliced, {"done": pr.ANCHORS["prediction_done"]})["done"] != 1:
        raise RuntimeError("第 0–4 段的切片中找不到“Prediction completed”那一行，推理脚本的分段可能被改动了")
    return sliced


# =============================================================================================
# 检查产物
# =============================================================================================
def summarize_capture(capture_dir: Path, movies: list[str]) -> dict:
    """统计每部影片抓到的帧数与检测点数；缺失的影片单独列出。"""
    import numpy as np

    per_movie, missing = {}, []
    for movie in movies:
        folder = capture_dir / movie
        frames = sorted(int(p.stem) for p in folder.glob("*.npz") if p.stem.isdigit()) if folder.is_dir() else []
        if not frames:
            missing.append(movie)
            continue
        detections = 0
        for t in frames:
            with np.load(folder / f"{t:04d}.npz") as data:
                coords = data["coords"]
                if data["features"].shape != (len(coords), 224):
                    raise RuntimeError(f"{folder / f'{t:04d}.npz'}：features 形状 {data['features'].shape} 不对")
                detections += len(coords)
        per_movie[movie] = {"frames": len(frames), "first_frame": frames[0], "last_frame": frames[-1],
                            "detections": detections}
    return {"movies": per_movie, "missing": missing}


def pack_movie(capture_dir: Path, movie: str) -> int:
    """把一部影片的逐帧文件 <影片>/<帧号>.npz 合并成一个 <影片>.npz，校验无误后删除逐帧目录。

    为什么要合并：每部影片约 100 个帧文件，一个分片约 5,000 个文件；Kaggle 对 notebook 输出
    （下载、由输出生成数据集）有 500 个文件左右的上限，逐帧文件很容易被截断。合并后每个分片只有约 50 个文件。
    合并顺序 = 帧号升序，与 step2 读取逐帧目录时的拼接顺序完全相同，所以训练对逐位不变。
    """
    import numpy as np

    folder = capture_dir / movie
    frames = sorted((int(p.stem), p) for p in folder.iterdir() if p.suffix == ".npz" and p.stem.isdigit())
    coords_parts, feature_parts = [], []
    for _, frame_path in frames:
        with np.load(frame_path) as data:
            coords_parts.append(data["coords"])
            feature_parts.append(data["features"])
    coords = np.concatenate(coords_parts)
    features = np.concatenate(feature_parts)
    # 先写临时文件、读回比对，再改名并删除逐帧目录：中途失败不会丢数据，也不会留下残缺的合并文件。
    temporary = capture_dir / f".{movie}.tmp.npz"
    np.savez_compressed(temporary, coords=coords, features=features)
    with np.load(temporary) as check:
        if not (np.array_equal(check["coords"], coords) and np.array_equal(check["features"], features)):
            raise RuntimeError(f"{movie}：合并文件读回后与逐帧数据不一致")
    temporary.replace(capture_dir / f"{movie}.npz")
    shutil.rmtree(folder)
    return len(coords)


def cleanup_pipeline_outputs(capture_dir: Path, subset_dir: Path) -> None:
    """删除推理管线在 /kaggle/working 下生成的大目录，只保留抓取结果（便于把输出保存为数据集）。

    只删除固定名单里的目录，并且绝不删除抓取目录本身或它的上级目录。子集目录里只有符号链接，
    删除它不会影响只读的训练数据。
    """
    protected = capture_dir.resolve()
    targets = [pr.WORKING_DIR / name for name in PIPELINE_SCRATCH_DIRS] + [subset_dir]
    for target in targets:
        target = Path(os.path.abspath(target))
        if not target.exists():
            continue
        if protected == target or target in protected.parents:
            print(f"跳过（包含抓取结果）：{target}")
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
    parser = argparse.ArgumentParser(description="在训练影片上以 capture 模式运行推理管线（第 0–4 段），导出坐标头训练特征。")
    parser.add_argument("--train-dir", type=Path, default=None,
                        help="竞赛训练集目录；默认 Kaggle 竞赛数据的 train/")
    parser.add_argument("--out", type=Path, default=DEFAULT_CAPTURE_DIR,
                        help="抓取结果目录（每部影片一个 <out>/<影片>.npz），默认 /kaggle/working/coord_capture")
    parser.add_argument("--shard", type=int, default=0, help="本次运行的分片编号（从 0 开始）")
    parser.add_argument("--num-shards", type=int, default=1, help="分片总数（199 部影片建议 4）")
    parser.add_argument("--movies", default=None,
                        help="可选：显式的影片列表，逗号分隔，或 @文件（每行一个）；之后仍按分片切分")
    parser.add_argument("--subset-dir", type=Path, default=pr.DEFAULT_SUBSET_DIR,
                        help="训练影片子集（符号链接）目录，默认 /kaggle/working/train_subset")
    parser.add_argument("--pipeline", type=Path, default=None,
                        help="推理脚本路径（.py 或 .ipynb）；默认 part3_inference_pipeline/biohub_final_inference.py")
    parser.add_argument("--dry-run", action="store_true", help="只建子集、打补丁、编译检查，不运行推理")
    parser.add_argument("--keep-frames", action="store_true",
                        help="保留逐帧文件 <影片>/<帧号>.npz，不合并成 <影片>.npz（默认合并，以免超过 Kaggle 输出的文件数上限）")
    parser.add_argument("--cleanup", action="store_true",
                        help="完成后删除推理管线的中间目录（tracking_repo、edge_cache 等），只留抓取结果")
    args = parser.parse_args(argv)

    train_dir = (args.train_dir or (pr.competition_dir() / "train")).resolve()
    capture_dir = Path(os.path.abspath(args.out))
    if "tracking_repo" in capture_dir.parts:
        raise SystemExit("抓取目录不能放在 tracking_repo 里面（推理管线第 3 段会删除它）")
    movies = select_movies(train_dir, args.movies, args.shard, args.num_shards)
    print(f"分片 {args.shard}/{args.num_shards}：{len(movies)} 部影片，例如 {movies[:3]}", flush=True)

    subset_dir = pr.make_train_subset(movies, train_dir, args.subset_dir)
    source = build_capture_source(subset_dir, capture_dir, args.pipeline)
    if args.dry_run:
        compile(source, "biohub_capture_features.py", "exec", dont_inherit=True)
        for line in source.splitlines():
            if line.startswith(("TEST_DIR = ", "os.environ['V1284_MODE']", "os.environ['V1284_CAPTURE']", "if False:")):
                print("补丁后：", line)
        print(f"dry-run 通过：第 0–{LAST_SECTION} 段共 {len(source.splitlines())} 行，可以编译；子集目录 {subset_dir}")
        return 0

    capture_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    # 像 notebook 一样在全新命名空间里执行第 0–4 段。GPU 推理在第 4 段末尾以子进程方式运行
    # （两张 GPU 时按影片交错分成两个子进程），capture 文件由子进程里的推理模块写出。
    try:
        pr.run_source(source, "biohub_capture_features.py")
    finally:
        # 无论成功与否都删除子集目录里的符号链接（见 common/pipeline_runner.remove_train_subset）。
        pr.remove_train_subset(subset_dir)
    seconds = time.time() - started

    summary = summarize_capture(capture_dir, movies)
    if args.keep_frames:
        layout = "<影片>/<帧号:04d>.npz：coords (N,4) int16 网格 [t,z,y,x]；features (N,224) float32"
    else:
        # 每部影片合并成一个文件（帧号升序），避免 Kaggle 输出的文件数上限；step2 两种布局都能读。
        for movie in summary["movies"]:
            packed = pack_movie(capture_dir, movie)
            if packed != summary["movies"][movie]["detections"]:
                raise RuntimeError(f"{movie}：合并后的检测点数 {packed} 与逐帧统计不一致")
        layout = "<影片>.npz：coords (N,4) int16 网格 [t,z,y,x]；features (N,224) float32（帧号升序拼接）"
    summary.update({"shard": args.shard, "num_shards": args.num_shards, "seconds": round(seconds, 1),
                    "layout": layout})
    manifest = capture_dir / f"capture_manifest_shard{args.shard}of{args.num_shards}.json"
    manifest.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    total = sum(v["detections"] for v in summary["movies"].values())
    print(f"抓取完成：{len(summary['movies'])} 部影片，{total:,} 个检测点，用时 {seconds / 60:.1f} 分钟；清单 {manifest}")
    if summary["missing"]:
        raise SystemExit(f"{len(summary['missing'])} 部影片没有抓到任何特征：{summary['missing'][:5]}")
    if args.cleanup:
        cleanup_pipeline_outputs(capture_dir, subset_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
