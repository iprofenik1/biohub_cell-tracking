"""在训练影片上复用第三部分的推理管线（两个 step1 导出脚本共用）。

为什么要这样做
==============
10 折坐标头（第 1 部分）和分裂补全打分器（第 2 部分）的训练数据，都必须来自**和最终推理完全相同**
的那条管线：同样的检测器权重、同样的 8 视角 TTA 与双模型融合、同样的关联模型、同样的 ILP 与后处理。
只要训练时看到的输入分布和推理时不一样（哪怕只是少了一个 TTA 视角），学到的修正量就会偏。
最稳妥的办法不是“另写一份相似的代码”，而是**直接执行第三部分的推理脚本本身**，只在几个精确的位置
做最小的文本替换（打补丁）：

1. 把“测试影片目录” ``TEST_DIR`` 指向一个只含训练影片的目录（训练影片带 GT，测试影片没有）；
2. 按需要切换坐标头的工作模式、跳过与训练数据无关的步骤；
3. 只执行需要的段（例如只跑到第 4 段的 GPU 推理为止）。

本模块只用标准库，可以在任何环境里 import；真正执行管线（``run_source``）必须在 Kaggle GPU
notebook 中，并挂载与推理 notebook 相同的数据集（支持包、副模型权重、DeepCenter、竞赛数据），
因为管线里的 ``/kaggle/...`` 路径保持原样。学生用法示例::

    !python part1_cv10_coord_head/step1_capture_features.py --shard 0 --num-shards 4

公开接口（名字与语义固定，两个 step1 脚本和验证脚本都依赖它们）
------------------------------------------------------------------
- ``load_pipeline_source(path=None) -> str``
      读取推理脚本全文；默认 ``<仓库>/part3_inference_pipeline/biohub_final_inference.py``
      （也接受 .ipynb，按顺序拼接全部代码单元）。
- ``apply_patches(source, patches) -> str``
      ``patches`` 是 ``(标签, 旧文本, 新文本)`` 的列表，按顺序应用；每个“旧文本”在当前源码中必须
      **恰好出现 1 次**，否则抛出 ``ValueError``，错误信息里带上标签。
- ``section_slice(source, first, last) -> str``
      取第 ``first`` 段到第 ``last`` 段（含）的代码；``last=None`` 表示一直到文件末尾。
      第 0 段标记之前的内容（文件头注释和 ``from __future__ import annotations``）总是保留在最前面。
- ``list_train_movies(train_dir) -> list[str]``    同时有 ``.zarr`` 与 ``.geff`` 的影片名（排序）。
- ``shard_movies(movies, shard, num_shards) -> list[str]``    ``movies[shard::num_shards]``。
- ``make_train_subset(movies, train_dir, dst_dir) -> Path``
      在 ``dst_dir`` 下为每部影片建符号链接 ``<影片>.zarr -> <train_dir>/<影片>.zarr``，目录布局与
      竞赛的 ``test/`` 完全一样（只有 .zarr，没有 .geff）。
- ``redirect_test_dir_patch(subset_dir) -> (标签, 旧文本, 新文本)``
      把第 2 段的 ``TEST_DIR = COMP_DIR / "test"`` 换成 ``TEST_DIR = Path('<subset_dir>')``。
- ``run_source(source, filename) -> dict``
      像 notebook 单元一样在全新命名空间 ``{'__name__': '__main__'}`` 中执行，返回该命名空间。

附加工具（非必需，但推荐使用）：``line_patch``（整行锚点补丁）、``ANCHORS``（下面列出的锚点行原文）、
``check_anchors``（检查每个锚点在源码中是否恰好出现一次）、``section_markers``、``competition_dir``。

段标记
======
教学版推理脚本用 ``# ===== 第 N 段：<中文标题> =====``（N = 0..12，全角冒号）分段；
原始 notebook 导出的单文件用 ``# ===== ORIGINAL X138 CELL N =====``（这是公开基线 notebook 沿用下来的
分段写法）。``section_slice`` 两种都认，所以同一套代码既能在教学版上用，也能在原始文件上做对照测试。

为什么补丁要锚定“整行原文”
==========================
推理脚本本身就大量使用“锚点替换”给推理子进程的脚本打补丁（第 3、4 段），它的约定是：锚点必须恰好
出现一次，否则立刻报错——补丁静默失效是最难查的 bug（例如低分峰 dump 没写出来，后处理的补缺就会
全程空转，而分数只是悄悄变差）。这里沿用同样的约定，并且更进一步：推荐用 ``line_patch`` 生成
“前后都带换行符”的整行锚点。这样：

- 注释里即使引用了同一段代码（教学版有大量中文注释），也不会被误匹配（注释行以 ``#`` 开头）；
- 缩进不同的同名语句不会被误匹配（例如 ``_D1_ACTIVE = False`` 在原文中作为子串出现 4 次，
  但顶格的整行只有 1 次）；
- 前缀相同的更长语句不会被误匹配。

注意：教学版只允许改注释，代码行逐字保留；因此下面的锚点在原始脚本和教学版中都应恰好出现一次。
修改推理脚本后，可以运行 ``check_anchors(load_pipeline_source())`` 确认全部锚点仍然各出现一次。

推荐锚点（``ANCHORS``，均为整行原文；“原第 N 行”指原始单文件中的行号，仅供对照）
==================================================================================
- ``test_dir``（第 2 段，原第 238 行）::

      TEST_DIR = COMP_DIR / "test"

- ``test_stems``（第 4 段，原第 1488 行）::

      test_stems = list_test_stems()

- ``coord_head_count_guard``（第 4 段，原第 1681 行）::

      if len(_myhead) != 10:

- ``coord_head_mode``（第 4 段，原第 1687 行）::

      os.environ['V1284_MODE']='candidate'

- ``coord_head_paths``（第 4 段，原第 1688 行）::

      os.environ['V1284_HEAD']=os.pathsep.join(str(p) for p in _myhead)

- ``prediction_done``（第 4 段最后一行，原第 1940 行）::

      print(f"Prediction completed in {predict_seconds / 60:.2f} minutes")

- ``d1_switch_default``（第 5 段，原第 4006 行；顶格的那一行，缩进的同名语句不算）::

      _D1_ACTIVE = False

- ``deepcenter_load``（第 5 段，原第 4654 行）::

      DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()

- ``first_pass_call``（第 5 段最后一句，原第 4806 行）::

      write_test_submission("base")

- ``final_pass_pp_apply``（第 11 段，原第 5843 行；前面 5 行把 5 个模块开关置 True）::

      _dr_saved_pp = pp_apply(selected_config)

``V1284_*`` 是坐标修正头（第 1 部分的 10 折坐标头所用的推理模块）的环境变量开关：
``V1284_MODE`` 取 ``zero``（不修正）/ ``capture``（只导出特征、不改坐标）/ 其它值（本管线为
``candidate``：加载 ``V1284_HEAD`` 列出的全部头，取有界位移的平均）；``V1284_CAPTURE`` 是 capture
模式的输出目录。

**不要**用作锚点的行（教学版里可能被合法地改写）：

- 坐标头数据集的 glob 行：教学版是
  ``_myhead = sorted(Path('/kaggle/input').rglob('biohub-cv10-coord-head/fold*.pt'))``，
  与原始文件不同（数据集名换成了学生自己上传的 ``biohub-cv10-coord-head``）；
- 紧随其后的 ``raise RuntimeError((...))`` 行：错误信息字符串允许翻译；
- 第 11 段最终 pass 中 ``write_test_submission(<标签>)`` 那一行：标签字符串允许中性化；
- 带行尾注释的配置行（第 0 段的 ``os.environ[...] = ...  # ...``）：行尾英文注释可能被翻译或删除；
- 任何补丁字符串内部的文本（它们是写给推理子进程的代码，后续补丁以它们为锚点）。

第 1 部分（10 折坐标头，``step1_capture_features.py``）推荐做法
--------------------------------------------------------------
只执行第 0–4 段（检测 + 关联 + ILP 在 GPU 子进程里完成；不需要后处理）::

    src = load_pipeline_source()
    src = apply_patches(src, [
        redirect_test_dir_patch(subset_dir),
        # 还没有训练好的头：capture 模式根本不读头文件，所以跳过“必须恰好 10 个头”的检查
        line_patch("坐标头：跳过 10 个头的挂载检查", ANCHORS["coord_head_count_guard"], "if False:"),
        # capture：每帧把 (检测坐标, 224 维特征) 存盘，坐标不做任何修正
        line_patch("坐标头：capture 模式", ANCHORS["coord_head_mode"],
                   "os.environ['V1284_MODE']='capture'\\n"
                   "os.environ['V1284_CAPTURE']='/kaggle/working/coord_capture'"),
    ])
    run_source(section_slice(src, 0, 4), "biohub_capture_features.py")

产物：``/kaggle/working/coord_capture/<影片>/<t:04d>.npz``（键 ``coords``：下采样网格上的整数
``[t, z, y, x]``，乘 1.625 µm 得物理坐标；``features``：N×224 float32）。环境变量在第 4 段末尾
启动推理子进程之前设置，子进程继承 ``os.environ``，所以补丁生效。

第 2 部分（分裂补全打分器，``step1_build_table.py``）推荐做法
--------------------------------------------------------------
执行第 0–5 段：第 0–4 段产生原始 ILP 图与两种缓存；第 5 段定义全部后处理函数并加载 DeepCenter
（``ANCHORS["deepcenter_load"]`` 那一行要保留）。第 5 段最后一句 ``write_test_submission("base")``
会立刻对全部影片跑一遍“第一遍”后处理并写 submission.csv，制作训练表时不需要，替换掉::

    src = apply_patches(load_pipeline_source(), [
        redirect_test_dir_patch(subset_dir),
        line_patch("跳过第一遍提交", ANCHORS["first_pass_call"], "pass"),
    ])
    ns = run_source(section_slice(src, 0, 5), "biohub_build_table.py")

第 4 段默认要求挂载恰好 10 个坐标头文件（``biohub-cv10-coord-head`` 数据集）。还没有自己的头时，
可以像第 1 部分那样跳过数量检查、并把 ``coord_head_mode`` 那一行换成 ``os.environ['V1284_MODE']='zero'``
（不修正坐标）——能跑通，但训练表的输入分布会与最终管线（10 折坐标头修正后的坐标）略有差异。
另外，这些训练影片本身参与过检测器与坐标头的训练，在它们上面得到的图比测试集上的更“干净”，
属于样本内数据，这一点在解读训练表时要记住。

之后由导出脚本自己在 ``ns`` 里按第 11 段最终 pass 的方式（``ANCHORS["final_pass_pp_apply"]`` 前面
的 5 行）打开 ``_DR_ACTIVE / _G1X1_ACTIVE / _D4_ACTIVE / _RR_ACTIVE / _D1_ACTIVE``（分别是：候选概率
重链接、全局位移估计 + 跳变感知平滑、关联特征修正、找回概率打分、分裂补全打分器），设置
``ns["D1_DUMP"] = []``，逐影片读入原始图后**直接调用** ``ns["filter_output_graph"](...)``。
不要用 ``write_test_submission``：它要求原始图数量等于影片数、会在超过截止时间后静默关闭修复阶段、
并把任何异常吞掉改写成兜底图——这三点都会悄悄污染训练表。

推理管线中与“换成训练影片”有关的事实
=====================================
1. ``TEST_DIR``（第 2 段）没有环境变量开关，只能改这一行。它有两个使用者，一个补丁同时覆盖：
   第 4 段 ``list_test_stems()`` 列出影片并写进 splits JSON、推理子进程命令 ``--data-dir str(TEST_DIR)``；
   第 5 段 ``read_test_frame()`` 在**调用时**读全局 ``TEST_DIR``（DeepCenter 热图、合成中点精修、
   分裂补全打分器的亮度特征都经它读原始帧）。
2. 原始 ILP 图、低分峰 dump、候选概率缓存、坐标头 capture 都由**同一次**推理子进程运行产生
   （第 4 段末尾 ``subprocess`` 启动；≥2 张 GPU 时按 ``--slice k::2`` 分两片并行，再由
   ``_merge_prediction_shards`` 合并）：

   - 原始图：``/kaggle/working/tracking_repo/predictions/<系统用户名>/unet_transformer/split_0/<影片>.geff``
     （``<系统用户名>`` 是支持包代码取的 ``$USER``，没有时为 ``unknown``，与 Kaggle 账号无关）
     （节点 t,z,y,x 为原分辨率体素坐标；边属性 ``edge_prob``）；
   - 低分峰 dump：``$BIOHUB_CACHE_DIR/<影片>.npz``，第 0 段设为 ``/kaggle/working/edge_cache``
     （找回与低分峰补缺读取它；缺失时这两步静默空转）；
   - 候选概率缓存：``/kaggle/working/x138_candidate_prob_cache/<影片>.probabilities.npz``
     （路径写死在第 4 段的补丁文本和第 5 段 ``filter_output_graph`` 里；打开候选概率重链接时缺失会报错）。

   之后的各遍后处理（第 5 段末的第一遍、第 11 段的最终 pass）只读取这些文件，不再用 GPU 推理。
3. 第 3 段每次都会删除并重建 ``/kaggle/working/tracking_repo``，所以影片子集目录必须放在它外面
   （默认 ``/kaggle/working/train_subset``）。
4. 第 6 段审计 ``/kaggle/working/retention_guard_*.jsonl``（第 4 段补丁逐帧写出的“双模型融合保留率
   守卫”日志）和写死路径的 ``/kaggle/working/submission.csv``：要求覆盖 ``TEST_DIR`` 下的全部影片。
   导出训练数据时应只执行到第 4 段或第 5 段，不要执行第 6 段及以后；那里的 submission.csv 即使存在，
   也是训练影片的结果，绝不能提交。
5. 第 3 段会做离线 pip 安装、复制支持包、校验哈希，副作用很大：每个 Python 进程只跑一个分片。
6. 每部影片的 GPU 推理在一张 T4 上约 4–6 分钟（检测 + 取特征 + 关联约 4–4.5 分钟，ILP 再加 5–75 秒，
   随影片密度变化；最终管线在 Kaggle 双 T4 上跑 4 部影片的实测）。两张卡并行时平均每部约 2.5 分钟，199 部训练影片约 8–9 小时，
   离一次 Kaggle 会话 12 小时的上限太近，所以用 ``--shard i --num-shards n`` 分多次会话运行，
   每个分片做成一个独立的 notebook，它的输出（notebook output）直接挂载给下一步。第 1 部分的抓取结果
   每片约 5,000 个文件，不要再把它另存为数据集（Kaggle 由 notebook 输出建数据集时有文件数上限，见第 1 部分 README）。
"""
from __future__ import annotations

import json
import linecache
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "ANCHORS",
    "COMPETITION",
    "COMPETITION_DIR_CANDIDATES",
    "DEFAULT_PIPELINE_PATH",
    "DEFAULT_SUBSET_DIR",
    "REPO_ROOT",
    "WORKING_DIR",
    "apply_patches",
    "check_anchors",
    "competition_dir",
    "line_patch",
    "list_train_movies",
    "load_pipeline_source",
    "make_train_subset",
    "redirect_test_dir_patch",
    "run_source",
    "section_markers",
    "section_slice",
    "shard_movies",
]

# ---------------------------------------------------------------------------------------------
# 路径常量。全部保持 Kaggle 的原始路径（推理脚本里写死的也是这些路径）；验证脚本若要在别处运行，
# 应显式传入目录参数，而不是依赖这里的默认值。
# ---------------------------------------------------------------------------------------------
# 本文件位于 <仓库>/common/pipeline_runner.py，仓库根目录 = 上两级。
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE_PATH = REPO_ROOT / "part3_inference_pipeline" / "biohub_final_inference.py"

COMPETITION = "biohub-cell-tracking-during-development"
# 与推理脚本第 2 段相同的两种挂载位置（Kaggle 新旧两种竞赛数据挂载方式），取第一个存在的。
COMPETITION_DIR_CANDIDATES = (
    Path(f"/kaggle/input/competitions/{COMPETITION}"),
    Path(f"/kaggle/input/{COMPETITION}"),
)
WORKING_DIR = Path("/kaggle/working")
# 影片子集目录：必须在 /kaggle/working/tracking_repo 之外（第 3 段会删除并重建那个目录）。
DEFAULT_SUBSET_DIR = WORKING_DIR / "train_subset"

# ---------------------------------------------------------------------------------------------
# 锚点：推理脚本中会被导出脚本替换的整行原文（详见模块文档字符串中的表格）。
# 这些行在教学版中逐字保留；若有人改动了它们，check_anchors 会立刻发现。
# ---------------------------------------------------------------------------------------------
ANCHORS: dict[str, str] = {
    # 第 2 段：测试影片目录（没有环境变量开关，只能改这一行）
    "test_dir": 'TEST_DIR = COMP_DIR / "test"',
    # 第 4 段：列出 TEST_DIR 下全部 *.zarr，写入推理子进程的 splits JSON
    "test_stems": "test_stems = list_test_stems()",
    # 第 4 段：坐标头文件必须恰好 10 个（10 折），capture 模式下要跳过
    "coord_head_count_guard": "if len(_myhead) != 10:",
    # 第 4 段：坐标修正头的工作模式（推理时为 candidate = 10 个头取平均）
    "coord_head_mode": "os.environ['V1284_MODE']='candidate'",
    # 第 4 段：把找到的头文件路径传给推理子进程
    "coord_head_paths": "os.environ['V1284_HEAD']=os.pathsep.join(str(p) for p in _myhead)",
    # 第 4 段最后一行：GPU 推理（检测 + 关联 + ILP）全部结束
    "prediction_done": 'print(f"Prediction completed in {predict_seconds / 60:.2f} minutes")',
    # 第 5 段：分裂补全打分器的总开关默认值（只有第 11 段最终 pass 才打开）
    "d1_switch_default": "_D1_ACTIVE = False",
    # 第 5 段：加载 DeepCenter（缺口否决、安全分裂否决、分裂补全打分器的图像特征都要用）
    "deepcenter_load": "DEEPCENTER_VETO_DETECTOR = load_deepcenter_veto_detector()",
    # 第 5 段最后一句：立即对全部影片跑“第一遍”后处理（全部新模块关闭）并写 submission.csv
    "first_pass_call": 'write_test_submission("base")',
    # 第 11 段：最终 pass 在 5 个模块开关全部置 True 之后、写提交之前的那一行
    "final_pass_pp_apply": "_dr_saved_pp = pp_apply(selected_config)",
}

# 段标记的两种写法：教学版（中文、全角冒号）与原始单文件（公开基线 notebook 沿用的英文写法）。
_SECTION_MARKER_RE = re.compile(r"^# ===== 第 (\d+) 段：")
_ORIGINAL_MARKER_RE = re.compile(r"^# ===== ORIGINAL X138 CELL (\d+) =====\s*$")


# =============================================================================================
# 读取推理脚本
# =============================================================================================
def load_pipeline_source(path: str | Path | None = None) -> str:
    """读取第三部分的推理脚本全文（UTF-8）。

    默认读取 ``<仓库>/part3_inference_pipeline/biohub_final_inference.py``。也接受 ``.ipynb``：
    按顺序把全部代码单元拼接起来（教学版 notebook 只有一个代码单元，必须保持单元不拆分，
    因为 ``from __future__ import annotations`` 只对它所在的那一个单元生效）。
    """
    file_path = Path(path) if path is not None else DEFAULT_PIPELINE_PATH
    if not file_path.is_file():
        raise FileNotFoundError(f"找不到推理脚本：{file_path}")
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix == ".ipynb":
        notebook = json.loads(text)
        cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
        if not cells:
            raise ValueError(f"{file_path} 中没有代码单元")
        parts = []
        for cell in cells:
            source = cell.get("source", "")
            parts.append("".join(source) if isinstance(source, list) else str(source))
        text = "\n".join(parts)
    return text


# =============================================================================================
# 打补丁
# =============================================================================================
def apply_patches(source: str, patches: Iterable[tuple[str, str, str]]) -> str:
    """按顺序应用 ``(标签, 旧文本, 新文本)`` 补丁，返回新源码。

    规则（与推理脚本内部给子进程脚本打补丁的规则相同）：
    - 每个“旧文本”在**当前**源码（即已应用前面补丁之后的源码）中必须恰好出现 1 次；
      0 次说明锚点失配（代码被改动、或补丁顺序错了），多次说明锚点不唯一、替换位置不确定，
      两种情况都立刻抛出 ``ValueError``，并在信息中注明是哪个补丁。
    - 补丁按列表顺序依次生效，后面的补丁可以锚定前面补丁写入的文本。
    """
    result = source
    for index, patch in enumerate(patches):
        if not isinstance(patch, (tuple, list)) or len(patch) != 3:
            raise ValueError(f"第 {index} 个补丁必须是 (标签, 旧文本, 新文本) 三元组，实际为 {patch!r}")
        label, old, new = patch
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            raise ValueError(f"补丁 [{label}]：旧文本必须是非空字符串，新文本必须是字符串")
        count = result.count(old)
        if count != 1:
            preview = old.strip("\n")
            if len(preview) > 160:
                preview = preview[:160] + "…"
            raise ValueError(
                f"补丁 [{label}]：锚点出现 {count} 次（必须恰好 1 次）：{preview!r}"
            )
        result = result.replace(old, new, 1)
    return result


def line_patch(label: str, old_line: str, new_text: str) -> tuple[str, str, str]:
    """生成“整行锚点”补丁：``old_line`` 必须作为完整的一行出现（含行首缩进，不含换行符）。

    实现方法是在前后各加一个换行符再做子串匹配：注释里引用同一段代码、缩进不同的同名语句、
    前缀相同的更长语句都不会被误匹配。``new_text`` 可以是多行（行之间用 ``\\n`` 分隔），
    缩进由调用者自己写全。
    """
    if "\n" in old_line:
        raise ValueError(f"补丁 [{label}]：整行锚点不能包含换行符")
    if not old_line.strip():
        raise ValueError(f"补丁 [{label}]：整行锚点不能是空行")
    return (label, "\n" + old_line + "\n", "\n" + new_text.strip("\n") + "\n")


def check_anchors(source: str, anchors: dict[str, str] | None = None) -> dict[str, int]:
    """返回每个锚点在 ``source`` 中作为“完整一行”出现的次数；正常情况下全部为 1。"""
    table = ANCHORS if anchors is None else anchors
    lines = source.splitlines()
    return {key: sum(1 for line in lines if line == text) for key, text in table.items()}


def redirect_test_dir_patch(subset_dir: str | Path) -> tuple[str, str, str]:
    """把第 2 段的 ``TEST_DIR = COMP_DIR / "test"`` 改成指向训练影片子集目录的补丁。

    为什么只改这一行就够：推理脚本里所有“读测试影片”的地方都通过全局变量 ``TEST_DIR``——
    第 4 段列影片、启动推理子进程（``--data-dir``），第 5 段后处理读原始帧（``read_test_frame``
    在调用时读取全局 ``TEST_DIR``）。子集目录里只放 ``<影片>.zarr`` 的符号链接，布局与竞赛
    ``test/`` 目录相同，所以管线完全感觉不到区别。使用绝对路径，因为推理子进程的工作目录是
    ``tracking_repo``。
    """
    target = os.path.abspath(os.fspath(subset_dir))
    return line_patch(
        "TEST_DIR 指向训练影片子集",
        ANCHORS["test_dir"],
        f"TEST_DIR = Path({target!r})",
    )


# =============================================================================================
# 按段切片
# =============================================================================================
def section_markers(source: str) -> dict[int, int]:
    """返回 ``{段号: 标记所在行的下标（从 0 起）}``；同一段号出现两次视为错误。"""
    markers: dict[int, int] = {}
    for line_index, line in enumerate(source.splitlines()):
        match = _SECTION_MARKER_RE.match(line) or _ORIGINAL_MARKER_RE.match(line)
        if match is None:
            continue
        number = int(match.group(1))
        if number in markers:
            raise ValueError(
                f"第 {number} 段的标记出现了两次（第 {markers[number] + 1} 行与第 {line_index + 1} 行）"
            )
        markers[number] = line_index
    return markers


def section_slice(source: str, first: int, last: int | None = None) -> str:
    """取第 ``first`` 段到第 ``last`` 段（含）的源码；``last=None`` 表示到文件末尾。

    - 段的范围：从 ``# ===== 第 first 段：`` 标记行开始，到 ``# ===== 第 last+1 段：`` 标记行之前
      （不含）为止；也接受原始文件的 ``# ===== ORIGINAL X138 CELL N =====`` 标记。
    - 第 0 段标记之前的内容（文件头注释与 ``from __future__ import annotations``）总是保留在最前面：
      这样任何一个切片都能单独编译执行，而且注解的求值方式与完整脚本一致。
    - 原始文本逐字保留（不改行尾、不去空行），所以 ``section_slice(src, 0, None) == src``，
      而且同一个命名空间里先后执行相邻的切片，与一次执行整段代码等价（文件头只是注释和
      一个 ``__future__`` 导入，重复执行无副作用）。
    """
    markers = section_markers(source)
    if 0 not in markers:
        raise ValueError("找不到第 0 段的标记，无法确定文件头的范围")
    if first not in markers:
        raise ValueError(f"找不到第 {first} 段的标记；已有的段：{sorted(markers)}")
    if last is not None:
        if last < first:
            raise ValueError(f"last={last} 小于 first={first}")
        if last not in markers:
            raise ValueError(f"找不到第 {last} 段的标记；已有的段：{sorted(markers)}")
    lines = source.splitlines(keepends=True)
    preamble = lines[: markers[0]]
    start = markers[first]
    if last is None or (last + 1) not in markers:
        if last is not None and last != max(markers):
            raise ValueError(f"第 {last} 段之后缺少第 {last + 1} 段的标记，无法确定结束位置")
        end = len(lines)
    else:
        end = markers[last + 1]
    return "".join(preamble) + "".join(lines[start:end])


# =============================================================================================
# 训练影片子集
# =============================================================================================
def competition_dir() -> Path:
    """与推理脚本第 2 段相同的竞赛数据目录探测：取第一个存在的候选，否则返回第一个候选。"""
    for candidate in COMPETITION_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    return COMPETITION_DIR_CANDIDATES[0]


def list_train_movies(train_dir: str | Path) -> list[str]:
    """列出 ``train_dir`` 中同时有 ``<影片>.zarr``（影像）和 ``<影片>.geff``（GT）的影片名，排序后返回。

    两者缺一都不能用于制作训练数据：没有影像就无法跑推理，没有 GT 就无法打标签。
    排序保证不同会话、不同分片看到同样的顺序（分片按下标切分）。
    """
    root = Path(train_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"训练数据目录不存在：{root}")
    movies = []
    for entry in root.iterdir():
        if not entry.name.endswith(".zarr"):
            continue
        stem = entry.name[: -len(".zarr")]
        if (root / f"{stem}.geff").exists():
            movies.append(stem)
    return sorted(movies)


def shard_movies(movies: Sequence[str], shard: int, num_shards: int) -> list[str]:
    """第 ``shard`` 个分片 = ``movies[shard::num_shards]``（交错切分）。

    交错而不是连续切块：影片名以胚胎前缀（44b6 / 6bba）开头，排序后同一胚胎连在一起；
    交错切分让每个分片都含两个胚胎、耗时也更均匀。
    """
    if int(num_shards) < 1:
        raise ValueError(f"num_shards 必须 ≥ 1，实际为 {num_shards}")
    if not 0 <= int(shard) < int(num_shards):
        raise ValueError(f"shard 必须在 [0, {num_shards}) 内，实际为 {shard}")
    return list(movies)[int(shard)::int(num_shards)]


def remove_train_subset(subset_dir: str | Path) -> None:
    """删除 make_train_subset 建的符号链接（它们指向只读的 /kaggle/input），目录空了再删目录。

    运行结束后这些链接已经没有用；留在 /kaggle/working 里，保存 notebook 输出时可能被当成目录跟进去复制
    （训练影像有几十 GB）。只删符号链接，绝不删真实文件，所以即使传错目录也不会误删数据。
    两个导出脚本都在 try/finally 里调用它：无论成功还是中途出错都会清理。
    """
    subset_dir = Path(os.path.abspath(subset_dir))
    if not subset_dir.is_dir():
        return
    for entry in subset_dir.iterdir():
        if entry.is_symlink():
            entry.unlink()
    if not any(subset_dir.iterdir()):
        subset_dir.rmdir()


def make_train_subset(movies: Sequence[str], train_dir: str | Path, dst_dir: str | Path) -> Path:
    """在 ``dst_dir`` 下为每部影片建符号链接 ``<影片>.zarr -> <train_dir>/<影片>.zarr``，返回该目录。

    为什么用符号链接：Kaggle 的 ``/kaggle/input`` 只读，而一部影片的影像有数百 MB，复制既慢又占
    ``/kaggle/working`` 的空间；符号链接零成本，推理子进程和 ``read_test_frame`` 都能透明地穿过它读取。
    为什么只放 .zarr 不放 .geff：让目录布局与竞赛 ``test/`` 一模一样，管线的行为也就与测试时一样；
    GT 留在 ``train_dir``，由 ``gt_io`` 直接读取。

    目录内容会被整理成**恰好**这些影片：多余的旧符号链接会被删除（第 4 段按目录内容列影片，
    第 6 段也会核对影片集合），但真实的目录或文件绝不删除——遇到时直接报错。
    """
    movie_list = list(dict.fromkeys(str(movie) for movie in movies))
    if not movie_list:
        raise ValueError("影片列表为空")
    source_root = Path(os.path.abspath(os.fspath(train_dir)))
    target_root = Path(os.path.abspath(os.fspath(dst_dir)))
    # 第 3 段会整个删除并重建 tracking_repo，放在里面的子集目录会在推理开始前消失。
    if "tracking_repo" in target_root.parts:
        raise ValueError(f"子集目录不能放在 tracking_repo 里面（第 3 段会删除它）：{target_root}")
    if target_root == source_root:
        raise ValueError("子集目录不能就是训练数据目录本身")
    target_root.mkdir(parents=True, exist_ok=True)

    wanted = {f"{movie}.zarr": source_root / f"{movie}.zarr" for movie in movie_list}
    for name, source in wanted.items():
        if not source.is_dir():
            raise FileNotFoundError(f"训练影像不存在：{source}")

    for entry in target_root.iterdir():
        if not entry.name.endswith(".zarr") or entry.name in wanted:
            continue
        if entry.is_symlink():
            entry.unlink()
        else:
            raise RuntimeError(f"子集目录中有不属于本分片的真实数据，拒绝删除：{entry}")

    for name, source in wanted.items():
        link = target_root / name
        if link.is_symlink():
            if os.readlink(link) == str(source):
                continue
            link.unlink()
        elif link.exists():
            raise RuntimeError(f"子集目录中已有同名的真实目录或文件，拒绝覆盖：{link}")
        os.symlink(str(source), str(link), target_is_directory=True)
    return target_root


# =============================================================================================
# 执行
# =============================================================================================
def run_source(source: str, filename: str, namespace: dict | None = None) -> dict:
    """像 notebook 单元一样执行 ``source``，返回执行后的全局命名空间。

    - 默认使用全新的命名空间 ``{'__name__': '__main__'}``：与 notebook 一样，顶层代码里的
      ``globals()`` 就是这个字典（例如第 5 段 ``_D1_MODULE["install"](globals(), None)`` 把分裂补全
      打分器安装进这个命名空间），所以导出脚本拿到返回值后可以直接读写其中的函数与全局开关。
    - 可选参数 ``namespace``：传入上一次的返回值，可以在同一命名空间里接着执行后面的段
      （例如先跑第 0–4 段，检查产物后再跑第 5 段）。
    - 编译时 ``dont_inherit=True``：不继承本模块自己的 ``__future__`` 设置，源码的语义只由它自己的
      ``from __future__ import annotations`` 决定，与在 notebook 里运行完全一致。
    - 把源码登记到 ``linecache``，出错时的 traceback 能显示 ``filename`` 中对应的代码行。
    """
    ns = {"__name__": "__main__"} if namespace is None else namespace
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    code = compile(source, filename, "exec", dont_inherit=True)
    exec(code, ns)
    return ns
