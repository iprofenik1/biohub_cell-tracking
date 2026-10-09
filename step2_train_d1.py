"""第 2 部分 · 第 2 步（CPU，几秒钟）：在训练表上拟合分裂补全打分器，输出可直接粘贴的 MODEL 字面量。

用法（本地或 Kaggle CPU notebook 均可，只需要 numpy + scipy）::

    python part2_d1_division_scorer/step2_train_d1.py \\
        --table-dir /kaggle/input/<分片0>/d1_table/final /kaggle/input/<分片1>/d1_table/final ... \\
        --groups geo,track,dc,int --l2 10 --threshold 2.0 --out /kaggle/working/d1_model

    # 训练完顺手把新 MODEL 写进第三部分推理脚本的一份副本（.py 或 .ipynb 都行）：
    python part2_d1_division_scorer/step2_train_d1.py --table-dir ... --out /kaggle/working/d1_model \\
        --patch-pipeline part3_inference_pipeline/biohub_final_inference.ipynb

输入：step1_build_table.py 写出的训练表目录（每部影片一个 JSON：``{movie, n_all, metrics?, rows: [...]}``，
rows 的每一行是一个“落在有 GT 标注的母细胞上”的候选，带 31 维特征和标签字段）。可以给多个目录（例如
4 个分片会话各自的输出），行按“目录顺序 → 目录内文件名排序 → 文件内行顺序”拼接。

模型（逐行复刻比赛中实际使用的训练代码，改动任何一步都会让系数对不上）
======================================================================
1. **标签**：正例 y = 1 ⇔ 母细胞 P 匹配到的 GT 节点确实分裂（div）且候选 B 匹配到它的某个 GT 子细胞
   （label）。“紧挨着 GT 分裂、但又不是正例”的行（near_div 且非正例）**身份含糊**：官方分裂指标对
   时间有 ±1 帧的容忍（母细胞一侧 = 分裂节点及其前一帧，子细胞一侧 = 子节点及其后一帧），这些行
   打成正例或负例都不对，所以**不参与拟合**，只在读数表里单独计数。其余都是负例——包括“B 恰好是 P 的
   唯一 GT 子节点”的行（那是该连成普通边的重链接问题，不是分裂）。
2. **分位截断**：每个特征按拟合行的 0.5% / 99.5% 分位数（numpy 默认线性插值）截断到 [lo, hi]。
   少数极端值（例如轨迹断裂导致的超大距离）不会主导线性模型；推理时同样先截断，还让每个特征都有了
   确定的上下界——推理模块里“惰性 DeepCenter 上界”正是利用这一点才是精确的。
3. **标准化**：z = (x − mu) / sd，sd 用总体标准差（ddof = 0），下限 1e-3（常数特征不能变成除以 0）。
   标准化之后各特征的权重大小可以直接比较，L2 正则对每个特征也一视同仁。
4. **类别加权**：正例只有几十个、负例几千个（发布模型的表：46 : 6,681），每个正例的权重 = 负例数 / 正例数
   （约 145），让两类在损失里分量相当；否则模型只要全判负就能把损失压得很低。
   代价：加权等价于把先验几率乘了约 145 倍，logit 整体上移约 ln 145 ≈ 5，所以 logit 不能当校准过的概率读，
   阈值要靠指标扫描来定（见 5）。
5. **损失与优化**：Σ 权重 · [log(1 + e^z) − y·z] + ½ · l2 · ‖w‖²。注意是**求和**而不是平均，截距 b **不**正则；
   相对于上万的加权样本，l2 = 10 是很弱的正则，主要防止个别系数在可分方向上发散。用 scipy 的 L-BFGS-B
   从全零出发、解析梯度、最多 1000 次迭代（其余容差用默认值）。换成“平均损失”或 sklearn 的 C 参数
   都会改变等效正则强度，系数就对不上了。
6. **阈值**：推理时 logit ≥ 2.0 才补边。2.0 来自读数表（跨胚胎打分，推算分数增量最大）和官方指标的阈值
   扫描（样本内 1～2 之间最优、曲线平坦，阈值 2 时样本内 +0.0082；留出胚胎 44b6 / 6bba 上 −0.0023 / +0.0017）。
   本脚本只把 --threshold 记录到输出里，不参与拟合。

两种验证（都在拟合全量模型之前打印）
====================================
- **跨胚胎**：训练集的影片来自两个胚胎（影片名前 4 个字符是胚胎编号）。用一个胚胎的行训练、给另一个
  胚胎打分，模拟“测试集是没见过的胚胎”的情形。注意发布模型的表里 44b6 只有 4 个正例，所以“只用 44b6
  训练、给 6bba 打分”的那个模型几乎没见过正例——这正是跨胚胎读数要按胚胎拆开看的原因。
- **按影片 5 折**：影片按名字排序后用 ``default_rng(0).integers(0, 5, 影片数)`` 随机分折（不保证各折均衡），
  同一部影片的行总在同一折。同一胚胎的影片彼此很像，5 折读数会比跨胚胎乐观。

读数表的口径（与比赛时用的诊断表完全一致）
==========================================
对每个 (影片, 母细胞 P) 只保留分数最高的那个候选（推理时一个母细胞最多补一个分叉）。对每个 logit 阈值：
分叉数 = 分数 ≥ 阈值的母细胞数；其中“真” = 正例，“含糊” = 含糊行，“假” = 其余；精度 = 真 / (真 + 假)；
divJ = 分裂 Jaccard 的推算值 (基线 TP + 真) / (基线 TP + FP + FN + 假)，score 增量 = 0.1 × divJ 的变化
（官方指标里分裂项的权重是 0.1）。基线的分裂 TP / FP / FN 来自训练表 JSON 里的 ``metrics``（官方指标在
“不加分叉”的最终图上的计数）；表里没有 ``metrics`` 时 divJ 两列无意义，脚本会打印 “—”。
只数有 GT 标注的母细胞并不是偷懒：官方分裂指标同样只能评判落在有标注母细胞上的分叉（比赛时阈值 2 在
199 部训练影片上新增 2,327 个分叉，只有 65 个落在有标注的母细胞上，其余既不算对也不算错）。
这仍只是推算：没有模拟推理时的每帧 / 每部影片上限，也没有模拟 ±1 帧窗口的细节（含糊行单列）。在发布模型
的那张表上它与官方指标吻合得很好（阈值 2：读数 25 真 / 39 假、+0.0082；官方指标配对 A/B：分裂 TP +25 /
FP +40、+0.0082），但比赛早期的标签表曾把 TP、FP 都多算约 2 倍，所以读数只用于比较阈值和特征组，
最终以官方指标的 A/B 为准。

输出（--out 目录）
==================
- ``model_literal.txt``：一行 ``MODEL = {...}``，格式与推理模块（d1_module.py / 推理脚本中的 ``_D1_SOURCE``）
  里那一行完全相同，可以整行替换；
- ``d1_model.json``：``{"MODEL": ..., "MODEL_CV": {...}, "D1_THRESHOLD": ..., "summary": {...}}``——MODEL_CV 是两个
  跨胚胎模型（键 = 胚胎编号，值 = “没见过这个胚胎”的模型），只用于离线分析，部署时 ``MODEL_CV = {}``。
- 可选 ``--patch-pipeline``：把推理脚本（.py 或单代码单元的 .ipynb）或 d1_module.py 复制一份，整行替换其中的
  ``MODEL = {...}``，必要时同步 ``D1_THRESHOLD = ...``，并重算只用于报告的 ``_D1_SHA``。一次只处理一个文件；
  要让 .ipynb、.py 与 d1_module.py 保持逐字一致，用同样的参数分别对三个文件各跑一次（拟合是确定性的），
  再用三份副本覆盖原文件（见本部分 README 第 7 节）。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

# 特征分组。顺序就是模型特征向量的顺序（--groups 给出的组按顺序拼接），必须与推理模块 MODEL["features"]
# 一致——推理时按 MODEL["features"] 的名字取值，所以组内顺序本身不影响推理，但影响“系数是否逐位复现”。
#   geo   几何 10 个；track 轨迹 12 个；dc DeepCenter 中心先验 3 个；int 亮度 6 个  → 部署模型共 31 维
#   prob  关联模型给出的链接概率 4 个：训练影片上的概率被检测器/关联模型“记住”了（它们就是在这些影片上
#         训练的），样本内过于自信、换到新胚胎就不可靠，所以部署模型不用；训练表里照样记录，便于分析。
#   holder 另一种候选生成方式（第二子细胞已被别的母细胞占用）的特征，本模块的训练表里没有；选它会被 main 拒绝。
GROUPS = {
    "geo": ["d_pb", "d_pa", "d_ab", "r_a", "r_b", "mid", "cos", "dz_ab", "sym", "flow"],
    "track": ["div1", "div1_miss", "div2", "div2_miss", "div3", "div3_miss", "b_fwd", "a_fwd", "p_back", "mutual_nn",
              "n_unclaimed10", "n_nodes10"],
    "dc": ["dc_b", "dc_a", "dc_p"],
    "int": ["c_b", "c_a", "c_p", "c_next", "drop_next", "ratio_ba"],
    "prob": ["p", "a_p", "p_best_other", "p_margin"],
    "holder": ["q_res", "q_d", "q_back", "q_alt", "q_gain"],
}
# 读数表固定的 logit 阈值网格（部署阈值 2.0 在其中）。
FIXED_THRESHOLDS = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
# 读数表需要的基线分裂计数（训练表 JSON 的 metrics 字段，官方指标口径）。
METRIC_KEYS = ("division_tp", "division_fp", "division_fn")


# =============================================================================================
# 读训练表
# =============================================================================================
def load_tables(table_dirs: list[str | Path]) -> tuple[list[dict], dict[str, dict], list[str], bool]:
    """读取一个或多个训练表目录，返回 (行列表, 每部影片的基线计数, 表名列表, 是否每部影片都有 metrics)。

    - 表名 = 目录名（例如 ``final``）；行的 ``movie`` 字段改写为 ``"<表名>/<影片>"``，同一部影片出现在
      两张表里时不会混在一起（比赛时曾把同一批影片在不同管线上导出的表并排比较）。
    - 读取顺序：目录按命令行顺序，目录内按文件路径排序（``sorted(glob)``），文件内按行顺序——
      行顺序决定特征矩阵的行顺序，进而决定分位数插值与求和的浮点细节，要逐位复现就不能改。
    - 不含 ``rows`` / ``movie`` 的 JSON（例如别的清单文件）跳过；同一 ``表名/影片`` 出现两次直接报错，
      否则这部影片的行会被重复计入。
    """
    rows: list[dict] = []
    base: dict[str, dict] = {}
    tags: list[str] = []
    all_metrics = True
    for table_dir in table_dirs:
        folder = Path(table_dir)
        if not folder.is_dir():
            raise SystemExit(f"训练表目录不存在：{folder}")
        tag = folder.name
        tags.append(tag)
        found = 0
        for path in sorted(glob.glob(str(folder / "*.json"))):
            d = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(d, dict) or "rows" not in d or "movie" not in d:
                print(f"  跳过（不是训练表文件）：{path}")
                continue
            key = f"{tag}/{d['movie']}"
            if key in base:
                raise SystemExit(f"影片 {key} 出现了两次（{path}）：同一部影片的表只能给一次")
            metrics = d.get("metrics")
            if not metrics or any(k not in metrics for k in METRIC_KEYS):
                all_metrics = False
                metrics = {}
            base[key] = dict(metrics, n_all=d["n_all"])
            for r in d["rows"]:
                rows.append(dict(r, movie=key))
            found += 1
        if found == 0:
            raise SystemExit(f"{folder} 中没有训练表 JSON")
    if not rows:
        raise SystemExit("训练表里一行都没有：检查 step1 是否正常结束、GT 匹配是否成功")
    return rows, base, tags, all_metrics


def matrix(rows: list[dict], names: list[str]) -> np.ndarray:
    """特征矩阵 (行数, 特征数)，float64。表里缺的特征记 0（例如没有 holder 组时）。"""
    return np.array([[float(r["feat"].get(n, 0.0)) for n in names] for r in rows], dtype=np.float64).reshape(len(rows), len(names))


# =============================================================================================
# 模型：分位截断 + 标准化 + 类别加权、L2 正则的逻辑回归
# =============================================================================================
def fit(X: np.ndarray, y: np.ndarray, l2: float) -> dict:
    """拟合打分器，返回 {lo, hi, mu, sd, w, b}（另附 nit / success 两个诊断字段，不进 MODEL）。

    每一行运算都与比赛时的训练代码相同（包括运算顺序），不要“顺手优化”：
    例如把求和改成平均、把 ddof 改成 1、给截距加正则、换优化器，系数都会变。
    """
    # 0.5% / 99.5% 分位数（只在拟合行上算）；某特征分位数相等（几乎常数）时让 hi 比 lo 大一点点，保证 hi > lo。
    lo, hi = np.quantile(X, 0.005, axis=0), np.quantile(X, 0.995, axis=0)
    hi = np.where(hi > lo, hi, lo + 1e-6)
    Xc = np.clip(X, lo, hi)
    # 总体标准差（ddof = 0），下限 1e-3：常数特征不能变成除以 0。
    mu, sd = Xc.mean(0), np.maximum(Xc.std(0), 1e-3)
    Z = (Xc - mu) / sd
    # 正例权重 = 负例数 / 正例数，负例权重 1：两类在损失中的总分量相同。
    weight = np.where(y > 0, (len(y) - y.sum()) / max(y.sum(), 1), 1.0)

    def loss(theta):
        # theta = [w_1 .. w_d, b]；z 是每行的 logit。
        z = Z @ theta[:-1] + theta[-1]
        # 加权对数损失（logaddexp(0, z) = log(1 + e^z)，数值稳定）求和 + ½·l2·‖w‖²（不罚截距）。
        value = (weight * (np.logaddexp(0, z) - y * z)).sum() + 0.5 * l2 * (theta[:-1] ** 2).sum()
        # 解析梯度：d/dz = (sigmoid(z) − y) · 权重；sigmoid 里把 z 截到 ±50 防止 exp 溢出。
        gz = (1 / (1 + np.exp(-np.clip(z, -50, 50))) - y) * weight
        return float(value), np.concatenate([Z.T @ gz + l2 * theta[:-1], [gz.sum()]])

    # L-BFGS-B，从全零出发，最多 1000 次迭代，其余容差取 scipy 默认值。
    res = minimize(loss, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 1000})
    return {"lo": lo, "hi": hi, "mu": mu, "sd": sd, "w": res.x[:-1], "b": float(res.x[-1]),
            "nit": int(res.nit), "success": bool(res.success)}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    """logit = 截断 → 标准化 → 点乘 w → 加 b；与推理模块的 logit() 是同一个公式。"""
    return ((np.clip(X, model["lo"], model["hi"]) - model["mu"]) / model["sd"]) @ model["w"] + model["b"]


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC（秩和公式）。同分按出现顺序排名（稳定排序），与比赛时的诊断口径一致，不做平均秩。"""
    order = np.argsort(score, kind="stable")
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score) + 1)
    n1 = y.sum()
    return float((rank[y > 0].sum() - n1 * (n1 + 1) / 2) / max(n1 * (len(y) - n1), 1))


# =============================================================================================
# 读数表
# =============================================================================================
def table(rows, score, y, base, label, thresholds=None, have_metrics=True):
    """按阈值列出“补多少分叉、其中真/假/含糊各多少、推算分数增量”，返回每个阈值一行的元组列表。

    每个 (影片, P) 只留最高分的候选：推理时一个母细胞最多新加一个分叉，读数要与之对应。
    """
    best = {}
    for i, r in enumerate(rows):
        key = (r["movie"], r["P"])
        if key not in best or score[i] > score[best[key]]:
            best[key] = i
    idx = np.array(sorted(best.values()), dtype=int)
    s, yy = score[idx], y[idx]
    amb = np.array([bool(rows[i]["near_div"]) and not y[i] for i in idx])
    # “B 是 P 的唯一 GT 子节点”：真实情况是一条普通的连续边（应由重链接连上），补成分叉就是假分叉。
    relink = np.array([bool(rows[i]["label"]) and not rows[i]["div"] for i in idx])
    tp0 = sum(m.get("division_tp", 0) for m in base.values())
    fp0 = sum(m.get("division_fp", 0) for m in base.values())
    fn0 = sum(m.get("division_fn", 0) for m in base.values())
    n_all = sum(m["n_all"] for m in base.values())
    j0 = tp0 / max(tp0 + fp0 + fn0, 1)
    base_text = f"基线分裂 TP/FP/FN {tp0}/{fp0}/{fn0} divJ {j0:.4f}" if have_metrics else "基线分裂计数 —（表中无 metrics）"
    print(f"  [{label}] 有候选的标注母细胞 {len(idx)}（全部母细胞上的候选 {n_all}）| 可达的真分裂 {int(yy.sum())}"
          f" | {base_text} | AUC {auc(s, yy):.3f}")
    order = np.argsort(-s, kind="stable")
    rows_out = []
    grid = thresholds if thresholds is not None else [s[order[min(n, len(order)) - 1]] for n in (10, 20, 30, 40, 60, 80, 120, 160, 240)]
    for thr in grid:
        top = np.where(s >= thr)[0]
        tp, am, rl = int(yy[top].sum()), int(amb[top].sum()), int(relink[top].sum())
        fp = len(top) - tp - am
        j1 = (tp0 + tp) / max(tp0 + fp0 + fn0 + fp, 1)
        rows_out.append((float(thr), len(top), tp, fp, am, 0.1 * (j1 - j0)))
        tail = (f"divJ {j0:.4f} -> {j1:.4f} = score {0.1 * (j1 - j0):+.5f}" if have_metrics else "divJ — / score —")
        print(f"      logit >= {thr:+7.3f}: 分叉 {len(top):4d} = 真 {tp:3d} + 假 {fp:4d}（其中 B 是 P 的唯一 GT 子节点: {rl}）"
              f" + 含糊 {am:3d} | 精度 {tp / max(tp + fp, 1):.3f} | {tail}")
    return rows_out


def report(rows, score, y, base, title, fixed, have_metrics=True):
    """每张表一个总读数；表里有两个胚胎时再按胚胎各给一个。返回 {读数名: 每阈值的元组列表}。"""
    out = {}
    for tag in sorted({r["movie"].split("/")[0] for r in rows}):
        embryos = sorted({r["movie"].split("/")[1][:4] for r in rows if r["movie"].startswith(tag + "/")})
        for e in ([None] + embryos if len(embryos) > 1 else [None]):
            prefix = tag + "/" + (e or "")
            sel = np.array([r["movie"].startswith(prefix) for r in rows])
            sub_rows = [r for r, k in zip(rows, sel) if k]
            sub_base = {k: v for k, v in base.items() if k.startswith(prefix)}
            label = f"{title} | {tag}" + (f" | 胚胎 {e}" if e else "")
            out[label] = table(sub_rows, score[sel], y[sel], sub_base, label, fixed, have_metrics)
    return out


# =============================================================================================
# MODEL 字面量
# =============================================================================================
def literal(m: dict, names: list[str], tables: list[str], groups: list[str], l2: float, note: str) -> dict:
    """推理模块里 MODEL 字典的格式：数组保留 6 位小数，另记录训练表名、特征组、l2 和备注。

    备注字符串沿用推理模块里的英文原文（"trained on all rows" / "trained without embryo xxxx"），
    这样用比赛时的训练表重训，得到的那一行可以直接与推理脚本中的 MODEL 行逐项核对（6 位小数）。
    推理只用 features / lo / hi / mu / sd / w / b；tables / groups / l2 / note 只是记录。
    """
    return {"features": names, "lo": [round(float(v), 6) for v in m["lo"]], "hi": [round(float(v), 6) for v in m["hi"]],
            "mu": [round(float(v), 6) for v in m["mu"]], "sd": [round(float(v), 6) for v in m["sd"]],
            "w": [round(float(v), 6) for v in m["w"]], "b": round(m["b"], 6), "tables": tables, "groups": groups,
            "l2": l2, "note": note}


def model_line(model_literal: dict) -> str:
    """推理模块中的那一行：``MODEL = `` + json.dumps(字典)（默认分隔符，与推理脚本里的写法一致）。"""
    return "MODEL = " + json.dumps(model_literal)


# =============================================================================================
# 把新 MODEL 写回推理脚本（第三部分）
# =============================================================================================
_MODEL_LINE_RE = re.compile(r"^MODEL = \{.*\}$", re.M)
_THRESHOLD_LINE_RE = re.compile(r"^D1_THRESHOLD = .*$", re.M)
_SHA_LINE_RE = re.compile(r"^_D1_SHA = '[0-9a-f]*'$", re.M)
_SOURCE_OPEN = "_D1_SOURCE = r'''"


def patch_model_text(text: str, new_model_line: str, threshold: float | None) -> tuple[str, list[str]]:
    """在一段源码文本里整行替换 MODEL，并（若存在）同步顶格的 D1_THRESHOLD 行与 _D1_SHA。

    - ``MODEL = {...}`` 必须恰好出现 1 行（推理脚本里它在 ``_D1_SOURCE`` 字符串中、顶格书写；d1_module.py 里也是）。
    - ``D1_THRESHOLD = ...``：推理脚本第 5 段在内嵌模块之前定义了这个全局变量，它**优先于**模块 DEFAULTS
      里的同名值（见 d1_module.py 的 _knobs）。所以改阈值要改这一行；d1_module.py 单独文件里没有这一行，跳过。
    - ``_D1_SHA``：推理脚本把 ``_D1_SOURCE`` 文本的 sha256 前 16 位写进运行报告，只用于记录“跑的是哪一版模块”，
      不参与任何判断；文本变了就顺手重算，免得报告里留下一个对不上的哈希。
    返回 (新文本, 说明列表)。
    """
    notes = []
    count = len(_MODEL_LINE_RE.findall(text))
    if count != 1:
        raise ValueError(f"MODEL = {{...}} 行出现 {count} 次（必须恰好 1 次）")
    text = _MODEL_LINE_RE.sub(lambda _m: new_model_line, text, count=1)
    notes.append("已替换 MODEL 行")
    if threshold is not None:
        hits = _THRESHOLD_LINE_RE.findall(text)
        if len(hits) == 1:
            text = _THRESHOLD_LINE_RE.sub(lambda _m: f"D1_THRESHOLD = {float(threshold)!r}", text, count=1)
            notes.append(f"D1_THRESHOLD 行：{hits[0]!r} -> {f'D1_THRESHOLD = {float(threshold)!r}'!r}")
        elif len(hits) == 0:
            notes.append("没有顶格的 D1_THRESHOLD 行（单独的模块文件）：阈值由推理脚本里的全局变量决定")
        else:
            raise ValueError(f"顶格的 D1_THRESHOLD 行出现 {len(hits)} 次，无法确定改哪一行")
    start = text.find(_SOURCE_OPEN)
    if start >= 0:
        body_start = start + len(_SOURCE_OPEN)
        body_end = text.find("'''", body_start)
        if body_end < 0:
            raise ValueError("_D1_SOURCE 字符串没有结束的三引号")
        sha = hashlib.sha256(text[body_start:body_end].encode("utf-8")).hexdigest()[:16]
        sha_hits = _SHA_LINE_RE.findall(text)
        if len(sha_hits) == 1:
            text = _SHA_LINE_RE.sub(lambda _m: f"_D1_SHA = '{sha}'", text, count=1)
            notes.append(f"_D1_SHA 已重算为 {sha}")
        else:
            notes.append(f"_D1_SHA 行出现 {len(sha_hits)} 次，未改动（它只用于报告）")
    return text, notes


def patch_pipeline_file(src_path: Path, dst_path: Path, new_model_line: str, threshold: float | None) -> list[str]:
    """把推理脚本复制一份并写入新 MODEL。支持 .py 和 .ipynb（只改含 MODEL 行的那个代码单元）。"""
    if src_path.suffix == ".ipynb":
        notebook = json.loads(src_path.read_text(encoding="utf-8"))
        cells = [c for c in notebook.get("cells", []) if c.get("cell_type") == "code"]
        texts = ["".join(c["source"]) if isinstance(c.get("source"), list) else str(c.get("source", "")) for c in cells]
        owners = [k for k, t in enumerate(texts) if _MODEL_LINE_RE.search(t)]
        if len(owners) != 1:
            raise ValueError(f"{src_path}：含 MODEL 行的代码单元有 {len(owners)} 个（必须恰好 1 个）")
        k = owners[0]
        new_text, notes = patch_model_text(texts[k], new_model_line, threshold)
        # 与 .py 分支一样先编译检查：MODEL 行写坏了要在本地就发现，而不是到 Kaggle 上运行时才报错。
        compile(new_text, str(dst_path), "exec", dont_inherit=True)
        cells[k]["source"] = new_text.splitlines(keepends=True)
        # 去掉旧的运行输出，避免别人误以为它们是新模型的结果。
        for cell in cells:
            cell["outputs"] = []
            cell["execution_count"] = None
        dst_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    else:
        new_text, notes = patch_model_text(src_path.read_text(encoding="utf-8"), new_model_line, threshold)
        compile(new_text, str(dst_path), "exec", dont_inherit=True)
        dst_path.write_text(new_text, encoding="utf-8")
    return notes


# =============================================================================================
# 主流程
# =============================================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拟合分裂补全打分器（逻辑回归），输出 MODEL 字面量。")
    parser.add_argument("--table-dir", nargs="+", required=True,
                        help="训练表目录（可多个）；每个目录里每部影片一个 JSON，目录名作为表名")
    parser.add_argument("--groups", default="geo,track,dc,int",
                        help="特征组，逗号分隔，按顺序拼接；部署模型 = geo,track,dc,int（31 维）")
    parser.add_argument("--l2", type=float, default=10.0, help="L2 正则系数（只罚权重，损失是求和）；部署模型用 10")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="推理时的 logit 阈值，不参与拟合；记录到输出里，并在 --patch-pipeline 时写进推理脚本的 D1_THRESHOLD；部署用 2.0")
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working/d1_model"), help="输出目录")
    parser.add_argument("--patch-pipeline", type=Path, default=None,
                        help="可选：推理脚本（.py / .ipynb）或 d1_module.py；写一份换上新 MODEL 的副本")
    parser.add_argument("--patched-out", type=Path, default=None,
                        help="副本路径；默认 <out>/<原文件名>（不会覆盖原文件）")
    args = parser.parse_args(argv)

    groups = args.groups.split(",")
    l2 = float(args.l2)
    # 不是组名的词当作单个特征名（方便做“只加一个特征”的对照实验）。
    names = [n for g in groups for n in GROUPS.get(g, [g])]
    rows, base, tags, have_metrics = load_tables(args.table_dir)
    tables = list(dict.fromkeys(tags))
    name_of = np.array([r["movie"].split("/")[1] for r in rows])
    # 胚胎编号 = 影片名前 4 个字符（训练集只有 44b6、6bba 两个胚胎）。
    embryo = np.array([n[:4] for n in name_of])
    # 正例：母细胞的 GT 节点分裂（div）且 B 匹配到它的 GT 子细胞（label）。
    y = np.array([float(r["label"] and r["div"]) for r in rows])
    # 含糊：紧挨 GT 分裂（near_div = 分裂节点、它的子节点或它的父节点）但不是正例 → 不参与拟合。
    amb = np.array([bool(r["near_div"]) and not (r["label"] and r["div"]) for r in rows])
    use = ~amb
    # 特征名必须在训练表里真实存在：matrix() 对缺失的键补 0，拼错的名字（或候选生成里根本没有的特征）
    # 会变成一列常数照样“训练成功”，到推理时 logit() 取不到这个键就抛异常，每部影片都悄悄退回安全分裂。
    absent = [n for n in names if not any(n in r["feat"] for r in rows)]
    if absent:
        raise SystemExit(f"训练表里没有这些特征：{absent}（检查 --groups 的拼写）")
    X = matrix(rows, names)
    print(f"训练表 {tables}：{len(rows)} 行，正例 {int(y.sum())}（其中第一个子细胞 A 也匹配上的："
          f"{int(sum(r['label'] and r['div'] and r['a_ok'] for r in rows))}），含糊 {int(amb.sum())}"
          f" | {len(names)} 个特征 {groups} | l2 {l2}")
    for e in sorted(set(embryo)):
        print(f"  胚胎 {e}：{int((embryo == e).sum())} 行，正例 {int(y[embryo == e].sum())}")
    if not have_metrics:
        print("  注意：部分影片的表里没有 metrics（官方指标的基线分裂计数），读数表的 divJ / score 两列显示为 —")
    # 单特征 AUC：只用一个特征排序时能把正例排多靠前。小于 0.5 表示“越小越像分裂”（例如 sym、mid）。
    print("  单特征 AUC：" + ", ".join(f"{n} {auc(X[use, k], y[use]):.2f}" for k, n in enumerate(names)))
    fixed = FIXED_THRESHOLDS
    summary = {"tables": tables, "groups": groups, "l2": l2, "threshold": float(args.threshold),
               "rows": len(rows), "positives": int(y.sum()), "ambiguous": int(amb.sum()),
               "embryos": {e: {"rows": int((embryo == e).sum()), "positives": int(y[embryo == e].sum())}
                           for e in sorted(set(embryo))},
               "have_metrics": have_metrics}

    if not use.any():
        raise SystemExit("训练表里全是含糊行，没有可以拟合的行")
    if not y[use].any():
        print("  注意：训练表里没有正例，下面的拟合与读数都没有意义（只适合检查流程能否跑通）")

    # ---- 跨胚胎：给胚胎 e 打分的模型从没见过 e 的任何一行 ----
    # 表里只有一个胚胎时（例如只跑了几部影片的冒烟测试），"另一个胚胎"没有行可训练，跳过这一项。
    # 这个判断只在比赛时的训练代码会直接报错的情形下起作用，不影响正常情形下的逐位复现。
    cv = np.zeros(len(rows))
    models_cv = {}
    if all((use & (embryo != e)).any() for e in set(embryo)):
        for e in sorted(set(embryo)):
            train = use & (embryo != e)
            models_cv[e] = fit(X[train], y[train], l2)
            cv[embryo == e] = predict(models_cv[e], X[embryo == e])
        print("\n跨胚胎读数（每个胚胎由没见过它的模型打分）：")
        summary["cross_embryo"] = report(rows, cv, y, base, "跨胚胎", fixed, have_metrics)
    else:
        print("\n跨胚胎读数：跳过（训练表里只有一个胚胎，没有“另一个胚胎”可用来训练）")
        summary["cross_embryo"] = {}

    # ---- 按影片 5 折：同一部影片的行永远在同一折 ----
    movies = sorted(set(name_of))
    rng = np.random.default_rng(0)
    fold_of = {m: int(k) for m, k in zip(movies, rng.integers(0, 5, len(movies)))}
    fold = np.array([fold_of[m] for m in name_of])
    cv5 = np.zeros(len(rows))
    # 影片太少时可能所有影片都落进同一折，那一折就没有训练行；同样只在原代码会报错时跳过。
    if all((use & (fold != k)).any() for k in set(fold.tolist())):
        for k in range(5):
            model = fit(X[use & (fold != k)], y[use & (fold != k)], l2)
            cv5[fold == k] = predict(model, X[fold == k])
        print("\n按影片 5 折读数：")
        summary["five_fold"] = report(rows, cv5, y, base, "按影片5折", fixed, have_metrics)
    else:
        print("\n按影片 5 折读数：跳过（影片太少，有一折没有训练行）")
        summary["five_fold"] = {}

    # ---- 全量模型（部署用）：全部非含糊行 ----
    model = fit(X[use], y[use], l2)
    print(f"\n全量模型：L-BFGS-B 迭代 {model['nit']} 次，收敛 {model['success']}；截距 b = {model['b']:+.6f}")
    # 标准化后的权重可以直接比较大小；看主要符号是否符合直觉（例如 sym、mid 负，d_ab、dc_b 正）。
    # 特征彼此高度相关（几个距离都由同一组位置算出），单个权重是“其他特征不变”时的效应，个别符号不宜单独解读。
    print("  全量模型的权重（标准化后，按绝对值排序）："
          + ", ".join(f"{n} {w:+.2f}" for n, w in sorted(zip(names, model["w"]), key=lambda kv: -abs(kv[1]))))
    for e, m in models_cv.items():
        if not m["success"]:
            print(f"  注意：不含胚胎 {e} 的跨胚胎模型没有收敛（{m['nit']} 次迭代）")

    final_literal = literal(model, names, tables, groups, l2, "trained on all rows")
    cv_literals = {e: literal(m, names, tables, groups, l2, f"trained without embryo {e}") for e, m in models_cv.items()}
    line = model_line(final_literal)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_literal.txt").write_text(line + "\n", encoding="utf-8")
    (out_dir / "d1_model.json").write_text(json.dumps({"MODEL": final_literal, "MODEL_CV": cv_literals,
                                                       "D1_THRESHOLD": float(args.threshold), "summary": summary},
                                                      ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("\n下面这一行可以整行替换推理模块里的 MODEL 行（阈值另由推理脚本的 D1_THRESHOLD 全局变量给出，"
          f"本次 --threshold {float(args.threshold)!r}）：")
    print(line)
    print(f"已写出：{out_dir / 'model_literal.txt'}、{out_dir / 'd1_model.json'}")

    if args.patch_pipeline is not None:
        src_path = Path(args.patch_pipeline)
        dst_path = Path(args.patched_out) if args.patched_out is not None else out_dir / src_path.name
        if dst_path.resolve() == src_path.resolve():
            raise SystemExit("--patched-out 不能与原文件相同：请先写副本，核对后再替换")
        for note in patch_pipeline_file(src_path, dst_path, line, float(args.threshold)):
            print("  " + note)
        print(f"已写出换上新 MODEL 的副本：{dst_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
