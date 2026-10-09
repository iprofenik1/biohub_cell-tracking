# 分裂补全打分器（推理期模块）
# 本文件的全文就是第三部分推理脚本中 _D1_SOURCE 字符串的内容（逐字相同）：推理脚本把它 exec 到一个独立命名空间，
# 再调用 install(globals(), None) 把打分器接到后处理链上。第二部分的训练表导出（step1_build_table.py）用的也是
# 这同一份代码（dump 模式）。代码行与比赛中实际运行的版本完全一致，这里只加了中文注释和中文文档字符串。
"""分裂补全打分器：在公开基线的"安全分裂"之后，用一个小型逻辑回归追加分叉（只增不删）。

一、为什么要"学"一个打分器
  1. 指标 = 调整后的边 Jaccard + 0.1 × 分裂 Jaccard，分裂项只认出度 ≥ 2 的预测节点（分叉）。ILP 本身允许一个
     母细胞接两个子细胞，但后处理的运动重链接用匈牙利算法做一对一分配，并（正常情况下）替换掉全部 ILP 边，所以
     重链接之后的图里没有分叉，分叉只能靠后面的分裂阶段补。训练集 199 段影像一共只有 151 次 GT 分裂；分裂项的
     权重虽然只有 0.1，但每一次都很值钱：样本内多做对一次分裂约 +0.00056，多一个假分叉只扣约 0.00007，
     盈亏平衡精度只要约 10%，所以宁可多试。
  2. 公开基线只在 add_safe_divisions_postlink 里用一串手工门限产生分叉（最近邻、两个子细胞各恰有一个后继时的
     继续分开、对称、DeepCenter 否决、数量上限），样本内 151 次分裂只做对 23 次；放宽门限又会让假分叉暴增——
     普通步长与分裂步长的分布严重重叠，只靠距离挑不出分裂。
  3. 误差分析：训练影像上漏掉的 128 次 GT 分裂中，有 44 次的第二个子细胞已经在图里，只是一条"没有父亲的轨迹
     起点"。只要补一条"母细胞 → 第二个子细胞"的边，就恢复了一次分裂。问题因此变成：对每个 (P, A, B) 三元组
     做二分类，判断它是不是一次真分裂。

二、候选三元组（candidates）
  P = t 帧中恰有一个子节点 A（在 t+1 帧）的节点；B = t+1 帧中没有父亲的节点（轨迹起点）。
  三道宽松的几何门限（µm）：|P − B| ≤ 12，|A − B| ≤ 16，A 与 B 的中点到 P 的预测位置 ≤ 8。
  预测位置 = P + 局部 flow（当前图中 t→t+1、离 P 最近的 12 条边位移的中位数），即"P 若不分裂，下一帧应在哪里"。
  门限只负责缩小候选集，真正的判断交给模型。训练影像上候选能覆盖 151 次分裂中的 46 次，这就是打分器的上限。

三、31 维特征（4 组）
  几何 10 个：P–B 距离 d_pb、P–A 距离 d_pa、A–B 距离 d_ab、A 到预测点距离 r_a、B 到预测点距离 r_b、
             中点到预测点距离 mid、A/B 偏移夹角余弦 cos、A–B 深度差 dz_ab、偏移不对称度 sym、局部 flow 幅度 flow
  轨迹 12 个：1/2/3 帧后分离增量 div1/div2/div3、1/2/3 帧后分离缺失 div1_miss/div2_miss/div3_miss、
             B 后续轨迹长度 b_fwd、A 后续轨迹长度 a_fwd、P 之前轨迹长度 p_back、B 是 A 最近无父起点 mutual_nn、
             附近无父起点数 n_unclaimed10、附近节点数 n_nodes10
  DeepCenter 中心先验 3 个：B 处中心先验 dc_b、A 处中心先验 dc_a、P 处中心先验 dc_p
  亮度 6 个：B 亮度对比 c_b、A 亮度对比 c_a、P 亮度对比 c_p、P 原位下一帧对比 c_next、P 原位亮度变化 drop_next、
             A、B 亮度差异 ratio_ba
  直觉：两个子细胞应大致对称地分居预测点两侧、之后继续分开；B 处应确有一个细胞（中心先验高）；母细胞原来的位置
  在下一帧会变暗，因为它已一分为二并离开了原位。（可选的第 5 组"链接概率"只在 dump 模式下记录，部署的模型不用。）

四、打分与阈值
  特征先截断到训练集的 0.5% / 99.5% 分位 [lo, hi]，再标准化 (x − mu) / sd，与权重 w 点乘再加偏置 b，得到 logit。
  logit ≥ 2.0 的候选进入补边队列。训练时正例加了权，logit 不是校准过的概率；2.0 是按官方指标扫描选出的排序阈值。

五、惰性 DeepCenter（精确）与时间预算（不精确）
  DeepCenter 热图每帧要跑一次 3D 网络，是最贵的一步。logit 对截断后的特征是线性的，把 3 个 DeepCenter 特征代入
  "最有利值"（权重为正取 hi、为负取 lo）就得到 logit 的严格上界；连上界都过不了阈值的候选不查热图。
  被打分的集合与全量计算完全相同，训练影像上只有约 4% 的候选需要查热图。
  另有每部影片 30 s 的图像特征时间预算：超时后剩下的候选不再打分。这一步不精确——机器很慢时输出会与全量计算
  不同；它只是防止个别影片拖垮 12 小时总时限的保险（可见集上整个打分阶段每部影片约 10～14 s，低于预算）。

六、补边规则
  按 logit 从高到低贪心补 P → B：每个母细胞、每个起点只用一次；B 不能已有父亲；P 的出度必须恰为 1。
  每帧最多 max(1, round(该帧节点数 × 2%)) 个、每部影片最多 max(1, round(边数 × 1%)) 个新分叉。
  公开基线自己的分叉一个不删。容错：任何异常都返回安全分裂阶段的结果，提交不受影响。

七、dump 模式（导出训练表）
  kernel 全局变量 D1_DUMP 是一个列表时，只把全部候选及其全部特征追加进列表、不改图（不用惰性上界，也不受时间
  预算限制）。训练表就是这样在训练影像上导出的，所以训练与推理共用同一个特征函数：特征的定义和实现只有一份，
  不会因为训练、推理各写一套代码而产生偏差。
  注意：stage 先检查开关 _D1_ACTIVE，所以 dump 模式同样要求 kernel 全局变量 _D1_ACTIVE 为真，否则列表保持为空。
"""
from __future__ import annotations

import time

# 元数据：NAME（模块的版本标签）、DESCRIPTION、STATS_KEYS 只供记录和离线评估工具使用，kernel 不读取。
# STATS_KEYS 列出本模块写进每部影片统计表（run_stats.csv）的计数字段，便于检查打分器在每部影片上做了什么：
#   m_d1_sources 母细胞数、m_d1_candidates 候选数、m_d1_heat_candidates 真正查了热图的候选数、
#   m_d1_time_skipped 因超时未打分的候选数、m_d1_scored 过阈值的候选数、m_d1_added 实际补上的分叉数、
#   m_d1_cap_skipped 因上限被跳过的候选数、m_d1_errors 出错次数、m_d1_seconds 用时。
NAME = "d1_pw10c_final"
DESCRIPTION = "Learned division scorer (geometry + track + image features), adds forks after the kernel's safe divisions."
STATS_KEYS = ["m_d1_sources", "m_d1_candidates", "m_d1_scored", "m_d1_added", "m_d1_cap_skipped", "m_d1_errors", "m_d1_seconds",
              "m_d1_heat_candidates", "m_d1_time_skipped"]

# 默认参数。kernel 全局里若有同名变量（推理脚本第 5 段在内嵌本源码之前定义了 D1_THRESHOLD 等），以全局值为准（见 _knobs）。
DEFAULTS = {
    # 门限 1：母细胞 P → 候选 B 的原始距离（不扣除 flow）不超过 12 µm。
    # 分裂的母→子步长本来就大（90% 分位约 9 µm），所以比普通链接放得宽。
    "D1_MAX_PARENT_UM": 12.0,     # candidate generation: mother -> candidate, raw
    # 门限 2：现有子细胞 A 与候选 B 的最大间距（姐妹间距）16 µm；姐妹间距的中位数约 10.6 µm。
    "D1_MAX_SISTER_UM": 16.0,     # existing child -> candidate
    # 门限 3：两个子细胞的中点到 P 的 flow 预测位置的最大距离 8 µm。
    "D1_MAX_MID_UM": 8.0,         # midpoint of the daughters against the mother's flow-predicted position
    # 逻辑回归 logit 的阈值。
    "D1_THRESHOLD": 2.0,          # on the model's logit
    # 每帧新增分叉上限的比例。注意：代码（stage 中的 sources_per_frame）统计的是该帧的全部节点数，
    # 不只是母细胞候选，右侧英文注释写的 "sources" 不准确，以代码为准。
    "D1_FRAME_FRAC_CAP": 0.02,    # new forks per frame, fraction of the sources of the frame
    # 每部影片新增分叉上限 = 边数 × 1%。
    "D1_GLOBAL_FRAC_CAP": 0.01,   # new forks per movie, fraction of the edges
    # 局部 flow 取最近 12 条边；轨迹长度类特征（b_fwd、a_fwd、p_back）封顶 6 帧。
    "D1_FLOW_K": 12,
    "D1_LIFE_CAP": 6,
    # 每部影片计算图像特征的时间预算 30 s；超时后剩下的候选不打分。
    "D1_MAX_SECONDS": 30.0,       # budget per movie for the image features; candidates after it are not scored
}

# MODEL 由第二部分的训练脚本 step2_train_d1.py 生成（它输出的就是下面这个字典字面量）。训练方法：
#   每个特征按训练表的 0.5% / 99.5% 分位数截断（lo / hi），再标准化（mu / sd，sd 下限 1e-3）；
#   类别极不平衡（约 6,760 行候选里只有 46 个正例，另有 33 个含糊行不参与拟合），所以正例按"负例数 / 正例数"加权；
#   L2 正则系数 10（只罚权重、不罚截距），用 L-BFGS-B 最小化加权对数损失。
#   正例 = 母细胞的 GT 节点确实分裂，且 B 匹配到它的某个 GT 子细胞；紧挨 GT 分裂却不是正例的候选，
#   因为指标允许早或晚 1 帧而标签含糊，不参与拟合。
#   部署的这组系数拟合自比赛中较早一版管线导出的训练表（当时用公开坐标头；候选概率重链接用未经关联特征修正的
#   概率；没有找回概率打分；找回阈值 0.965；ILP 分裂权重为公开基线的取值，部署时为 0.4）。特征函数与现在完全相同，只是输入的图略有差别，所以用最终管线
#   重新导出训练表再拟合，得到的系数会有小幅差异，这是正常的。
# 为什么用逻辑回归：正例只有几十个，复杂模型很容易过拟合；线性模型的权重可以逐个检查符号是否符合生物学直觉，
#   例如 偏移不对称度 sym 为负、A–B 距离 d_ab 为正、B 处中心先验 dc_b 为正、P–A 距离 d_pa 为负。
# 打分只用 features / lo / hi / mu / sd / w / b；tables / groups / l2 / note 只记录训练配置（训练表名、特征组、
#   正则系数、备注）。
# MODEL_CV 供离线跨胚胎验证使用：以影片名前 4 个字符（胚胎编号）为键，该胚胎的影片由"没见过这个胚胎"的模型打分；
#   部署版为空字典，所有影片都用 MODEL。
MODEL = {"features": ["d_pb", "d_pa", "d_ab", "r_a", "r_b", "mid", "cos", "dz_ab", "sym", "flow", "div1", "div1_miss", "div2", "div2_miss", "div3", "div3_miss", "b_fwd", "a_fwd", "p_back", "mutual_nn", "n_unclaimed10", "n_nodes10", "dc_b", "dc_a", "dc_p", "c_b", "c_a", "c_p", "c_next", "drop_next", "ratio_ba"], "lo": [2.448793, 0.046345, 3.542905, 0.0343, 2.952938, 0.73338, -0.995011, 0.019889, 0.111989, 0.02772, -3.402026, 0.0, -4.000569, 0.0, -4.658971, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.001126, 0.005726, 0.005452, 0.000852, 0.021582, 0.024178, -0.620013, -1.3265, 0.0012], "hi": [11.977296, 10.485302, 15.526533, 6.561809, 14.852751, 7.762169, 0.986574, 13.185784, 1.984885, 9.152932, 6.408745, 1.0, 8.392196, 1.0, 10.320039, 1.0, 6.0, 6.0, 6.0, 1.0, 2.0, 7.0, 0.506047, 0.531592, 0.5339, 1.113205, 1.213366, 1.160295, 0.96956, 0.2097, 1.543474], "mu": [8.889406, 2.595021, 9.797805, 1.580937, 9.37478, 4.628661, -0.131946, 4.984972, 1.419937, 1.928905, 0.601882, 0.03761, 1.062762, 0.058124, 1.395041, 0.07849, 5.535603, 5.59447, 5.057083, 0.907091, 0.659283, 3.044894, 0.189169, 0.274468, 0.273868, 0.275767, 0.346952, 0.348286, 0.220021, -0.128616, 0.276579], "sd": [2.244214, 1.865648, 2.400534, 1.199692, 2.424355, 1.423947, 0.581777, 3.194513, 0.402842, 1.632958, 1.502838, 0.19025, 2.048998, 0.233978, 2.478845, 0.268941, 1.216507, 1.201166, 1.752065, 0.290305, 0.582827, 1.436037, 0.121271, 0.13196, 0.132238, 0.202048, 0.236264, 0.23226, 0.231756, 0.219304, 0.289671], "w": [-0.558303, -1.779386, 1.738813, 0.325492, 0.606021, -1.341639, 0.147214, 0.135142, -1.798294, -0.341071, 1.367594, -0.173286, 0.500037, -1.319845, -0.013831, 0.305234, -0.636592, -1.230608, -0.518446, -0.041567, -0.120584, -0.407722, 1.410103, -0.598574, 0.760385, -0.043112, 0.761937, -0.053072, -0.69009, -0.622365, -0.678791], "b": -8.195921, "tables": ["pubw"], "groups": ["geo", "track", "dc", "int"], "l2": 10.0, "note": "trained on all rows"}
MODEL_CV = {}

# 特征分组（每个特征的中文名见模块文档字符串）。部署的 MODEL 用几何 + 轨迹 + DeepCenter + 亮度共 31 维。
# FEATURES_PROB（关联模型给出的链接概率）刻意不用：公开权重（检测 + 关联网络）就是在这些训练影像上训练的，样本内的链接概率被
#   "记住"了、过于自信，泛化到新胚胎时不可靠。dump 模式仍会记录这 4 个特征，便于分析。
FEATURES_GEO = ["d_pb", "d_pa", "d_ab", "r_a", "r_b", "mid", "cos", "dz_ab", "sym", "flow"]
FEATURES_TRACK = ["div1", "div1_miss", "div2", "div2_miss", "div3", "div3_miss", "b_fwd", "a_fwd", "p_back", "mutual_nn",
                  "n_unclaimed10", "n_nodes10"]
FEATURES_DC = ["dc_b", "dc_a", "dc_p"]
FEATURES_INT = ["c_b", "c_a", "c_p", "c_next", "drop_next", "ratio_ba"]
FEATURES_PROB = ["p", "a_p", "p_best_other", "p_margin"]


# 参数读取：kernel 全局（ns）里有同名变量且不为 None 时用全局值，否则用 DEFAULTS。
# 这样调参只需在推理脚本里改一行全局变量，不必改这份内嵌源码。
def _knobs(ns):
    return {k: (ns[k] if ns.get(k) is not None else v) for k, v in DEFAULTS.items()}


def candidates(ns, nodes_by_id, edges, dataset, bundle, frame_cache, dc_cache, stats, probs=None, with_image=True,
               lazy=None):
    """生成全部候选三元组 (母细胞 P, 现有子细胞 A, 无父起点 B)，并计算它们的特征。

    返回列表，每个元素是 {"P", "A", "B", "t", "feat"}，按帧号排序。
    lazy = (model, threshold) 时启用"惰性 DeepCenter"：热图是最贵的部分（每帧一次网络前向）。若把 DeepCenter 特征
    换成最有利的取值，logit 仍到不了阈值，这个候选就不查热图，并标记 c["out"] = True。被打分的集合与全量计算完全相同。
    lazy = None（dump 模式）时每个候选都计算全部特征。
    """
    # 本模块通过 exec 运行在独立命名空间里，numpy、cKDTree、读帧函数、DeepCenter 打分函数都从 kernel 全局 ns 中取，
    # 保证与后处理其他部分用的是同一份实现（例如 read_test_frame 读的就是 kernel 的 TEST_DIR）。
    np = ns["np"]
    cKDTree = ns["cKDTree"]
    knob = _knobs(ns)
    scale = np.array(ns["VOXEL_SCALE_UM"], dtype=np.float64)
    life = int(knob["D1_LIFE_CAP"])

    # 坐标换算：节点坐标是原始体素索引 (z, y, x)，乘 VOXEL_SCALE_UM（z 1.625 µm，y/x 0.40625 µm）变成 µm。
    # z 方向的体素比 y/x 粗 4 倍，所有距离都必须在 µm 下比较。children / parent 来自当前图（已含安全分裂补的边）。
    ids = sorted(nodes_by_id)
    vox = {i: (float(nodes_by_id[i]["z"]), float(nodes_by_id[i]["y"]), float(nodes_by_id[i]["x"])) for i in ids}
    pos = {i: np.array(vox[i]) * scale for i in ids}
    t_of = {i: int(nodes_by_id[i]["t"]) for i in ids}
    children, parent = {}, {}
    for e in edges:
        s, g = int(e["source_id"]), int(e["target_id"])
        if s in pos and g in pos:
            children.setdefault(s, []).append(g)
            parent[g] = s
    ids_by_t = {}
    for i in ids:
        ids_by_t.setdefault(t_of[i], []).append(i)

    # 局部运动场：对每个 t，收集当前图中全部 t→t+1 边的位移（P→A 自己也在其中）；至少 4 条边才建场，否则 flow = 0。
    # flow(t, 点) = 起点离该点最近的 12 条边的位移，逐轴取中位数。胚胎里的细胞成片协同运动，邻居位移的中位数比单个位移稳健，
    # 少数错连的边也带不偏它。P + flow 就是"P 若不分裂，下一帧应在的位置"（下文称预测点）。
    field = {}
    for t, frame_ids in ids_by_t.items():
        src, disp = [], []
        for i in frame_ids:
            for g in children.get(i, []):
                if t_of[g] == t + 1:
                    src.append(pos[i])
                    disp.append(pos[g] - pos[i])
        if len(src) >= 4:
            field[t] = (cKDTree(np.stack(src)), np.stack(disp))

    def flow(t, point):
        if t not in field:
            return np.zeros(3)
        tree, disp = field[t]
        _d, idx = tree.query(point, k=min(int(knob["D1_FLOW_K"]), len(disp)))
        return np.median(disp[np.atleast_1d(idx)], axis=0)

    # 轨迹辅助函数：
    #   descendant(i, k)：沿"唯一子节点"链走 k 步，途中断开或分叉就返回 None（用于分离增量 div_k）；
    #   forward(i)：从 i 沿唯一子节点链往后能走几帧，封顶 6；backward(i)：沿父节点链往前能走几帧，封顶 6。
    # 轨迹长短反映可信度：长期被追踪的母细胞和能延续下去的子细胞更可能是真的，极短的残片多是假检测或断裂碎片。
    def descendant(i, k):
        for _ in range(k):
            nxt = children.get(i, [])
            if len(nxt) != 1:
                return None
            i = nxt[0]
        return i

    def forward(i):
        n = 0
        while n < life and len(children.get(i, [])) == 1:
            i = children[i][0]
            n += 1
        return n

    def backward(i):
        n = 0
        while n < life and i in parent:
            i = parent[i]
            n += 1
        return n

    # DeepCenter 中心先验：独立训练的公开 3D U-Net 输出"细胞中心"热图（在 xy 池化后的网格上），
    # 取该点周围 z±1、池化网格 y/x±2 小窗内的最大值。没有加载 DeepCenter 时返回 −1（之后取 max(0, ·) 变成 0）。
    def dc(t, v):
        if bundle is None:
            return -1.0
        value = ns["deepcenter_score_point"](dataset, int(t), v, bundle, frame_cache, dc_cache)
        return -1.0 if value is None else float(value)

    # 局部亮度：小框（z±1、y/x±4 原始体素，约 4.9×3.7×3.7 µm，大致一个细胞核）的均值，
    # 和大框（z±3、y/x±12，约 11.4×10.2×10.2 µm，代表局部背景）的中位数；两者都至少取 1，防止 log(0)。
    # 后面用两者的对数比当"对比度"：除以局部背景，消除不同影像、不同深度整体亮度的差异。
    def intensity(t, v):
        frame = ns["read_test_frame"](dataset, int(t), frame_cache)
        z, y, x = int(round(v[0])), int(round(v[1])), int(round(v[2]))
        patch = frame[max(0, z - 1):z + 2, max(0, y - 4):y + 5, max(0, x - 4):x + 5]
        wide = frame[max(0, z - 3):z + 4, max(0, y - 12):y + 13, max(0, x - 12):x + 13]
        if patch.size == 0 or wide.size == 0:
            return 1.0, 1.0
        return max(float(patch.mean()), 1.0), max(float(np.median(wide)), 1.0)

    # by_target：每个目标节点收到的全部 (概率, 源) 列表，只用于可选的链接概率特征（dump 模式记录，部署模型不用）。
    by_target = {}
    if probs:
        for (s, g), p in probs.items():
            if p is None:
                continue
            cur = by_target.get(g)
            if cur is None:
                by_target[g] = [(float(p), s)]
            else:
                cur.append((float(p), s))

    # ---- 候选生成 ----
    # 对每个 t：B 取 t+1 帧全部无父节点（轨迹起点）；P 取 t 帧中恰有一个子节点、且子节点在 t+1 帧的节点。
    # 已经有两个子节点的母细胞（例如公开基线刚补上分叉的）不再参与：每个母细胞最多补一个第二子细胞。
    out = []
    n_sources = 0
    for t in sorted(ids_by_t):
        next_ids = ids_by_t.get(t + 1)
        if not next_ids:
            continue
        starts = [i for i in next_ids if i not in parent]
        if not starts:
            continue
        start_arr = np.stack([pos[i] for i in starts])
        start_tree = cKDTree(start_arr)
        next_tree = cKDTree(np.stack([pos[i] for i in next_ids]))
        sources = [i for i in ids_by_t[t] if len(children.get(i, [])) == 1 and t_of[children[i][0]] == t + 1]
        n_sources += len(sources)
        if not sources:
            continue
        # 门限 1：|P − B| ≤ 12 µm（对全部无父起点做 KD 树球查询）。
        near = start_tree.query_ball_point(np.stack([pos[i] for i in sources]), r=float(knob["D1_MAX_PARENT_UM"]))
        for P, hits in zip(sources, near):
            if not hits:
                continue
            A = children[P][0]
            # 预测点 = P + 局部 flow；r_a = A 相对预测点的残差向量。
            f = flow(t, pos[P])
            predicted = pos[P] + f
            ra = pos[A] - predicted
            na = float(np.linalg.norm(ra))
            d_pa = float(np.linalg.norm(pos[A] - pos[P]))
            nn_of_a = None
            for j in hits:
                B = starts[j]
                # 门限 2：姐妹间距 |A − B| ≤ 16 µm；门限 3：A、B 的中点到预测点 ≤ 8 µm——
                # 真分裂时两个子细胞对称地落在"母细胞本应到达的位置"两侧，中点应贴近预测点。
                d_ab = float(np.linalg.norm(pos[A] - pos[B]))
                if d_ab > float(knob["D1_MAX_SISTER_UM"]):
                    continue
                mid = float(np.linalg.norm((pos[A] + pos[B]) / 2.0 - predicted))
                if mid > float(knob["D1_MAX_MID_UM"]):
                    continue
                rb = pos[B] - predicted
                nb = float(np.linalg.norm(rb))
                # mutual_nn 用：t+1 帧全部无父起点中离 A 最近的那个（每个 P 只查一次）。
                if nn_of_a is None:
                    _d, k = start_tree.query(pos[A])
                    nn_of_a = starts[int(k)]
                # ---- 几何特征（10 个，距离单位 µm）----
                # d_pb / d_pa / d_ab：三边长；r_a / r_b：A、B 到预测点的距离；mid：中点到预测点的距离；
                # cos：残差向量 (A − 预测点) 与 (B − 预测点) 的夹角余弦，真分裂两子分居两侧，cos 接近 −1；
                #   （cos 其实可由 r_a、r_b、mid 推出：4·mid² = r_a² + r_b² + 2·r_a·r_b·cos，信息与它们重复；模型里它的权重也很小）；
                # dz_ab：两子的 z 差；sym = |r_a − r_b| / max((r_a + r_b)/2, 0.5)：两子离预测点是否一样远（0.5 µm 是防除零的下限）；
                # flow：局部整体运动的幅度（运动快的区域，各种距离本身就偏大，模型需要知道这一点）。
                # ---- 轨迹特征（这里 6 个，div_k 在下面）----
                # b_fwd / a_fwd：B、A 往后的轨迹长度；p_back：P 往前的轨迹长度（都封顶 6 帧）；
                # mutual_nn：离 A 最近的无父起点是否就是 B（只检查 A→起点这一个方向）；
                # n_unclaimed10：预测点 10 µm 内的无父起点数（封顶 10）；n_nodes10：预测点 10 µm 内 t+1 帧的全部节点数（封顶 15）
                #   ——越拥挤越容易配错。
                feat = {
                    "d_pb": float(np.linalg.norm(pos[B] - pos[P])), "d_pa": d_pa, "d_ab": d_ab, "r_a": na, "r_b": nb, "mid": mid,
                    "cos": float(np.dot(ra, rb) / max(na * nb, 1e-6)), "dz_ab": float(abs(pos[A][0] - pos[B][0])),
                    "sym": abs(na - nb) / max((na + nb) / 2.0, 0.5), "flow": float(np.linalg.norm(f)),
                    "b_fwd": float(forward(B)), "a_fwd": float(forward(A)), "p_back": float(backward(P)),
                    "mutual_nn": float(nn_of_a == B),
                    "n_unclaimed10": float(min(len(start_tree.query_ball_point(predicted, 10.0)), 10)),
                    "n_nodes10": float(min(len(next_tree.query_ball_point(predicted, 10.0)), 15)),
                }
                # 分离增量 div_k（k = 1、2、3）：A、B 各沿唯一子节点链再走 k 帧后的间距，减去当前姐妹间距，截断到 [−10, 15] µm。
                # 分裂后两个子细胞会持续远离（div_k > 0）；检测抖动造成的"假姐妹"不会系统性地分开。
                # 链在途中断开或再分叉时记 div_k = 0、div_k_miss = 1，让模型单独学习"缺失"本身意味着什么。
                for k in (1, 2, 3):
                    da, db = descendant(A, k), descendant(B, k)
                    if da is None or db is None:
                        feat[f"div{k}"], feat[f"div{k}_miss"] = 0.0, 1.0
                    else:
                        feat[f"div{k}"] = float(np.clip(np.linalg.norm(pos[da] - pos[db]) - d_ab, -10.0, 15.0))
                        feat[f"div{k}_miss"] = 0.0
                # 可选的链接概率特征：p = P→B 的概率，a_p = P→A 的概率，p_best_other = 其他源指向 B 的最大概率，p_margin = 两者之差。
                if probs is not None:
                    p = float(probs.get((P, B), 0.0) or 0.0)
                    others = [pp for pp, ss in by_target.get(B, []) if ss != P]
                    best_other = max(others, default=0.0)
                    feat.update({"p": p, "a_p": float(probs.get((P, A), 0.0) or 0.0), "p_best_other": best_other,
                                 "p_margin": p - best_other})
                out.append({"P": int(P), "A": int(A), "B": int(B), "t": int(t), "feat": feat})

    # ---- 图像特征（亮度 6 个 + DeepCenter 3 个）：全部候选生成之后再算 ----
    # 惰性 DeepCenter 的上界：对每个 DeepCenter 特征，权重为正取上界 hi、为负取下界 lo，即"最有利值"（best_case）。
    # 打分时特征会先截断到 [lo, hi] 再线性组合，所以代入最有利值得到的就是 logit 在任何热图取值下的严格上界；
    # 上界仍低于阈值的候选，无论热图是多少都不会被选中，跳过热图查询不改变结果（精确）。
    if with_image:
        best_case = None
        if lazy is not None:
            model, threshold = lazy
            index = {name: k for k, name in enumerate(model["features"])}
            best_case = {}
            for name in FEATURES_DC:
                if name in index:
                    k = index[name]
                    best_case[name] = float(model["hi"][k]) if float(model["w"][k]) > 0 else float(model["lo"][k])
        # 时间预算：从这里开始计时，超过 30 s 后剩下的候选直接标记 out、不再打分。这一步不精确（机器慢时结果会变），
        # 只是防止个别影片候选过多拖垮总时限的保险。dump 模式（lazy = None）不受预算限制，每个候选都算全部特征。
        # 候选按帧号顺序处理，原始帧缓存和热图缓存只需保留最近几帧（右侧英文注释的意思）。
        started = time.time()
        budget = float(knob["D1_MAX_SECONDS"])
        n_heat = n_late = 0
        for c in out:                     # ordered by frame: the frame and heat-map caches stay small
            feat, t = c["feat"], c["t"]
            vp, va, vb = vox[c["P"]], vox[c["A"]], vox[c["B"]]
            if lazy is not None and time.time() - started > budget:
                c["out"] = True
                n_late += 1
                continue
            # 亮度特征（都是对数比，并截断到固定范围，防止极端值主导）：
            #   c_b / c_a / c_p：B、A（t+1 帧）和 P（t 帧）相对局部背景的对比度——这里是否真有一个细胞核；
            #   c_next：t+1 帧 P 原位置的亮度相对 t 帧 P 的局部背景；drop_next：P 原位置从 t 到 t+1 的亮度变化——
            #     分裂后两个子细胞向两侧移开，母细胞原来的位置在下一帧会变暗；
            #   ratio_ba：两个子细胞的亮度差异，一分为二的姐妹细胞亮度相近。
            ib, bb = intensity(t + 1, vb)
            ia, ba = intensity(t + 1, va)
            ip, bp = intensity(t, vp)
            inext, _ = intensity(t + 1, vp)
            feat["c_b"] = float(np.clip(np.log(ib / bb), -2, 3))
            feat["c_a"] = float(np.clip(np.log(ia / ba), -2, 3))
            feat["c_p"] = float(np.clip(np.log(ip / bp), -2, 3))
            feat["c_next"] = float(np.clip(np.log(inext / bp), -2, 3))
            feat["drop_next"] = float(np.clip(np.log(inext / ip), -3, 2))
            feat["ratio_ba"] = float(np.clip(abs(np.log(ib / ia)), 0, 3))
            # 先用便宜的特征 + DeepCenter 最有利值算 logit 上界；过不了阈值就不查热图。
            if best_case is not None:
                if logit(model, {**feat, **best_case}, np) < threshold:
                    c["out"] = True
                    continue
            # 可能过阈值的候选才真正查 3 次热图：B、A 在 t+1 帧，P 在 t 帧；热图缺失时为 −1，取 max(0, ·)。
            n_heat += 1
            feat["dc_b"], feat["dc_a"], feat["dc_p"] = (max(0.0, dc(t + 1, vb)), max(0.0, dc(t + 1, va)), max(0.0, dc(t, vp)))
        stats["m_d1_heat_candidates"] = n_heat
        stats["m_d1_time_skipped"] = n_late
    stats["m_d1_sources"] = n_sources
    stats["m_d1_candidates"] = len(out)
    return out


# 打分：特征按 MODEL["features"] 的顺序排成向量 → 截断到 [lo, hi] → 标准化 (x − mu) / sd → 与 w 点乘再加 b，得到 logit。
# 截断让少数极端值不主导模型，也给每个特征确定的上下界——上面的惰性上界正是利用了这一点。
def logit(model, feat, np):
    x = np.array([float(feat[name]) for name in model["features"]], dtype=np.float64)
    lo, hi = np.asarray(model["lo"], dtype=np.float64), np.asarray(model["hi"], dtype=np.float64)
    x = np.clip(x, lo, hi)
    return float(((x - np.asarray(model["mu"])) / np.asarray(model["sd"])) @ np.asarray(model["w"]) + float(model["b"]))


# 安装：把 kernel 全局的 add_safe_divisions_postlink 换成包装器 stage，motion_relink_edges 换成 relink_keeping_probs。
# 后处理总链 filter_output_graph 在调用时按全局名字查找这两个函数，所以替换后每一遍后处理都会经过包装器。
# 参数 source_of 未使用。
def install(ns, source_of):
    base_stage = ns["add_safe_divisions_postlink"]
    base_relink = ns["motion_relink_edges"]
    held = {"probs": None}

    # 透明包装：先记下重链接用到的链接概率字典，再原样调用原函数，结果不变。只有 dump 模式（记录概率特征）用得到；
    # 找回补进了节点时会再重链接一次，held 里留下的是最后一次调用的概率。
    def relink_keeping_probs(nodes_by_id, stats, learned_edge_probs=None):
        held["probs"] = learned_edge_probs
        return base_relink(nodes_by_id, stats, learned_edge_probs)

    # 包装后的分裂阶段：
    #   1. 先运行公开基线自带的安全分裂（手工门限），它补的分叉一个都不删；
    #   2. 只有 kernel 全局 _D1_ACTIVE 为真（最终 pass）时才继续，dump 模式同样要求它为真；
    #   3. 在安全分裂之后的图上生成候选、打分、贪心补边。
    def stage(nodes_by_id, edges, stats, dataset=None, deepcenter_bundle=None, frame_cache=None, deepcenter_cache=None):
        out = base_stage(nodes_by_id, edges, stats, dataset=dataset, deepcenter_bundle=deepcenter_bundle,
                         frame_cache=frame_cache, deepcenter_cache=deepcenter_cache)
        # D1_DUMP 为列表即 dump 模式；MODEL_CV 为空，所以 model 总是 MODEL。
        dump = ns.get("D1_DUMP")
        model = MODEL_CV.get(str(dataset)[:4], MODEL) if dataset else MODEL
        # 开关：推理脚本定义了 _D1_ACTIVE（默认 False），因此只看它；_G1X1_ACTIVE 只是没定义 _D1_ACTIVE 时的后备。
        # 没有影片名或图里没有边时也直接返回。
        if not ns.get("_D1_ACTIVE", ns.get("_G1X1_ACTIVE", False)) or not dataset or not out:
            return out
        if dump is None and model is None:
            return out
        # 下面整个过程包在 try 里：任何异常都只记一次 m_d1_errors，并返回安全分裂的结果，保证提交不受影响。
        started = time.time()
        try:
            np = ns["np"]
            knob = _knobs(ns)
            frame_cache = frame_cache if frame_cache is not None else {}
            deepcenter_cache = deepcenter_cache if deepcenter_cache is not None else {}
            # dump 模式：全部特征都算（含链接概率），不用惰性上界和时间预算，保证训练表的每一行都完整。
            # 打分模式：部署模型不含概率特征（need_prob 为假），含图像特征（need_image 为真），启用惰性上界。
            need_prob = dump is not None or any(name in FEATURES_PROB for name in model["features"])
            need_image = dump is not None or any(name in FEATURES_DC + FEATURES_INT for name in model["features"])
            cands = candidates(ns, nodes_by_id, out, dataset, deepcenter_bundle, frame_cache, deepcenter_cache, stats,
                               probs=held["probs"] if need_prob else None, with_image=need_image,
                               lazy=None if dump is not None or not need_image else (model, float(knob["D1_THRESHOLD"])))
            # dump 模式：只把候选及特征追加到列表、不改图。训练表就是这样在训练影像上导出的。
            if dump is not None:
                dump.extend(cands)
                return out
            # 打分：跳过被惰性上界或时间预算标记为 out 的候选，其余 logit ≥ 2.0 的进入队列。
            # 阈值 2.0 来自官方指标的阈值扫描（1～2 之间最优，曲线平坦）；正例加了权，它不对应校准概率 sigmoid(2) ≈ 0.88。
            scored = []
            for c in cands:
                if c.get("out"):
                    continue
                value = logit(model, c["feat"], np)
                if value >= float(knob["D1_THRESHOLD"]):
                    scored.append((value, c))
            stats["m_d1_scored"] = len(scored)
            # 按 logit 从高到低排序，同分时按 P、B 编号排序，保证结果可复现。
            scored.sort(key=lambda item: (-item[0], item[1]["P"], item[1]["B"]))
            # 上限：每帧 max(1, round(该帧全部节点数 × 2%))——变量名叫 sources_per_frame，实际统计的是该帧全部节点；
            # 每部影片 max(1, round(边数 × 1%))。防止模型在个别碎片化的影片上系统性出错时一次加入大量假分叉。
            sources_per_frame = {}
            for node in nodes_by_id.values():
                sources_per_frame[int(node["t"])] = sources_per_frame.get(int(node["t"]), 0) + 1
            global_cap = max(1, int(round(len(out) * float(knob["D1_GLOBAL_FRAC_CAP"]))))
            used_sources, used_targets, per_frame = set(), set(), {}
            has_parent = {int(e["target_id"]) for e in out}
            out_degree = {}
            for e in out:
                out_degree[int(e["source_id"])] = out_degree.get(int(e["source_id"]), 0) + 1
            added = []
            skipped = 0
            # 贪心补边 P → B：每个母细胞、每个起点只用一次；B 不能已有父亲；P 当前出度必须恰为 1（不产生三叉）。
            # 达到上限的候选只是跳过（continue，不是 break），其他帧的候选仍有机会。
            # 分裂项对假分叉的惩罚很轻（盈亏平衡精度约 10%），所以阈值和上限都可以相对宽松。
            for value, c in scored:
                P, B, t = c["P"], c["B"], c["t"]
                if P in used_sources or B in used_targets or B in has_parent or out_degree.get(P, 0) != 1:
                    continue
                frame_cap = max(1, int(round(sources_per_frame.get(t, 0) * float(knob["D1_FRAME_FRAC_CAP"]))))
                if len(added) >= global_cap or per_frame.get(t, 0) >= frame_cap:
                    skipped += 1
                    continue
                # 新边不带链接概率（edge_prob = None）；safe_division / d1_division / d1_logit 只是标记，下游不读取。
                added.append({"source_id": P, "target_id": B, "edge_prob": None, "distance_um": float(c["feat"]["d_pb"]),
                              "safe_division": 1, "d1_division": 1, "d1_logit": round(value, 3)})
                used_sources.add(P)
                used_targets.add(B)
                per_frame[t] = per_frame.get(t, 0) + 1
            stats["m_d1_added"] = len(added)
            stats["m_d1_cap_skipped"] = skipped
            stats["m_d1_seconds"] = round(time.time() - started, 2)
            # safe_divisions_added 是安全分裂与打分器补的分叉总数（后处理日志一起打印）。
            if added:
                stats["safe_divisions_added"] = int(stats.get("safe_divisions_added", 0)) + len(added)
                return [*out, *added]
            return out
        except Exception as exc:  # noqa: BLE001
            stats["m_d1_errors"] = stats.get("m_d1_errors", 0) + 1
            print(f"  [{dataset}] D1 division scorer skipped (non-fatal): {type(exc).__name__}: {exc}")
            return out

    # 替换 kernel 全局中的两个函数。
    ns["add_safe_divisions_postlink"] = stage
    ns["motion_relink_edges"] = relink_keeping_probs
