"""坐标修正模块（推理期）：在每个检测点首次被检出时，用冻结的 U-Net 特征回归亚体素位移。

第三部分推理管线的第 4 段把本文件原样写成 scripts/v1284_coordinate_refinement.py，并给推理脚本
（支持包的 scripts/predict_unet_transformer.py）打 4 处补丁（导入本模块、删掉坐标转 int16 的一行，以及下面两处）。
与本模块直接相关的两处是：
  1) 每帧检测出峰之后、写入节点表之前调用 refine()，修正这一帧全部检测点的坐标；
  2) 把关联网络的 _index_features 换成本模块的 index_features（三线性插值取特征）。
文件名与环境变量名中的 V1284 是“坐标修正头”框架的代号，为了与推理管线保持一致没有改名。

为什么要修正坐标：
  检测在降采样网格上进行（z 不变，y、x 每 4 个像素取 1 个），网格三个方向都是 1.625 µm，
  检测峰只能落在格点上，存在量化误差。在训练影像上，检测与相配 GT 的平均距离约 1.6～1.9 µm（随检测器和
  配对半径而变），其中 z 方向的误差最大：GT 和检测的 z 都只能取整数层，相邻两层就差 1.625 µm。
  这里用一个很小的 MLP 读取检测器自己的特征（U-Net 冻结，不再训练），为每个检测点预测一个
  亚体素位移（单位 µm）。修正后的浮点坐标进入之后的关联、ILP、后处理和输出。
  比赛按 7 µm 半径匹配节点，坐标修正几乎不改变“能否配上”；收益来自关联打分（三线性取特征、
  位置编码、坐标差）和后处理中各种距离门限变得更准。

三种模式（环境变量 V1284_MODE，由推理管线第 4 段设置）：
  zero      不修正，只把坐标转成 float32，用作“没有坐标头”的对照；
  capture   不修正，把每帧的 (检测坐标, 224 维特征) 存盘：这就是生成坐标头训练数据的方式
            （第一部分 step1 用它导出特征，step2 再与 GT 配对得到训练样本）；
  其它值    推理管线用 candidate：加载 V1284_HEAD 中的全部头（10 折坐标头共 10 个），
            对它们各自的有界位移取平均，再加到检测坐标上。
"""
import os
from pathlib import Path
import numpy as np
import torch

# 降采样网格的格距（µm，顺序 z, y, x）。原始体素 z = 1.625 µm、y/x = 0.40625 µm，y/x 每 4 个取 1 个后
# 三个方向都是 1.625 µm（各向同性）。头输出的位移单位是 µm，除以 SPACING 才是网格单位；
# 末尾的安全检查再乘回 SPACING，把网格位移换回 µm。
SPACING = np.array([1.625, 1.625, 1.625], dtype=np.float32)
# 7 个采样位置（网格单位，顺序 z, y, x）：中心，以及 -z、+z、-y、+y、-x、+x 六个轴向邻点（各相距 1 格 = 1.625 µm）。
# 这个顺序决定 224 维特征的排列。训练特征就是用本模块的 capture 模式导出的，所以训练与推理天然一致。
OFFSETS = ((0,0,0), (-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1))
# 已加载的头：每个推理子进程第一次调用 refine 时从磁盘读入，之后每帧复用。
_CACHE = None


def make_head():
    """坐标头网络：MLP 224 → 32 → 3（SiLU 激活），共 7,299 个参数。

    输入是 sample_features 给出的 224 维特征（推理时先用该头训练折上的均值、尺度标准化），
    输出 3 维向量 δ（z, y, x，单位 µm），再经 bounded() 变成真正的位移。
    最后一层的权重和偏置都初始化为 0：未训练时无论输入是什么，输出都恒为 0，即“不修正”（恒等映射）。
    训练从恒等出发，只学对损失有帮助的修正，不会一开始就随机挪动坐标；第一步只有最后一层收到梯度，
    它一旦离开 0，前一层也开始学习。推理时 load_state_dict 会覆盖这些初值，零初始化只影响训练的起点。
    """
    head = torch.nn.Sequential(torch.nn.Linear(224, 32), torch.nn.SiLU(), torch.nn.Linear(32, 3))
    torch.nn.init.zeros_(head[-1].weight)
    torch.nn.init.zeros_(head[-1].bias)
    return head


def bounded(head, x):
    """有界输出：把头的原始输出 δ 映射成 2δ/(1+‖δ‖)，其中 ‖δ‖ 是三维向量的模长，方向保持不变。

    映射后的模长 2‖δ‖/(1+‖δ‖) 恒小于 2，所以每个头给出的修正位移都小于 2 µm：只做网格间距（1.625 µm）
    量级的微调，即使输入异常也不会把检测点挪到很远的地方。‖δ‖ 很小时约等于 2δ（近似线性），‖δ‖ 很大时饱和。
    训练时 Huber 损失直接作用在这个有界输出上，训练与推理用的是同一个映射。
    """
    delta = head(x)
    return 2.0 * delta / (1.0 + torch.linalg.vector_norm(delta, dim=-1, keepdim=True))


def sample_features(feature, arr):
    """取每个检测点的 224 维输入特征 = 7 个位置 × 32 通道。

    feature：主模型在这一帧的 32 通道 U-Net 特征图，形状 (1, 32, Z, Y, X)，网格就是检测用的 1.625 µm 网格。
      它是开启特征 TTA 之后的 8 视角平均特征（其中 x 翻转视角重复计了一次，见推理管线第 4 段），
      取自这一帧第一次出现的那个滑窗。训练坐标头时抓取的正是同一份特征，训练与推理的输入分布才一致。
    arr：这一帧的检测，形状 (N, 4)，列为 [t, z, y, x]，是检测峰所在体素的整数网格坐标。
    返回 (N, 224)：[中心特征, 邻点1 − 中心, …, 邻点6 − 中心]。
    “邻点 − 中心”相当于特征沿 ±z、±y、±x 的有限差分（局部梯度），告诉网络真实中心偏向哪一侧；
    中心特征本身提供这个细胞的外观与所处位置的上下文。
    """
    xyz = torch.as_tensor(arr[:, 1:], device=feature.device, dtype=torch.long)
    blocks = []
    for offset in OFFSETS:
        loc = xyz + torch.tensor(offset, device=feature.device)
        # 越界的邻点截断到图像边界上（此时该方向的差分为 0），保证索引合法。
        for axis, size in enumerate(feature.shape[-3:]):
            loc[:, axis].clamp_(0, size-1)
        blocks.append(feature[0, :, loc[:,0], loc[:,1], loc[:,2]].T)
    # 中心特征 + 6 个方向差分（邻点 − 中心），拼成 7 × 32 = 224 维。
    return torch.cat([blocks[0]] + [b - blocks[0] for b in blocks[1:]], dim=1)


def index_features(self, maps, coords, mask):
    """三线性插值取特征，替换关联网络原来的 _index_features（推理管线把它挂到模型类上，主、副模型都生效）。

    原实现把坐标截断成整数（.long()）后直接取该体素的特征；坐标修正后检测点带小数，截断会丢掉修正。
    这里取包围该点的 8 个体素，按到各体素的距离做三线性加权，关联网络就能读到亚体素位置的特征。
    坐标为整数时 frac = 0，只有 (0,0,0) 角的权重为 1、其余 7 个角权重为 0，结果与原来的整数取址完全相同：
    所以不修正坐标时（zero / capture 模式），关联结果与原流程一致。
    maps：(B, C, Z, Y, X) 特征图；coords：(B, N, 3) 网格坐标 (z, y, x)；mask：(B, N)，有效点排在前面。
    """
    out = torch.zeros((*coords.shape[:2], maps.shape[1]), device=maps.device, dtype=maps.dtype)
    for batch in range(len(maps)):
        n = int(mask[batch].sum())
        if not n:
            continue
        q = coords[batch, :n].clone()
        # 先把坐标截断到网格范围内，再取左下角体素 low 与小数部分 frac。
        for axis, size in enumerate(maps.shape[-3:]):
            q[:,axis].clamp_(0, size-1)
        low = q.floor().long()
        frac = q-low
        # 遍历 8 个角：某一轴取 +1 的角，在该轴上的权重为 frac，否则为 1 − frac；三轴权重相乘。
        for z in (0,1):
            for y in (0,1):
                for x in (0,1):
                    shift = torch.tensor([z,y,x], device=maps.device)
                    loc = low+shift
                    for axis, size in enumerate(maps.shape[-3:]):
                        loc[:,axis].clamp_(0,size-1)
                    weight = torch.where(shift.bool(), frac, 1-frac).prod(dim=1)
                    out[batch,:n] += maps[batch,:,loc[:,0],loc[:,1],loc[:,2]].T * weight[:,None]
    return out


def refine(ds_path, t, arr, feature):
    """修正第 t 帧全部检测点的坐标；每帧检测完成后、写入节点表之前调用一次。

    ds_path：当前影像的 .zarr 路径（capture 模式用它的文件名建子目录）；t：帧号；
    arr：(N, 4) 整数网格坐标 [t, z, y, x]；feature：这一帧的 32 通道特征图（见 sample_features）。
    返回 (N, 4) 坐标；candidate 模式下为 float32，z、y、x 带亚体素小数（仍是网格单位）。
    推理脚本的 predict_video 带有 @torch.no_grad()，这里的前向不会记录梯度。
    """
    global _CACHE
    # 模式由推理管线通过环境变量传给子进程；没有设置时直接 KeyError，避免“以为修正了，其实没有”。
    mode = os.environ['V1284_MODE']
    # 空帧（没有检测）原样返回。
    if not len(arr):
        return arr
    # zero：不修正，只转成 float32（与修正后的数据类型一致），作“无坐标头”的对照。
    if mode == 'zero':
        return arr.astype(np.float32)
    x = sample_features(feature, arr).float()
    # capture：导出训练数据。把整数网格坐标 [t, z, y, x]（z、y、x 乘 1.625 才是 µm）和 224 维特征存成
    # <V1284_CAPTURE>/<影像名>/<帧号:04d>.npz（键 coords、features）。坐标不变，所以下游结果等同于不加坐标头。
    # 之后与 GT 配对，得到 (特征, GT − 检测 的位移) 训练样本。
    if mode == 'capture':
        folder = Path(os.environ['V1284_CAPTURE']) / ds_path.stem
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder/f'{int(t):04d}.npz', coords=arr, features=x.cpu().numpy())
        return arr
    # 其它模式（推理管线用 candidate）：第一次调用时加载 V1284_HEAD 中的全部头（多个路径用 os.pathsep 连接，
    # 10 折坐标头是 fold0.pt … fold9.pt）。每个 .pt 里有 state_dict，以及该头在自己的训练折上算出的
    # 输入均值 mean 和标准差 scale（下限 1e-3，防止除以 0；标准化必须与训练时完全一致）。
    if _CACHE is None:
        _CACHE = []
        for _path in os.environ['V1284_HEAD'].split(os.pathsep):
            saved = torch.load(_path, map_location='cpu', weights_only=True)
            head = make_head().to(feature.device)
            head.load_state_dict(saved['state_dict']); head.eval()
            _CACHE.append((head, saved['mean'].to(feature.device), saved['scale'].to(feature.device)))
    # 10 折集成：每个头先用自己的 mean/scale 标准化输入，各自输出有界位移（µm），再对所有头取平均。
    # 10 个头各自见过不同的约 90% 训练影像，误差不完全相关，取平均相当于小型集成，能降低方差。
    # 每个有界位移的模长都小于 2 µm，它们的平均（凸组合）模长也小于 2 µm。只有 1 个头时就是单头的计算。
    # 平均位移除以 SPACING（1.625 µm/格）换算成网格单位。
    shift = torch.stack([bounded(head, (x-mean)/scale) for head, mean, scale in _CACHE]).mean(dim=0).cpu().numpy() / SPACING
    result = arr.astype(np.float32).copy()
    result[:,1:] += shift
    # 修正后的点截断在网格范围内（不能落到图像外）；第 0 列帧号不变。
    result[:,1:] = np.clip(result[:,1:], 0, np.asarray(feature.shape[-3:])-1)
    # 安全检查：出现非有限值，或任何一个点的实际位移超过 2 µm（理论上限是 2 µm，多留 0.00001 µm 容纳 float32 舍入误差），
    # 说明头文件或输入有问题，直接报错，而不是悄悄输出坏坐标。
    if not np.isfinite(result).all() or np.max(np.linalg.norm((result[:,1:]-arr[:,1:])*SPACING,axis=1)) > 2.00001:
        raise RuntimeError('invalid V1284 displacement')
    return result
