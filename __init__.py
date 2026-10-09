"""两个训练数据导出脚本共用的工具包。

- ``common.pipeline_runner``：读取第三部分的推理脚本，按精确锚点打补丁（例如把测试影片目录换成训练影片
  子集），再像 notebook 一样执行其中需要的段。这样训练数据与最终推理走的是同一条管线，分布完全一致。
  只依赖标准库。
- ``common.gt_io``：读取竞赛训练集的真值图（``train/<影片>.geff``），并提供逐帧匹配工具
  （坐标头训练对用的贪心最近邻匹配；分裂补全打分器打标签用的、与官方指标相同的最优二分匹配）。
  依赖 numpy，scipy / tracksdata 在用到时才导入。

step 脚本从仓库根目录运行，例如 ``python part1_cv10_coord_head/step1_capture_features.py``；
脚本开头把仓库根目录加入 ``sys.path`` 后即可 ``from common import pipeline_runner, gt_io``。
这里不在包级别导入子模块，避免只想用 pipeline_runner 时也被迫导入 numpy。
"""

__all__ = ["gt_io", "pipeline_runner"]
