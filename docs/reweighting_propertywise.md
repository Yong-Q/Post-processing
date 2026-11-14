# V2：来源轨迹保留，微扰与性质分别检验

适用于当前 46 条轨迹。用户已确认生产轨迹温度、压力稳定；按这个声明保留全部来源，
不再以最慢的局部特征挡住整条轨迹。旧作业 126008 的结果与拒绝证据原样保留。

这里没有重新独立确认温压稳定：已有缓存没有瞬时温度，能量文件的压力仅为位力项，
不能把设定温度或位力压力自动标成完整热力学温压轨迹。来源准入依据明确记录为用户确认。
温压稳定不单独证明任意目标分布有足够重叠，目标标签仍需后处理检查。

## 修正的层次

1. **来源准入**：全部 46 条保留，可直接统计；不是只保留上版通过全局检查的 8 条。
2. **微扰方向**：温度看能量 U，单独外势看其耦合的 F，联合扰动同时看 U、F。
   一个方向的相关时间长，不阻断其他方向。按完整生产段使用缓存，时间窗会截到完整块。
3. **具体性质**：c1、径向 OZ c2、受限计数响应、结构因子、窗口累积量和局部特征分别检查。
   各标签从自己的 delta-method 影响量检查时间相关，并选可容纳至少八块的分块尺度；
   非线性变换仍在每个 jackknife 复制样本内完整重算，不用线性近似代替标签误差。
4. **局部特征逐列**：稀疏缺失列不删除其他列；逐列输出 `training_eligible_element_mask`。
   一个很慢或时间不稳定的特征不再删除整个 508 维向量或整个状态。

相关时间用于防止把相近帧当独立观测、低估误差。它并不改变精确的平衡重加权公式，
也不表示慢性质不可测。接受标准仍是重叠、配对新增统计误差、分块和时间段一致性；
原热库和重加权值都保留各自误差。没有额外证明未采到的状态不存在，资格仍有明确适用边界。

## 数据与管理

程序入口 `tools/run_reweight_propertywise.py`，Slurm 入口 `tools/reweight_propertywise.slurm`。
复用 `run_126008/<case>/frames.npz` 的逐帧 RDF/特征等缓存，不重读全部原子轨迹或补能量。
复用前核对原始轨迹身份记录、CSV/NPZ/缓存能量和 step、模型哈希，并记录缓存 SHA256。
同一文件系统使用只读用途的硬链接；跨文件系统使用指向原缓存的符号链接，不修改旧缓存。
引用的旧缓存需要保留。

新结果按 `groups/<direct|temperature|field_XX|joint_XX>/<case>` 保存；
每组包含独立范围、来源和标签。只有 `direct` 组生成来源状态，其他组避免重复直接状态。
状态 ID 带来源和组名，不冲突；全部派生状态继续共享原轨迹 `split_group`。
`source_admission/` 保存每条来源准入依据，`ranges/` 保存分方向范围，
`perturbation_ranges_all.json` 在生成数据前统一发布；`catalog.jsonl` 合并最终标签记录。

局部指标的 `status=partially_qualified` 表示必须按元素掩码取值，不能整向量当作合格标签。
标签的 `block_frames_used` 给出实际误差分块长度；原始关联不充分时只将对应标签置为未合格。
源/目标 delta-method 影响量的自相关检查只是一阶诊断，配合两种块尺度和前后时间段检查，
不冒充独立目标态验证。

相对 c1、有限 N 分箱响应、体相径向 OZ c2 的定义及零模限制继续遵守
[原数据定义](reweighting.md)。能量缺少原构型哈希的未决前提和严格独立验证标记也继续传播。

## 提交示例

```bash
export REWEIGHT_CACHE_ROOT=/path/to/run_126008
export REWEIGHT_SOURCE_ROOT=/path/to/run_125255
export REWEIGHT_ENERGY_DIR=/path/to/frozen_code/energy_scan
export REWEIGHT_OUTPUT_BASE=/path/to/new_v2_output
export REWEIGHT_PYTHON=/path/to/python
sbatch --output="$REWEIGHT_OUTPUT_BASE/logs/%j.out" \
       --error="$REWEIGHT_OUTPUT_BASE/logs/%j.err" tools/reweight_propertywise.slurm
```

按用户要求提交后不监听；实际范围和资格由作业自行写出。
