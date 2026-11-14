# 从现有 46 条轨迹生成小微扰静态数据

**当前推荐 V2：[来源保留、按方向和性质分别检验](reweighting_propertywise.md)。**
下文记录 V1 的定义与旧流程。V1 用所有指标的最慢时间尺度作全局准入，导致仅 8 条进入生成，
不应据此认定其他 38 条来源不可用。V2 按用户确认的温压稳定条件保留 46 条来源，
分别检查目标方向和标签；定义、来源标记和训练分组限制仍保留。

输入是现有 `run_125255` 的逐帧 HDF5，以及 `energy_scan/` 中已按 `%.16g`
导出的 CSV/float64 NPZ。复用轨迹和能量，不生成目标态动力学。
[公式、接受规则、文献依据与限制](reweighting_design.md)。

## 执行顺序

整个批处理在一个 Slurm 作业中分阶段执行，不安装监听器：

1. 严格核对能量 CSV/NPZ/step、温度、物种、盒密度和模型标记，缓存逐帧统计量。
2. 对所有来源状态扫描升温/降温幅度，以及三条晶胞方向的五种光滑周期势形状。
   每一方向的正负幅度分别记录；第一半时间块选择范围，第二半确认，不根据确认段反复调幅。
3. 写出全体 `perturbation_ranges_all.json` 后，才开始生成标签。
4. 每个通过确认的范围采四个内部幅度，再单独筛选温度/外势联合扰动。
   每个最终点仍需检查；范围上限不意味着其组合自动通过。

温度搜索以能量涨落自动定标，另设相对温差 2% 的搜索上限；这不是预先允许 2%。
外势定义为

$$
U_{\rm ext}(X)=k_BT_0 a\sum_m f(s_m),\qquad
s_m=(r_m H^{-1})_d\bmod1.
$$

`f` 为 `cos1`、`sin1`、`cos2`、`sin2`、`cos1+0.5sin2`，周期与晶胞相容。
`a` 是无量纲参数，`kBT0*a` 为每中心势的系数（eV）；混合形状的峰值不是恰好 1。
这是定义明确的**标量中心外势**，不是电场，不引入假定偶极矩或任意电荷。
坐标不变，每帧只改变整盒概率权重。

## 接受含义

帧/时间块 ESS 比例至少 0.90，最大单帧质量最多为均匀值的两倍。
相关时间从全部特征分子和分母、RDF、密度模式及分箱计数响应矩估计；
保留至少八个完整块，并检查更粗分块及前后时间段。
每类标签使用配对分块 jackknife，要求

$$
\mathrm{SE}(\widehat A_{\rm target}-\widehat A_{\rm source})
\leq\mathrm{SE}(\widehat A_{\rm source}).
$$

这是新增统计不确定性的筛选，不要求真实响应小于原误差条，也不证明未采样区域没有偏差。
任何自相关估计、烧入截取或分块检验，都不能取代目标态独立采样的物理确认。
首尾能量变化不能通过除以含巨大元素参考能的绝对总能量来认定已经平衡。

当前能量元数据无原轨迹哈希，原后处理也没有生产模型哈希。
输入同名、step、T、N/V 和能量模型 SHA256 可以核对，但精确构型/生产模型关联仍依据数据生产者声明。
这个未独立核验的前提随每份派生数据保留，`strict_independently_verified_training_eligible=false`。
`training_eligible_by_label` 仅指在该声明成立时，通过有限采样诊断的指定分辨率标签。

## 输出与训练读取

每条轨迹有独立目录：

| 文件 | 内容 |
|---|---|
| `provenance.json` | 原轨迹哈希、能量哈希、模型、T、盒密度、特征基、假设、分块/时间窗 |
| `frames.npz` | 可复用逐帧紧凑统计量，避免每个目标重新读取全部分子特征 |
| `perturbation_ranges.json` | 正/负温差、各势形状的正/负幅度、校准/确认检查及失败点 |
| `states.h5` | 各状态权重、标签、误差与标签级资格 |
| `manifest.json` | 该来源的状态目录与拒绝原因 |

全局有 `perturbation_ranges_all.json`、`catalog.jsonl`、各阶段汇总和最终
`collection_summary.json`。失败输入另写 `.failure.json`，不覆盖或删除失败证据。
来源不满足平衡/时间块检查时，保留原因，不生成合格扰动标签；既有原始结果仍在原目录。

`catalog.jsonl` 的 `origin` 明确区分：

- `direct_estimate`：来源状态在本次记录时间窗内等权统计。
- `reweighted_diagnostics_passed`：至少一类指定标签通过采样检查。
- `reweighted_unqualified`：候选计算完成，但没有合格标签。

必须进一步查看 `training_eligible_by_label`；不能因为一个状态的局部特征通过，就假定它的 c2 也通过。

`states.h5` 每个编号子组包括：

- `frame_weights`：与完整 `source_step`/`source_frame` 对齐；烧入和被截去的尾帧权重为零。
- `local_observables/value,stderr,added_difference_stderr`：508 特征及局部几何、取向/氢键标量的
  有效分子条件均值和误差。根组的 `observable_names` 加 `saved_observable_indices` 给出对应名称。
  没有支持的指标列单独列出；分子取向还必须满足该分子的对称性。
- `connected_structure_factor`：保存非零 k 的连通中心密度协方差，扣除相干密度均值项。
- `pooled_window_count_cumulants`：不同半径固定窗口的平均粒子数、方差、三/四阶累积量及空穴概率；
  外势下这是指定空间窗口的混合统计，不是局部 c3/c4，也不是热力学 S(0)。
- 纯温度分支的 `rdf` 与 `bulk_c2`：先加权 RDF，再做 OZ，误差包含非线性变换；
  另检查窗函数、缩短截断、正谱误差余量和直接非零 k 结构因子。
  合格用途仅限 `training_radial_mask` 和 `training_planar_mask`，k 域结果为辅助有限离散估计。
  截断零模一律不作为合格的热力学 c1 密度积分依据。
- 外势分支的 `profile_6/12/24`：密度、箱平均外势及空间中心化相对 c1。
  c1 为给定箱定义的近似；不自动获得连续空间分辨率资格，不提供未知的绝对 μ。
- `profile_6/12/canonical_projected_c2`：计数协方差在固定总数切空间上的
  Legendre 响应余量，带正交基 Q、均值和误差。这是指定箱/有限 N 的响应量，
  **不是体相径向 OZ c2，也不是无限体系连续位置—取向 c2**。

如需模型逐帧输入，可按 `provenance.json` 的 `source_hdf5` 读取原始逐分子特征，
配合 `frame_weights` 汇总；不要给一帧里的每个分子复制一个状态 c2 当局部真值。
需要其他角度分布时，可使用相同帧权重从原始保存的分子/邻居几何继续计算；
本批次不声称已经为每种原始高维分布生成了独立目标标签。

**同一来源的所有目标共享 `split_group`。** 训练、验证、测试不能随机拆散这些派生状态，
必要时还要按整个物种/状态族划分。增加重加权目标数量不增加独立轨迹数量。

## 提交

```bash
export REWEIGHT_SOURCE_ROOT=/path/to/postprocess_collection
export REWEIGHT_ENERGY_DIR=/path/to/frozen_code/energy_scan
export REWEIGHT_OUTPUT_BASE=/path/to/new_result_base
export REWEIGHT_PYTHON=/path/to/python
export REWEIGHT_ENERGY_COMMIT=353cb2b
sbatch --output="$REWEIGHT_OUTPUT_BASE/logs/%j.out" \
       --error="$REWEIGHT_OUTPUT_BASE/logs/%j.err" tools/reweight_collection.slurm
```

先建立日志目录，并从固定源码快照目录提交。结果为 `run_<jobid>`；已存在目录会拒绝覆盖。
