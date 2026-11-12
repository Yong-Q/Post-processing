# 水 MD 后处理程序

直接处理现有 LAMMPS `.dump` 或 `.dump.gz`，不启动 MD。当前下载样本来自 Zenodo 24123911：H2O-T986.4-r00-rho0.6700 和 H2O-T986.4-r04-rho0.3811，各约 112 MB，均已核对原始字节数和 MD5。样本类型 O=4、H=2，667 个 O；记录说明存帧间隔 0.5 ps。文件名的密度只是标签，报告另按实际盒体积计算密度。

## 运行

建议使用 Python 3.10。克隆后在仓库根目录建立环境；轨迹需自行放入 `data/`，不包含在代码仓库中。运行时限制本地线程，避免多个进程抢占全部 CPU：

```bash
git clone https://github.com/Yong-Q/Post-processing.git
cd Post-processing
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -m waterpost inspect \
  --input data/H2O-T986.4-r00-rho0.6700.dump.gz
```

先做小样本运行，输出路径必须不存在：

```bash
.venv/bin/python -m waterpost analyze \
  --input data/H2O-T986.4-r00-rho0.6700.dump.gz \
  --output results/my_check \
  --start 200 --max-frames 24 --block-frames 6 \
  --expensive-stride 4 --pair-stride 4 --max-lag 6
```

`--start 200` 在这里仅用于测试包含换配对后的真实帧，不表示已经证明前 100 ps 足够平衡。`--max-frames` 不设置时处理所选范围内所有帧。`--stride` 改变实际读取帧间隔，而 `--frame-dt-ps` 始终指原始文件相邻帧的时间间隔。

批量处理下载目录中的两条水轨迹：

```bash
.venv/bin/python batch.py \
  --input-dir data --output-root results/all_water \
  --frame-dt-ps 0.5 --block-frames 100 \
  --expensive-stride 10 --pair-stride 5
```

程序拒绝覆盖已有结果。每个状态点单独存储；失败会保留 `report.json`、日志和已有输出，便于定位。批量命令会实际运行全量后处理，需由使用者选择执行时机；本次交付不自动等同于已经完成所有帧的全量物性分析。

## 周期边界与 H 传递时的瞬时水分子识别

支持正交盒及 LAMMPS restricted triclinic 盒，盒向量按行存储。局部距离采用最短镜像，跨边界 OH 键先重建；截断必须小于保守的唯一镜像半径。三斜盒使用 ASE 的格点最小镜像，不能简单对笛卡尔坐标逐轴取余。

H 传递可以是研究体系中的正常过程。这里检查的不是“有没有发生反应”，而是初始 `mol` 标签能否继续代表当前水分子。高密度样本中第162帧后已有H离开原O或换到其他O附近，因此不能把初始两个H永远绑定到原O。默认逐帧识别：

```text
--topology-mode nearest --bond-cut 1.35
```

每个 H 仅按最短距离归属最近 O，超过 1.35 Å 则不归属。只有恰好两个合适 H 的 O 才有水分子朝向；其他中心标记 `water_valid=False`，OH 和依赖水朝向的特征为 NaN。所有原始 O/H 坐标、力、ID、当前归属及 O 配位仍保留，不强行把 H3O/OH/游离 H 拼成水。该方法是几何判据，不是电子成键或化学反应的最终判断；应比较不同 `--bond-cut`。

当一个O失去H后又获得另一个H，只要当前重新构成有效H2O，就立即参与后续帧的水分子角度统计；不因它曾经发生过H传递而永久排除。对于共享H、短暂OH/H3O等环境，保存原子几何和归属，但不强行定义标准水的角平分线/分子平面。共享质子的归属会依赖几何截断，需用不同`--bond-cut`评估，当前程序不声称有电子成键判据。

两套局部表示并行保存：`features`描述全O/H环境，`water_features`只使用属于当前有效H2O的邻居信息。水—水角度位于`orientation/`；`orientation_all_O/`保留以所有邻居O为候选的对照统计。每种角度仍会排除它实际需要而不存在的方向，不用零角度替代缺失朝向。H2O筛选得到的是“给定当前是水”的条件统计，不能冒充包含所有中间态的无条件分布。

### 两个 H 的等价性

同一H2O中的两个普通H属于同一物种，不设置不同元素类型或H1/H2专用嵌入。H1/H2只是在数组中存两根OH的槽位。角平分线使用两个OH方向的对称和；OH角度合并统计；H位点基展开求和；平面相关使用消除法向任意符号的表达，体坐标空间分布对H交换对应的两种参考系对称化。交换两个H槽位不应改变不变特征和物理统计。

真实原子ID仅在时间追踪中使用：判断一条O-H是否还是同一原子对，或水的两个H组成是否变化。水分子组成用无序H集合比较，不把槽位交换当成H传递。`molecular_frame`、`body_force`和`body_torque`是带参考系约定的坐标分量，部分轴在H交换下变号，它们不是可直接拼入不变向量的物理标量；预测模型若使用这些分量需要相应对称/等变处理。输出的176维不变特征与这些原始坐标分量分开保存。

`--topology-mode fixed` 严格使用原始 mol 分组，出现异常 OH 时失败并保留证据。OH 时间相关按真实 H 原子 ID 匹配，不把交换后的另一个 H 当成原来的键。

氢键几何：供体 O 到受体 O 距离 `< --hbond-cut`，供体 O 处 OH 与 OO 夹角 `<= --hbond-angle`。默认 3.5 Å、30°。当前氢键以有效 H2O 提供 OH 供体，受体为几何 O 位点；不是所有离子物种的专用氢键模型。

MSD 若有 xu 或 image flags 则使用原始无跳跃信息；否则在固定盒中由相邻存帧的最短镜像增量推断。无法排除两帧间移动超过半盒的未观测跳跃。可变盒且无 image 信息时，程序不输出这类动力学结果。氢键“连续”仅指所有已保存帧连续，不能恢复两帧间的短暂断裂。

时间相关另分三种含义：`oxygen_motion/c1,c2`跟踪同一O处的瞬时水朝向，允许换H；`orientation/persistent_H2O/`要求整个已保存时间窗口内O-H2组成未变，是条件化的分子转动统计；`orientation/OH_identity_matched/`按真实O/H原子ID匹配两个端点。后两者都不是完整反应动力学，也无法识别0.5 ps存帧间被漏掉的短暂事件。

## 输出

| 文件 | 内容 |
|---|---|
| `environment.h5` | 原始坐标和力、原始/当前原子 ID、H 归属、有效水掩码、逐中心 q/LSI/配位/近邻距离、邻居表、氢键、分子合力及以 O 为原点的力矩、局部特征、球谐系数、建议时间划分 |
| `distributions.h5` | RDF、距离—取向联合分布、体坐标 O/H 空间分布、三体角和 g3、氢键与网络、q 等联合分布、空间相关；保留 counts、bin、分块结果与误差 |
| `collective.h5` | 非零盒波矢结构因子、子体积粒子数及累积量、取向结构因子、特征均值/协方差/PCA/边缘分布、特征与诊断目标的关联 |
| `dynamics.h5` | MSD、位移分布、非高斯参数、取向/标量时间相关、H 身份匹配的 OH 相关、第一/第二壳层驻留、离散氢键生存、环境转移、交叉相关、块长扫描 |
| `direct_correlations.h5` | 由OO RDF经OZ得到的c2(r)、c2_hat(k)、平面投影、零模和分块诊断；未知mu时保留c1的未知偏移 |
| `report.json` | 输入 SHA256、参数、单位、帧数、拓扑异常、周期处理、86 项能力与实际范围、跳过原因 |

`--expensive-stride` 控制周期 Voronoi、Steinhardt q4/q6 和网络环统计；`--pair-stride` 控制 OO/OH/HH RDF。两者均相对于选中的帧编号计数，其他普通局部统计逐选中帧计算。空间方向、环和直方图范围均有限，并不宣称穷尽所有高阶量。

环采用每条无向边最短替代路径得到的去重环，最多 8 个节点；周期贯通环另外计数，不混作局部闭环，不宣称是全部 primitive rings。RDF 排除同一个当前几何归属簇的分子内项；游离 H 不被错误地互相排除。OO RDF 用 N(N-1)/V 的有限 N 理想参考，积分配位相应使用 (N-1)/V。

所有概率密度注明统计测度。直方图的概率在有限值及给定 bin 范围内归一化，需同时检查 `n_finite`、`n_in_range`；原始逐分子数据保留，可重新选范围。空间加权组中的 `g` 表示按配对数归一化的平均乘积，不是 RDF；标量的 connected 数据另存。高维互信息只提供低维 plug-in 探索估计，不当作精确熵。

## 与新板书中的特征场接口

默认 `nrad=4,lmax=3` 输出 176 维不变特征：

- 32 个径向加权的数量/分子相对朝向标量通道。
- 144 个 O、H 及 O–H 通道的球谐 power 不变量，包含邻居联合贡献和自项，不声称全为纯三体项。
- 等变球谐系数形状为 `[中心,2种位点,4径向基,16角向分量]`，另行保存；可用 `--no-equivariant` 关闭。

同一基的H2O专用版本另存为`water_features`及`water_equivariant_coefficients`，每套不变量均为176维；不是声称已经训练了一个352维模型。`water_orientation_weighted_means`保存7类角向矩对4个径向权重的条件平均，形状`[帧,中心,7,4]`；无有效水邻居时为NaN，并保留邻居计数。O通道及邻居计数按O到中心的距离，H位点球谐通道按H到中心的距离截断，H必须归属于当前有效水。

中心水无有效朝向时，176 维向量为 NaN；统计向量分布时排除这些中心并记录数量。环境中无有效朝向的邻居不会伪造方向贡献。原始原子坐标/等变系数保留了与这些不变向量不同的信息。

q、配位、氢键等诊断指标单独保存，不默认全部作为可微 MLIP 输入。局部几何表示不是平衡 density 的直接替代证明；特征边缘分布、协方差和二维 PCA 分布也不等于完整高维 rho(xi)。SPE、lambda、FEM/GEN 的具体方程尚未确认；本程序不伪装实现它们。

均值、PCA 和描述性统计默认使用所分析数据；若用于预测模型，应仅在训练集拟合这些变换。保存的 `proposed_split_role` 按整帧分组，默认60/20/20并留可配置时间间隔，不能保证独立；时间间隔必须结合相关时间调整，跨状态泛化应整条轨迹/状态点留出。

## 后处理信息和性质明细

下面列出程序实际保存的信息。默认长度为Å，时间为ps；力与力矩的单位必须由原LAMMPS输入确认。非有限值和未采样区间有明确含义，不应统一填零。

### 原始信息和瞬时身份

以下路径均位于`environment.h5`。

| 信息 | 数据字段 | 说明 |
|---|---|---|
| 原始帧、步数、时间、周期盒 | `source_frame`, `step`, `time_ps`, `cell` | 可回到原始dump定位，cell的盒向量按行存储 |
| 原子ID、类型、初始mol | `raw_atom_ids`, `raw_atom_types`, `original_mol_ids` | 原子身份不因H传递而重编号 |
| 原始原子坐标与力 | `raw_positions`, `raw_forces` | 原子按raw_atom_ids对应；缺力字段时不生成力数据 |
| O坐标、全部H坐标 | `o`, `raw_h_positions`, `raw_H_atom_ids` | 包括非H2O中心及未归属H |
| 当前H归属和O的H数 | `h_owner`, `assigned_h_count` | H归属为O中心索引；-1表示超过距离阈值未归属 |
| 当前水分子及构型 | `water_valid`, `current_atom_ids`, `oh`, `molecular_frame` | 逐帧更新；不适用水朝向的中心为NaN，不删除整帧 |
| 可用的其他原始字段 | `velocities`, `charges_native`, `native_unwrapped_o` | 只在输入实际提供时保存，不凭坐标补造 |
| 时间划分建议 | `proposed_split_role` | 整帧train/validation/test/间隔标记；相关性需自行核查 |

### 距离、配位和局部结构

| 性质 | 主要输出 | 统计对象及含义 |
|---|---|---|
| OO、OH、HH径向分布 | `distributions.h5/rdf/OO,OH,HH` | 全原子/O位置统计；OH/HH排除同一当前几何归属簇的分子内项 |
| RDF峰谷及积分配位 | `rdf/OO/diagnostics` | 有限分箱的峰谷诊断；积分配位与有限N归一化一致 |
| 第一、第二壳层配位 | `environment.h5/coordination,second_shell`及`local/`分布 | 当前按所有O近邻计数，不自动等于水邻居数 |
| 水邻居与其他O数 | `valid_water_neighbor_count`, `other_oxygen_neighbor_count` | env-cut内以O距离定义；用于解释水角度样本覆盖 |
| 第1–6近邻距离 | `nearest`及`distributions.h5/nearest/` | 按所有O排序；含r5-r4及r4/r5联合分布 |
| 四面体q | `q`, `q_first_shell_valid`, `local/q` | 四个最近O的角向几何；q可为负，四邻居未必全是H2O |
| 近邻径向离散程度 | `radial_cv`, `local/radial_cv` | 四个最近O距离的变异系数，非独立热力学量 |
| LSI | `lsi`, `local/lsi` | 规定lsi-cut下的邻居距离间隙方差 |
| Steinhardt q4/q6及邻居平均版本 | `environment.h5/expensive/`及`local/steinhardt_*` | 周期O近邻的有序指标，按expensive-stride采样 |
| Voronoi体积、面数、表面积及形状 | `environment.h5/voronoi/`、`distributions.h5/voronoi/` | 周期O点剖分；形状代理为A³/(36πV²)，按expensive-stride采样 |
| 分子内OH长度、HOH角 | `distributions.h5/intramolecular/` | 当前有效H2O，提供长度—角度联合分布 |
| 局部量联合分布 | `distributions.h5/joint/` | q—配位、q—r5、q—氢键数、q—Voronoi体积及两壳层占据等 |

### 水的朝向、角度与三体几何

主目录`distributions.h5/orientation/`要求中心与邻居当前均为有效H2O；`orientation_all_O/`是保留全部候选O的对照。水的角平分线是几何朝向，不是真实电偶极。

| 信息或性质 | 输出字段/组 | 含义 |
|---|---|---|
| 中心水朝向与连心线夹角 | `orientation/center/` | u_i与O_i指向O_j的方向 |
| 邻居水朝向与回指中心方向夹角 | `orientation/neighbor/` | u_j与O_j指向O_i的方向 |
| 两水朝向夹角 | `orientation/bisector_pair/` | u_i·u_j |
| 水分子平面相对角 | `orientation/plane_abs/` | 平面法向点积绝对值，处理相同H交换的等价性 |
| 平面相对于连心线的方向 | `orientation/plane_vs_r_abs/` | 平面法向与连心线点积绝对值 |
| 绕连心线的相对扭转 | `orientation/twist_abs/` | 投影法向间夹角的绝对余弦；不是唯一Euler角 |
| 具体OH朝向 | `orientation/OH_vs_r` | 两根OH与连心线的夹角，合并等价H的统计 |
| 距离—角度及多个角的联合分布 | `orientation/*`, `center_neighbor_joint`, `distance_center_neighbor` | 使用cos角测度，保留距离与取向耦合 |
| 壳层及氢键条件统计 | 各角度下`all,first,second,hbond,non_hbond` | 明确条件下的水—水样本，附原始计数 |
| 各向同性检查 | `isotropy/` | 实验室系OH、角平分线、平面方向分布 |
| 体坐标中的O/H空间分布 | `bodyframe/O_voxels,H_voxels` | 有效中心水的参考系，保留所有邻近位点；不是只含H2O邻居的图 |
| OOO角、距离—距离—夹角 | `triplet/OOO_cos,r_r_cos` | 全O三体位置统计及g3，不是c3 |
| 仅含水的OOO角 | `triplet/water_OOO_cos` | 中心及两个邻居都为当前有效H2O |
| 三体角与邻居联合朝向 | `triplet/angle_neighbor_orientation` | 三个当前有效水；all-O对照另存`triplet_all_O/` |
| 三体相对两体因子化的偏离 | `triplet/r_r_cos/difference_from_pair_product` | 与Kirkwood两体乘积比较；超出RDF范围的参考不外推 |

### 氢键和网络

| 性质 | 主要输出 | 说明 |
|---|---|---|
| 氢键三元组 | `environment.h5/neighbors/*/hbond_triplets` | 当前供体O、H槽位、受体O索引；实际H身份由current_atom_ids查回 |
| 未阈值化距离—方向角分布 | `distributions.h5/hbond/distance_angle` | 有效水提供OH，候选受体为所有O |
| 供体、受体和总氢键数 | `hbond_donors,hbond_acceptors`及`local/` | 几何定义和计数规则固定 |
| 供受体类别 | `joint/donor_acceptor` | 可提取D2A2等比例，包含无氢键类别 |
| 分叉氢键代理 | `hbond/acceptors_per_H` | 每根有效水OH匹配的受体数 |
| 网络度、连通分量、局部聚类 | `network/`及`environment.h5/expensive/` | 按expensive-stride采样 |
| 局部环和周期贯通环 | `network/contractible_shortest_ring_size,winding_shortest_ring_size` | 有限长度的最短替代路径环，不是全部primitive环 |
| 氢键链二面角 | `network/chain_dihedral_cos` | 局部四O链几何统计 |

### 集体结构、空间关联和局部环境分布

| 性质 | 主要输出 | 说明 |
|---|---|---|
| OO、OH、HH结构因子 | `collective.h5/structure_factors/` | 记录的非零盒波矢，含self/同分子贡献；非实验加权散射 |
| 取向纵向/横向结构因子 | `orientation_structure_longitudinal,transverse` | 有效水几何角平分线，不是电介质响应 |
| 子体积粒子数与空腔概率 | `window_counts,window_statistics,window_distributions` | 多种球半径；含方差、三/四阶累积量，固定N的有限尺寸限制保留 |
| q/配位/氢键数空间关联 | `distributions.h5/spatial/product_*` | 平均乘积和减去全局均值平方后的connected统计 |
| 取向空间关联 | `spatial/orient_P1,P2,longitudinal,transverse` | 按有效方向配对归一化 |
| OO位置两体熵近似 | `rdf/OO/diagnostics/OO_position_pair_entropy_over_kB` | 仅位置两体近似，不是水的完整熵 |
| 低维互信息 | 各二维分布的`plugin_mutual_information_nats`属性 | plug-in探索估计，有稀疏分箱偏差 |
| 全环境特征统计 | `collective.h5/feature_statistics/` | 176维均值、协方差、边缘分布、PCA与目标关联 |
| H2O专用特征统计 | `collective.h5/water_feature_statistics/` | 176维水邻居版本的均值、协方差、边缘分布和有效中心数 |

### 运动、时间关联和力

| 性质 | 主要输出 | 身份及采样限制 |
|---|---|---|
| O位移、MSD、非高斯参数 | `dynamics.h5/oxygen_motion/` | 同一O原子，不依赖H属于哪一个水 |
| 扩散系数 | `fit-diffusion`命令 | 显式指定线性时间窗；主程序不自动输出未经检查的D |
| 位移分布、自van Hove | `self_van_hove/` | 有限r范围内归一化，保留计数和边界 |
| 自/集体中间散射 | `intermediate_scattering/` | 固定盒中选定的有限波矢子集 |
| 同一O处的瞬时朝向C1/C2 | `oxygen_motion/c1,c2` | 允许H交换，解释为O中心局部取向记忆 |
| 未观测到换H的水转动C1/C2 | `orientation/persistent_H2O/` | 全部保存帧O-H2组成不变的条件样本，另存样本数 |
| 同一O-H身份的相关 | `orientation/OH_identity_matched/` | 匹配H原子ID，可跨H槽位变换；仅端点条件 |
| 分子平面C2 | `orientation/plane_normal/` | 偶数阶处理法向变号；仍是同O的瞬时有效水参考 |
| 净转角分布 | `lag_angle_distribution/` | 离散滞后时间；无法恢复帧间完整转动路径 |
| 壳层驻留和间歇相关 | `first_shell/,second_shell/` | 当前按O身份定义的壳层，不是限定不反应的水分子生存 |
| 氢键生存、间歇相关 | `hbond/` | 按供受体O对，非同一H三元组寿命；连续仅指保存帧 |
| q/LSI/配位/氢键数时间相关 | `scalar/` | 同一O中心观测量 |
| 供受体环境转移计数 | `environment_transitions/` | 多个滞后下的描述性计数，不声称Markov模型已成立 |
| 平移—转动、位移—氢键变化关联 | `cross_correlations/` | 对可用样本做一致掩码，不直接解释为因果 |
| 合力、以O为原点的力矩 | `environment.h5/molecule_force,molecule_torque_about_O` | 当前有效O-H2原子集合，H传递前后未必是同一物质分子 |
| 分子参考系的力与力矩分量 | `body_force,body_torque` | 参考系H标签约定另存，部分轴在H交换下变号 |
| 力/力矩分布及结构关联 | `distributions.h5/force,torque,joint/` | 与q、氢键数的联合统计；native单位未经核查不转换 |

### 采样和可追溯性

`dynamics.h5/stationarity/`比较前后半段均值；`sampling/`保存帧平均量的自相关、积分相关时间估计、有效帧数估计、块长扫描及block bootstrap区间。`distributions.h5`保留原始计数、分箱边界、分块计数与误差。`report.json`保存输入SHA256、参数、实际帧数、拓扑筛选数量、单位及86项能力状态。

“能计算”不代表采样已经充分。高温H传递不应被默认为MD错误；同时，水条件统计、全原子结构统计和反应中间态统计也不能混称为同一个分布。完整能力与尚需其他字段的项目见[性质清单](docs/water_MD_property_inventory_20261111.md)。

## 明确不输出的量

当前 dump 没有能量、温度、速度、应力、偶极或极化率字段。不用坐标捏造这些量，不输出真实介电常数、光谱、黏度、热导率、热容、绝对化学势或自由能。force 默认 `native_unknown`；确认原 LAMMPS `units` 后才设置 `--force-unit`，不凭 MACE-LES 名称假定 eV/Å。

c2的OO标量OZ反演现已实现，公式和原文依据见[c1/c2说明](docs/direct_correlations.md)。mu未知时，同温度密度扫描可输出带未知整体常数的相对c1，不伪造绝对值。c3/c4、额外MD状态点及外场计算仍不执行。g3与Kirkwood两体乘积的比较只是普通三体结构统计；所需r_jk超出实际RDF范围时结果为NaN，不外推尾部。

## 未知 mu 时直接提取体相 c1 和 c2

无需mu、无需重跑MD：

```bash
.venv/bin/python -m waterpost rdf-only \
  --input data/H2O-T986.4-r00-rho0.6700.dump.gz \
  --output results/rdf_r00 --rdf-max 12 --bins 240
```

每个状态输出OZ的c2及其诊断。多个同温度、同模型状态的相对c1用`c1-scan`积分；不默认从最低密度外推到零，也不默认将最低密度的绝对c1设为0。完整步骤、假设、PNAS与PRL公式页码及HDF5字段见[直接关联计算说明](docs/direct_correlations.md)。

本程序以全部O中心构造标量OZ关联，H传递时也保持O中心身份。该结果与瞬时H2O角度的条件统计用途不同，不宣称已恢复完整分子取向c2或完整反应性水的c1泛函。

## 检查、比较及扩散拟合

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=.test_tmp_user
.venv/bin/python -m waterpost compare \
  --results results/state_A results/state_B --output comparison.json
.venv/bin/python -m waterpost fit-diffusion \
  --results results/state_A --start-ps 20 --end-ps 100
```

扩散拟合窗口由使用者根据 MSD 曲线指定，上面的时间只是命令示例，不是本体系已验证的线性区。没有自动选择一个斜率冒充可靠扩散系数。

完整输入扫描可运行 `tools/audit_inputs.py`，核查每帧及原 mol 分组异常；它不启动动力学。完整清单见 [后处理性质清单](docs/water_MD_property_inventory_20261111.md)；各项实际能力应以本次运行 `report.json` 的覆盖状态为准。


## 验证记录与数据来源

[软件与周期几何验证摘要](validation/README.md)；本地曾对两条实际轨迹各24帧进行端到端验证，并完整审计各2001帧。该结果不代表全部帧的物性已经收敛。

Zenodo 记录号为 `24123911`。数据记录若仍是草稿，下载需记录所有者授权；本仓库不包含 Token 或轨迹。
