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

## 周期边界与高温水的拓扑

支持正交盒及 LAMMPS restricted triclinic 盒，盒向量按行存储。局部距离采用最短镜像，跨边界 OH 键先重建；截断必须小于保守的唯一镜像半径。三斜盒使用 ASE 的格点最小镜像，不能简单对笛卡尔坐标逐轴取余。

高密度样本原始 `mol` 分组在第 162 帧已出现超长 OH；不能继续不加检查地当作固定水分子。默认：

```text
--topology-mode nearest --bond-cut 1.35
```

每个 H 仅按最短距离归属最近 O，超过 1.35 Å 则不归属。只有恰好两个合适 H 的 O 才有水分子朝向；其他中心标记 `water_valid=False`，OH 和依赖水朝向的特征为 NaN。所有原始 O/H 坐标、力、ID、当前归属及 O 配位仍保留，不强行把 H3O/OH/游离 H 拼成水。该方法是几何判据，不是电子成键或化学反应的最终判断；应比较不同 `--bond-cut`。

`--topology-mode fixed` 严格使用原始 mol 分组，出现异常 OH 时失败并保留证据。OH 时间相关按真实 H 原子 ID 匹配，不把交换后的另一个 H 当成原来的键。

氢键几何：供体 O 到受体 O 距离 `< --hbond-cut`，供体 O 处 OH 与 OO 夹角 `<= --hbond-angle`。默认 3.5 Å、30°。当前氢键以有效 H2O 提供 OH 供体，受体为几何 O 位点；不是所有离子物种的专用氢键模型。

MSD 若有 xu 或 image flags 则使用原始无跳跃信息；否则在固定盒中由相邻存帧的最短镜像增量推断。无法排除两帧间移动超过半盒的未观测跳跃。可变盒且无 image 信息时，程序不输出这类动力学结果。氢键“连续”仅指所有已保存帧连续，不能恢复两帧间的短暂断裂。

## 输出

| 文件 | 内容 |
|---|---|
| `environment.h5` | 原始坐标和力、原始/当前原子 ID、H 归属、有效水掩码、逐中心 q/LSI/配位/近邻距离、邻居表、氢键、分子合力及以 O 为原点的力矩、局部特征、球谐系数、建议时间划分 |
| `distributions.h5` | RDF、距离—取向联合分布、体坐标 O/H 空间分布、三体角和 g3、氢键与网络、q 等联合分布、空间相关；保留 counts、bin、分块结果与误差 |
| `collective.h5` | 非零盒波矢结构因子、子体积粒子数及累积量、取向结构因子、特征均值/协方差/PCA/边缘分布、特征与诊断目标的关联 |
| `dynamics.h5` | MSD、位移分布、非高斯参数、取向/标量时间相关、H 身份匹配的 OH 相关、第一/第二壳层驻留、离散氢键生存、环境转移、交叉相关、块长扫描 |
| `report.json` | 输入 SHA256、参数、单位、帧数、拓扑异常、周期处理、86 项能力与实际范围、跳过原因 |

`--expensive-stride` 控制周期 Voronoi、Steinhardt q4/q6 和网络环统计；`--pair-stride` 控制 OO/OH/HH RDF。两者均相对于选中的帧编号计数，其他普通局部统计逐选中帧计算。空间方向、环和直方图范围均有限，并不宣称穷尽所有高阶量。

环采用每条无向边最短替代路径得到的去重环，最多 8 个节点；周期贯通环另外计数，不混作局部闭环，不宣称是全部 primitive rings。RDF 排除同一个当前几何归属簇的分子内项；游离 H 不被错误地互相排除。OO RDF 用 N(N-1)/V 的有限 N 理想参考，积分配位相应使用 (N-1)/V。

所有概率密度注明统计测度。直方图的概率在有限值及给定 bin 范围内归一化，需同时检查 `n_finite`、`n_in_range`；原始逐分子数据保留，可重新选范围。空间加权组中的 `g` 表示按配对数归一化的平均乘积，不是 RDF；标量的 connected 数据另存。高维互信息只提供低维 plug-in 探索估计，不当作精确熵。

## 与新板书中的特征场接口

默认 `nrad=4,lmax=3` 输出 176 维不变特征：

- 32 个径向加权的数量/分子相对朝向标量通道。
- 144 个 O、H 及 O–H 通道的球谐 power 不变量，包含邻居联合贡献和自项，不声称全为纯三体项。
- 等变球谐系数形状为 `[中心,2种位点,4径向基,16角向分量]`，另行保存；可用 `--no-equivariant` 关闭。

中心水无有效朝向时，176 维向量为 NaN；统计向量分布时排除这些中心并记录数量。环境中无有效朝向的邻居不会伪造方向贡献。原始原子坐标/等变系数保留了与这些不变向量不同的信息。

q、配位、氢键等诊断指标单独保存，不默认全部作为可微 MLIP 输入。局部几何表示不是平衡 density 的直接替代证明；特征边缘分布、协方差和二维 PCA 分布也不等于完整高维 rho(xi)。SPE、lambda、FEM/GEN 的具体方程尚未确认；本程序不伪装实现它们。

均值、PCA 和描述性统计默认使用所分析数据；若用于预测模型，应仅在训练集拟合这些变换。保存的 `proposed_split_role` 按整帧分组，默认60/20/20并留可配置时间间隔，不能保证独立；时间间隔必须结合相关时间调整，跨状态泛化应整条轨迹/状态点留出。

## 明确不输出的量

当前 dump 没有能量、温度、速度、应力、偶极或极化率字段。不用坐标捏造这些量，不输出真实介电常数、光谱、黏度、热导率、热容、绝对化学势或自由能。force 默认 `native_unknown`；确认原 LAMMPS `units` 后才设置 `--force-unit`，不凭 MACE-LES 名称假定 eV/Å。

c2 的 OZ 反演、c3/c4、额外状态点及外场计算目前均不执行。g3 与 Kirkwood 两体乘积的比较只是普通三体结构统计；所需 r_jk 超出实际 RDF 范围时结果为 NaN，不外推尾部。

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
