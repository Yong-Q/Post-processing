# 每帧总势能（MACE-LES）

对密度扫描的 46 个已完成状态，用**与生产 MD 完全相同的势函数**（`pair_style symmetrix/mace/float32/kk`
+ `maceles_off_chno.json`）对**每一帧坐标**做单点能量计算。用于同一势函数下、变温 + 已知位置外势的
**重加权**：重加权只需要每帧的整盒势能 $U(X_i)$。

## 数据

| 项 | 值 |
|---|---|
| 状态数 | 46（CO₂ 9、H₂O 9、C₂H₄ 8、CH₄ 8、NH₃ 8、C₂H₆ 4） |
| 每状态帧数 | 2001（步长 0, 1000, …, 2 000 000，即 1 ns，0.5 ps/帧） |
| CSV | `<sim_id>.energy.csv` × 46（带表头，`%.16g`） |
| npz | `npz/<sim_id>.npz` × 46 + `npz/all_states.npz`（float64） |
| 汇总 | `summary.csv`（含 ESS、权重集中度、自相关时间等诊断列） |

## 格式与单位

每个 CSV **带表头** `step,pe_eV,press_bar`，2001 行数据对应步数 0, 1000, …, 2 000 000。
同一批数据另有 float64 的 npz（`npz/<sim_id>.npz`，以及合并的 `npz/all_states.npz`），
字段 `step`(int64)、`pe`(float64)、`press`(float64)、`meta`(JSON)。

```
step,pe_eV,press_bar
0,-3425165.3,498.86272
1000,-3425167.1,511.22406
...
2000000,-3425161.2,-1206.8952
```

- **`pe_eV` 是整盒总势能**（extensive，eV），**不含动能**，**不是每原子平均值**
- `press_bar` 是同一构型的**维里部分**（bar），**不含动能项**——dump 里没有速度，`rerun` 不积分，
  所以 LAMMPS 打印的 `press` = 维里/(3V)。物理压力应为
  $P = P_{\rm virial} + Nk_BT/V$（例如水 ρ\*=0.67：−2855 bar + 9300 bar ≈ **0.64 GPa**）
- `step` 是原始 LAMMPS 步数（`reset_timestep 0` 之后的生产段）

## 计算方法

```bash
lmp -k on t 1 -in in.energy
```

```lammps
units metal
atom_style molecular/kk
atom_modify map yes
boundary p p p
read_data <该状态的初始构型>
pair_style symmetrix/mace/float32/kk streamed_edges direct profile capacity \
           jit_host_artifact <host_artifact.so>
pair_coeff * * <maceles_off_chno.json> C H N O
neighbor 2.0 bin
neigh_modify every 10 delay 0 check yes
run_style verlet/kk
thermo 1
thermo_style custom step atoms pe press
thermo_modify format float %.16g
rerun <traj.dump> dump x y z
```

`rerun` 顺序读 dump，每个快照做一次力/能量评估，不积分。设置与生产 run 完全一致，
所以 `pe` 与 MD 运行时打印的 `PotEng` 是同一个量。

脚本：`scripts/energy_eval_one.sh`（单状态）、`scripts/energy_eval.sbatch`（24 进程批处理）、
`scripts/states.txt`（46 个状态清单）。

## 温度重加权的能量精度要求

**不需要重新采样或重跑 MD。** 继续使用现有 46 条轨迹；如需提高能量输出精度，
用相同生产势函数重新评估已有帧即可。如果还保留了未舍入的原始高精度能量，直接重新导出。
当前 CSV/NPZ 已由生产者重新导出；后续小微扰流程复用这些能量，不再补单点。

旧的 0.1 eV 版本已被 `%.16g`（16 位有效数字）版本取代（git 历史可查）。
下面的不同采样能量最小间隔不是引擎数值精度；float64 保存也不能提高 float32 计算精度。
对于固定 N、V、相同势函数的温度重加权，

$$
\ln q_i=(\beta_0-\beta)U_i,\qquad
|\delta\ln q_i|=|\beta_0-\beta|\,|\delta U_i|,
\qquad \beta=(k_BT)^{-1}.
$$

这里 $q_i$ 是未归一化权重，$U_i$ 是整盒势能，$k_B=8.617333262145\times10^{-5}$ eV/K。
若希望单帧未归一化权重的相对误差不超过 $\epsilon$，可采用保守要求

$$
|\delta U_i|\leq\frac{\ln(1+\epsilon)}{|\beta_0-\beta|}.
$$

该界限针对单帧权重，并不直接等于归一化后的性质误差。
例如能量按 0.1 eV 四舍五入时，舍入误差上限为 0.05 eV；目标温度降低 5% 时，
C₂H₄（430.4 K）和 H₂O（986.4 K）的单帧未归一化权重最大相对影响分别约为 **7.4%** 和 **3.1%**。
总能量中的元素参考能常数会抵消，不能用误差占绝对总能量的比例判断重加权精度。

建议：

- 在 `rerun` 前设置 `thermo_modify format float %.16g`，避免文本输出造成额外舍入。
- 整盒势能的实际数值误差尽量达到 **0.001 eV 左右**，具体容许值按上式和目标温差确定；
  这是整盒要求，不是每原子要求。输出位数不代表 float32 引擎具有同等计算精度。
- 在代表性构型上比较重复计算和可信参考实现的能量差；比较跨构型的误差变化，
  同一状态内与构型无关的常数偏移不影响归一化权重。现有低精度 CSV 不能用于确认毫电子伏精度。
- 用新输出目录保留旧 CSV（本次产物在 `energy_out_hp/`）。脚本会跳过已有至少 2002 行（2001 数据 + 表头）的文件。

单状态重新评估示例（在原计算集群、具有生产输入和模型文件的环境执行）：

```bash
ENERGY_OUT=/publicfs10/fs10-m9/home/m9s005258/qy/cdft/energy_out_high_precision \
  bash energy_scan/scripts/energy_eval_one.sh H2O-T986.4-r00-rho0.6700
```

批处理时同样向作业导出 `ENERGY_OUT`，并确认作业实际调用的是更新后的单状态脚本。
新文件需再次核对 2001 帧及逐帧 step 对齐，并重新生成汇总；旧 `summary.csv` 不会自动更新。
能量精度通过后，还应检查权重集中程度、有效样本数和时间相关性。
提高能量精度不能补足轨迹未覆盖的构型；同温度、相同势函数、仅添加已知位置外势时，
原势能在权重比中抵消，无需为该项提高原势能精度。

## 校验

| 检查 | 结果 |
|---|---|
| 全部 46 个文件 | 2001 行 + 表头，步数严格 0–2 000 000、间距 1000，无重复/缺帧/非有限值 |
| 能量分辨率（相邻不同值最小间隔） | 中位 **2.2e-6 eV**，最大 1.1e-5 eV（旧版为 0.1 eV） |
| 与旧 0.1 eV 版本逐帧对照 | 最大差 **0.0502 eV** = 0.1 eV 量化的理论舍入上限，物理量一致 |
| 与 MD restart 交叉核对 | H2O r00 末帧 −1 388 147.3386 eV，restart 单点 −1 388 147.3 eV（差 0.04，源于单点输出精度） |
| 首尾 1/5 段能量均值差 | 0.9–18.7 eV；需与分块误差比较，不能除以含巨大元素参考能的总能量来认定平衡 |
| npz 与 CSV | 逐元素差 **0** |

## 温度重加权诊断（5% 降温）

权重 $w_i\propto\exp[(\beta_0-\beta)(U_i-U_{\min})]$，$N=2001$ 帧/状态：

| 物种 | ESS/N @1% | ESS/N @2% | ESS/N @5% | 最大单帧权重（5%） |
|---|---|---|---|---|
| H2O | 0.82 | 0.46 | 0.033 | 0.14 |
| CO2 | 0.72 | 0.33 | 0.021 | 0.39 |
| NH3 | 0.80 | 0.40 | 0.023 | 0.52 |
| CH4 | 0.32 | 0.10 | 0.011 | 0.35 |
| C2H4 | 0.28 | 0.05 | 0.003 | 0.67 |
| C2H6 | 0.58 | 0.12 | 0.002 | 0.79 |

上述表格仅是帧权重集中度诊断，**不能据此认定 1% 温差已经可用**。
5% 时 ESS 只剩 4–66 帧，权重明显集中；即使更小温差也应检查时间相关、来源平衡性和各性质的误差。
本仓库的[范围扫描与小微扰流程](../docs/reweighting.md)对每条轨迹分别确定通过诊断的幅度，
不采用统一的 1% 或 2% 允许范围。不同密度盒不能未经处理直接作为 MBAR 的可交换构型。

## 注意

- 能量是**模型给出的绝对值**（含每元素参考能），不是内聚能；做重加权时只用**差值**，
  常数项会自动抵消。
- float32 引擎，单帧能量重复性约 1e-3 eV；跨构型比较时不要把这个量级当物理信号。
- 这些能量对应**已经保存的 46 条轨迹**；ρ\*=0.02 的 6 个状态没有跑，所以没有对应文件。
