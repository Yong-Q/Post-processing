# 每帧总势能（MACE-LES）

对密度扫描的 46 个已完成状态，用**与生产 MD 完全相同的势函数**（`pair_style symmetrix/mace/float32/kk`
+ `maceles_off_chno.json`）对**每一帧坐标**做单点能量计算。用于同一势函数下、变温 + 已知位置外势的
**重加权**：重加权只需要每帧的整盒势能 $U(X_i)$。

## 数据

| 项 | 值 |
|---|---|
| 状态数 | 46（CO₂ 9、H₂O 9、C₂H₄ 8、CH₄ 8、NH₃ 8、C₂H₆ 4） |
| 每状态帧数 | 2001（步长 0, 1000, …, 2 000 000，即 1 ns，0.5 ps/帧） |
| 文件 | `<sim_id>.energy.csv` × 46 |
| 汇总 | `summary.csv` |

## 格式与单位

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
thermo_style custom step atoms pe
rerun <traj.dump> dump x y z
```

`rerun` 顺序读 dump，每个快照做一次力/能量评估，不积分。设置与生产 run 完全一致，
所以 `pe` 与 MD 运行时打印的 `PotEng` 是同一个量。

脚本：`scripts/energy_eval_one.sh`（单状态）、`scripts/energy_eval.sbatch`（24 进程批处理）、
`scripts/states.txt`（46 个状态清单）。

## 校验

| 检查 | 结果 |
|---|---|
| 全部 46 个文件 | 2001 行，步数 0–2 000 000，间距 1000，无重复/缺帧 |
| 与 MD 的 restart 交叉核对 | 1 ns 构型 PE = −1 388 147.3 eV（轨迹末帧一致，差 < 1 eV） |
| 与 PyTorch 参考对照 | 同一构型下能量偏差 **2e-4 ~ 6e-3 meV/atom**（见 1112 仓库 `validation/`） |
| 能量漂移 | 首尾 1/5 段均值差 0.1–5 eV（相对总能量 1e-6），无漂移 |

## 注意

- 能量是**模型给出的绝对值**（含每元素参考能），不是内聚能；做重加权时只用**差值**，
  常数项会自动抵消。
- float32 引擎，单帧能量重复性约 1e-3 eV；跨构型比较时不要把这个量级当物理信号。
- 这些能量对应**已经保存的 46 条轨迹**；ρ\*=0.02 的 6 个状态没有跑，所以没有对应文件。
