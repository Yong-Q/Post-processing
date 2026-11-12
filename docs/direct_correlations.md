# 从现有体相 MD 提取 c1 和 c2

本模块只处理已有轨迹，不运行额外MD，也不要求用户事先知道化学势。直接得到的是体相OO位置投影的c2；同温度密度扫描在声明单组分标量映射后得到c1差值。绝对c1的积分常数保留为未知，除非另有参考值。

## 原文依据

- Bui–Cox，PNAS 2026，DOI [10.1073/pnas.2610049123](https://doi.org/10.1073/pnas.2610049123)：第2页式(2)、(3)定义c1与平衡密度/外场/化学势的关系；第9页canonical MD训练把每个状态的mu作为潜变量。
- Dijkman等，PRL 2025，DOI [10.1103/PhysRevLett.134.056103](https://doi.org/10.1103/PhysRevLett.134.056103)：第6页式(A1)–(A5)给出OZ反演和平面投影；第7页式(A7)、(A8)明确二阶导数监督缺少一阶导数常数，另外使用bulk density及已知mu的offset loss。

所以“PRL使用体相数据”不等于“只凭一个状态的g(r)就唯一得到绝对c1”。本程序不把零密度条件或未采样区间的外推偷偷代入现有数据。

## c2 的公式和单位

均匀、各向同性的标量表示下，rho是O中心的数密度，单位Å^-3，不能把g cm^-3直接代入。

$$
h(r)=g_{OO}(r)-1,
\qquad
\widetilde h(k)=4\pi\int_0^\infty r^2h(r)\frac{\sin kr}{kr}\,dr.
$$

$$
S(k)=1+\rho\widetilde h(k),\qquad
\widetilde c_2(k)=\frac{\widetilde h(k)}{1+\rho\widetilde h(k)}.
$$

$$
c_2(r)=\frac1{2\pi^2}\int_0^\infty k^2\widetilde c_2(k)\frac{\sin kr}{kr}\,dk.
$$

c2(r)、g、h、S无量纲；hhat/chat的单位为Å^3。没有用HNC等闭合把g(r)转换成额外的化学势估计。

与PRL平面几何接口对应：

$$
\overline c_2(z)=2\pi\int_{|z|}^\infty r c_2(r)\,dr,
\qquad
\frac{\beta}{A}\frac{\delta^2F_{\rm ex}}{\delta\rho(z)\delta\rho(z')}
=-\overline c_2(|z-z'|).
$$

bar c2单位Å^2。程序输出连续泛函Hessian核，不将其直接冒充某个网络的离散Hessian；后者还需对应的面积及网格间距因子。

## 有限网格怎么实现

原始RDF必须使用从0开始的均匀径向分箱。r_j=(j+1/2)dr，R=N dr，k_m=(m+1)pi/R。用SciPy的未正交归一化DST-II/III实现：

$$
\widetilde h_m=\frac{2\pi\Delta r}{k_m}\operatorname{DST}_{II}[r_jh_j]_m,
\qquad
c_{2,j}=\frac{\operatorname{DST}_{III}[k_m\widetilde c_{2,m}]_j}{4\pi Rr_j}.
$$

该缩放在rho=0时精确离散往返h；Gaussian解析变换用于检查符号、4pi因子和k尺度。实际推断仍假设RDF以外h可以截断为0，并检查c2是否足够短程。自然DST网格没有直接测得零波矢。

零波矢另由截断的中点积分估计：

$$
\widetilde h(0)\simeq4\pi\Delta r\sum_j r_j^2h_j,
\qquad
\widetilde c_2(0)\simeq\frac{\widetilde h(0)}{1+\rho\widetilde h(0)}.
$$

同时保存由重建c2(r)积分得到的值及两者差异。有限R、有限k网格和长程尾部会使它们不同；不能选一个更好看的数掩盖不一致。固定N下全盒粒子数方差为0，不能把它直接当成S(0)。

默认不改尾部、不裁剪负S。非正或过小S会被标记，相关c_hat位置为NaN；若正k网格有无效值，c2(r)整体不生成有限曲线，保留诊断。可选`--oz-window cosine`只是在最后一段对h平滑衰减，属于额外数值处理，必须比较敏感性；不是论文要求的默认操作。

RDF使用现有N(N-1)/V理想参考，不宣称因此已经消除了canonical有限尺寸误差。若原生非零k密度模式可用，还比较直接S(k)和截断RDF变换的S(k)。这有助于识别尾部/有限N问题，但不自动修正零模。

## mu 未知时 c1 得到什么

PNAS的约定为：

$$
c_1(\mathbf r)=\ln\left[\frac{\Lambda^3}{\zeta}\rho(\mathbf r)\right]
+\beta V_{\rm ext}(\mathbf r)-\beta\mu,
\qquad c_{1,b}=-\beta\mu_{\rm ex}.
$$

体相Vext=0、mu未知时，单个状态只能给出上述组合，不能从ln rho单独分离c1。`c1/c1_plus_beta_mu`保存这个组合；没有依据时不创建绝对`c1/c1`数据集。reference_volume=Lambda^3/zeta默认1 Å^3，对应PNAS第9页的训练约定，不是由MD测得的分子配分函数。

若有同温度、同相互作用模型的单组分标量密度扫描，则：

$$
\frac{dc_{1,b}}{d\rho}=\widetilde c_2(0;\rho,T),
\qquad
c_{1,b}(\rho)-c_{1,b}(\rho_{\rm ref})
=\int_{\rho_{\rm ref}}^\rho\widetilde c_2(0;\rho',T)\,d\rho'.
$$

程序沿已有密度点做梯形积分，返回`delta_c1`，以某个已测状态为参考，**不把其绝对c1设为0**。密度点只有两个时只是粗糙两点积分，密度离散误差与零模系统误差不因积分而消失。提供`--anchor-c1`才会额外生成已锚定的c1曲线；mu并非唯一可能的锚定信息。

对存在H传递的水，g_OO是所有O中心的统计；它不是完整rho(r,Omega)的关联，更不是纯H2O条件筛选后的同一热力学对象。由其标量OZ构造的c2及由单组分和式积分的c1应标为有效O中心表示。CLI要求`--assume-single-component`显式声明此模型假设，不能将它直接提升为已验证的完整反应性水分子自由能。

## 直接使用当前轨迹

在仓库根目录运行；两个命令都只读已有dump，不需要mu：

```bash
.venv/bin/python -m waterpost rdf-only \
  --input data/H2O-T986.4-r00-rho0.6700.dump.gz \
  --output results/rdf_r00 --rdf-max 12 --bins 240 --block-frames 100

.venv/bin/python -m waterpost rdf-only \
  --input data/H2O-T986.4-r04-rho0.3811.dump.gz \
  --output results/rdf_r04 --rdf-max 12 --bins 240 --block-frames 100

.venv/bin/python -m waterpost c1-scan \
  --results results/rdf_r04 results/rdf_r00 \
  --output results/c1_relative --model-id YOUR_SAME_MODEL_ID \
  --temperature-K 986.4 --assume-single-component
```

`rdf-only`只做OO RDF、少量原生结构因子和直接关联，避免重复全部局部特征。完整`analyze`现在也默认附加直接关联输出；用`--no-direct-correlations`可关闭。已有后处理结果可直接使用：

```bash
.venv/bin/python -m waterpost correlations \
  --results results/existing_state --output results/correlations_state
```

所有输出路径均要求不存在，原始MD和已有结果不覆盖。已有RDF变换和c1积分不需要再下载轨迹或跑MD。若以后另有beta mu或beta mu_ex，可用可选参数固定偏移，但当前未知mu时不用填写。

## 输出字段

| 文件与路径 | 内容 |
|---|---|
| distributions.h5/rdf/OO | 全O的g(r)、分箱、计数、参考曝光和block g |
| direct_correlations.h5/c2/r,c2_r | 径向c2及其半格点坐标 |
| c2/k,h_hat,S_k,c2_hat,valid_k | 非零k谱、OZ结果及有效掩码 |
| c2/h_hat_zero,S_zero_estimate,c2_hat_zero | 由截断RDF估计的零模，不是固定N全盒涨落 |
| c2/c2_real_integral_midpoint,zero_mode_discretization_difference | 零模一致性诊断 |
| c2/z,c2_planar,beta_over_area_functional_hessian | PRL平面投影及符号约定 |
| c2/raw_tail_mean_g,raw_tail_max_abs_h | 未平滑原RDF的尾部检查 |
| blocks/ | 各block的c2及有效block数、SEM；不是误差已收敛的证明 |
| nonzero_k_crosscheck/ | 直接密度模式S(k)与RDF变换对照（有对应输入时） |
| c1/ | 未知mu时保存c1+beta mu及状态；有依据时才保存绝对c1 |
| bulk_c1.h5/delta_c1 | 同温度密度扫描上的相对体相c1 |
| bulk_c1.h5/rho_A_minus3,c2_hat_zero | 用于积分的状态与导数估计，便于审查 |

这一步是从轨迹提取训练/检验信息，没有训练神经泛函。后续若拟合F_ex或c1，仍需保证偏移、热力学状态、理想项约定和密度表示一致。
