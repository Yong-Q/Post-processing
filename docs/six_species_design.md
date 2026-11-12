# 六种纯分子体系的通用后处理设计

目标：在原有水程序之外增加H2O、NH3、CO2、CH4、C2H4、C2H6共同可用的分子中心、构型、对称特征、空间/时间统计、力和RDF/OZ入口。各取真实轨迹运行验证，不新增MD，不跨分子/温度积分c1。保持既有水入口兼容。

实现选择：共用LAMMPS读取、PBC、直方图、DST/OZ算法；新增分子定义与通用表示，避免将OH角和水四面体的意义套给其他分子。分子中心默认重原子骨架质心：H2O取O、NH3取N、CH4取C、C2物种取两C中点；CO2取C与两个O的质量质心（仅在线性对称极限与C重合）。完整质量质心另存；质量按元素组成计算，不固定水质量。

全局type默认1:C、2:H、3:N、4:O，必须通过实际原子数和组成验证，可显式配置。初始mol只用于固定重原子骨架；H按每帧到最近合适重原子的距离动态归属。分别报告组成及重原子骨架有效性，不能强行把解离/转移中间态当作目标分子。CO2无H，需正确处理空H集合。

## 分子识别接口

`waterpost/molecules.py`公开：

- `SPECS`：六种纯分子定义，包含composition、molar_mass、orientation_kind。
- `MolecularTopology(frame, species, type_map=None, mode='nearest', bond_cut=1.35)`。
- `extract(frame)`返回字典：`centers[N,3]`、`com[N,3]`、`molecule_valid[N]`、`heavy_valid[N]`、`owner[Natoms]`（未归属H=-1）、`positions[Natoms,3]`、`elements[Natoms]`（C=0,H=1,N=2,O=3）、`relative_positions[Natoms,3]`、`atom_ids[Natoms]`、`molecule_ids[N]`、`members`（每中心当前原子索引列表）、`hydrogen_count[N]`、`axis[N,3]`、`axis_kind`、`plane_normal[N,3]`、`geometry_samples`（key->一维/二维数值样本，单位在key或说明中）、`forces`/`molecular_force`/`molecular_torque`（若输入提供）。
- 属性：`species`、`spec`、`ids`、`initial_owner`、`element_codes`、`molecule_ids`。`spec`至少含`composition`、`molar_mass`、`orientation_kind`（字典字段）。
- positions与centers使用相同盒原点约定；centers在主周期盒内。完整分子重建通过最短镜像，重原子骨架重建先选同分子重原子为锚点。强制验证组成、恒定原子ID/类型、可解释的骨架和H成键距离。

H2O有极性几何轴和无向平面；NH3有C3对称极性轴但无任意H定义的方位角；CO2无向轴；CH4无唯一极性/无向轴；C2H4有C-C无向轴和无向平面；C2H6有C-C无向轴。CH4采用H集合的球谐三阶形状相关追踪取向。C2H6输出所有等价H组合的H-C-C-H扭角，而非选择固定H1。

## 通用表示接口

`waterpost/molecular_features.py`公开：

`encode_molecules(centers, positions, elements, owner, cell, valid=None, cutoff=6., nrad=3, lmax=3, neighbor_valid_only=False)`。

返回：`features[N,L]`、`names`、`coefficients[N,4,nrad,(lmax+1)^2]`、`shape_moments[N,4,(lmax+1)^2]`、`shape_strength[N,4,lmax+1]`、径向基参数、邻居覆盖计数（可选）。

不对等价原子赋不同嵌入。环境按C/H/N/O位点展开；排除中心自身所属原子，保留未归属H为环境信息。`neighbor_valid_only=True`则只保留属于当前有效目标分子的环境位点。内部球谐矩由当前分子全部位点构造；特征包含环境power、内部形状power，以及内部形状与环境的旋转标量耦合，不能只编码邻居而丢掉中心自身朝向。按通道种类和l说明变换规律；无效中心的`features`为NaN。标量不变量对整体旋转、平移、周期镜像、同类原子及分子排列不变。

特征阶数与多体信息不作无限完备性保证。保存原始环境和内部几何，以便未来增加展开阶数。通用版不制造唯一Euler角，不声称有限特征能无损逆构型。

## 主流程与验证

新增molecular CLI（或analyze --species路由）调用通用pipeline，输出通用HDF5与CSV/JSON能力表，保留水专用入口。RDF/OZ使用每个重原子骨架一个中心，C2物种不得把两个C当成两分子；相对c1要求同物种、同中心定义、同温度、同模型及共同网格。

验证覆盖：六类构型的手算中心和质量、任意周期镜像/三斜盒、C/H等价标签和分子排列、CH4无唯一轴、CO2无极性P1、C2跨盒中点、H转移掩码、实轨迹读取与全帧RDF、各物种通用特征/时间统计。独立审查关注配置和接口的物理语义；输入、样本计算和系统物理确认分开记录。
