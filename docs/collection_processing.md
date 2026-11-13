# 整批轨迹后处理

`tools/run_collection.py`枚举目录中全部六物种`*.dump.gz`，忽略未完成`.part`。当前46条输入按物种/温度独立处理，不要求每个物种拥有相同数量的密度点。

处理顺序：全帧中心RDF/OZ → 每条全帧通用局部特征与时间统计 → 同物种/同温度/同中心定义下的相对c1。输入位置、ID、当前分子归属、中心、逐帧构型/角度/508维特征均保存；没有补势能或执行变温/外场重加权，不把平均结果冒充逐帧输入。

全部轨迹共享8.5 Å编码截断、nrad=3、lmax=3。诊断窗口优先取本状态中心RDF第一个可分辨峰后的极小值；低密度曲线未分辨出极小值时，显式使用相同物种/温度的参考状态窗口，并在`analysis_windows.json`记录回退原因，不宣称该值是本状态的实测壳层边界。

RDF范围12 Å/240 bins，全选中帧计算。局部特征、普通角度和构型逐帧计算；Voronoi/packing q4/q6/适用网络每10帧，主分析附带RDF每10帧，最终关联标签优先使用独立全帧RDF目录。时间滞后到100帧，邻居/HB生存到20帧，原始保存间隔0.5 ps；采样与隐藏事件限制保持不变。

每个状态独立保存结果/日志/参数指纹。已完成且参数一致时复用；失败或中断输出保留，在`--retry-failed`下创建新attempt目录，不覆盖原数据。`progress.json`原子更新整批状态，`collection_summary.json`区分完成、失败、跳过。c1仅使用可用正k谱和零模的状态，排除状态另列；未知绝对偏移保持未知。

## Slurm

示例（先建立日志目录）：

```bash
mkdir -p runs/logs
export POSTPROCESS_INPUT_DIR=/absolute/path/to/trajectories
export POSTPROCESS_OUTPUT_BASE=/absolute/path/to/results
export POSTPROCESS_PYTHON=/absolute/path/to/environment/bin/python
export POSTPROCESS_CODE_ROOT=/absolute/path/to/Post-processing
sbatch --chdir="$POSTPROCESS_CODE_ROOT" \
  --output=/absolute/path/to/runs/logs/post_%j.stdout \
  --error=/absolute/path/to/runs/logs/post_%j.stderr \
  tools/postprocess_collection.slurm
```

脚本申请单节点48 CPU、128G，worker数不超过Slurm分配和实际轨迹数。所有worker单线程；在计算分配中检查glibc和关键依赖。数据直接写共享结果目录，没有节点临时scratch或新MD。为可追溯，正式运行建议使用不可变的代码快照及独立输出根目录。

轻量准备/测试允许在登录机进行，完整批量处理应使用计算分配。状态检查使用`squeue`/`sacct`及结果目录的进度文件。程序不会将后台提交误报为所有结果已完成，也不把仅通过软件测试当物性收敛。
