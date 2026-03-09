# HW1 进度记录（中文）

## 当前状态

- 当前工作主线是 Triton 路线：`glm_asr_triton_template`
- 端到端 benchmark 已经跑通，准确率保持 `100.0%`
- 当前重点不再是修 benchmark 环境，而是优化 decoder decode 路径

## 已完成事项

### 1. Cluster / benchmark 流程确认

- 已确认测试必须在 compute node 上进行，head node 只用于 `git pull/push` 和轻量操作
- 在 compute node 上测试前，使用：

```bash
source /opt/conda/bin/activate mls
```

- 进入 compute node 的命令：

```bash
srun -p Teaching --gres=gpu:1 --mem=16G --pty bash
```

- 实测发现 `16G` 在当前环境下会在权重加载阶段 OOM，因此后续 profiling/benchmark 使用了更大的内存配额完成验证

### 2. benchmark harness 问题定位与修复

- 已定位到 `benchmark_student.py` 的关键问题：
  - 它是直接从模板目录导入模块
  - 因此包级 `__init__.py` 中的默认运行配置没有真正生效
- 修复前，`glm_asr_triton_template` 实际跑的是错误配置：
  - `Linear.BACKEND=torch`
  - `MLP.FUSED=True`
  - `EncoderMLP.FUSED=True`
- 这会让模板版在同一节点上表现异常慢

### 3. benchmark / profiling 脚本改动

- 已修改 `benchmark_student.py`
  - 显式应用统一的 Triton runtime config
  - 支持通过环境变量覆盖
- 已修改 `benchmark_detailed.py`
  - 显式应用统一的 Triton runtime config
  - 增加 warmup，避免首轮 CUDA/cuBLAS 初始化污染计时
  - 增加 decode profiling 细分：区分 first decode step 和 steady-state decode step

## 当前关键结果

### 最新优化：`top_k=1` 快路径

- 已在 `glm_asr_triton_template/model.py` 中加入 `top_k == 1` 的快路径
- 原先即使 `top_k=1`，每一步仍然会对整个词表执行 `argsort` 和采样流程
- 现在在 `top_k=1` 时直接走 `argmax`，避免不必要的整词表排序

### 最新优化：首个 decode step 切换到 `past_key_values` 缓存路径

- 已把 `generate()` 改成：
  - 对完整 prompt / audio context 先做一次 `use_cache=True` 的 prefill
  - 后续每一步只对新 token embedding 做 decode
  - 通过 `past_key_values` 复用之前的 KV cache
- 这比“每一步都把整段 `inputs_embeds` 重新送进 decoder”更合理，也直接针对了首个 decode step 之后的生成热路径

实测：

```bash
python benchmark_student.py glm_asr_triton_template --warmup 0 --runs 1
```

得到：

- `Time: 6378.0 ms`
- `Tokens: 13`
- `Accuracy: 100.0%`
- `Status: PASS`

相对上一版 `6785.1 ms`，再次提升了约 `407.1 ms`，约 `6.0%`

最新实测：

```bash
python benchmark_student.py glm_asr_triton_template --warmup 0 --runs 1
```

得到：

- `Time: 6785.1 ms`
- `Tokens: 13`
- `Accuracy: 100.0%`
- `Status: PASS`

相对修复 benchmark 配置后的 `7437.8 ms`，再次提升了约 `652.7 ms`，约 `8.8%`

### 修复 benchmark 配置后的端到端结果

在 compute node 上运行：

```bash
python benchmark_student.py glm_asr_triton_template --warmup 0 --runs 1
```

得到：

- `Time: 7437.8 ms`
- `Tokens: 13`
- `Accuracy: 100.0%`
- `Status: PASS`

### 修复 profiling 后的 template 阶段分析

在 compute node 上运行：

```bash
python benchmark_detailed.py glm_asr_triton_template --runs 2 --seq-len 128
```

得到：

- Audio Encoder: `345.10 ms`
- Multi-modal Projector: `1.86 ms`
- Decoder Prefill: `77.66 ms`
- Decode Step (avg): `84.39 ms`
- Decode Step (steady-state): `66.90 ms`
- First decode step: `241.85 ms`

估算总时间：

- Total (estimated for 50 tokens): `4644.18 ms`

### 当前结论

- 当前最主要的性能热点是 decoder decode 阶段
- decode 的高方差主要来自第一个 decode step
- steady-state decode 已经明显更低，当前更值得追的是首步 setup 成本，而不是把所有 decode step 视为同一种慢

## 代码差异中值得继续检查的点

- `glm_asr_triton_template/layers.py` 中，多处把转置缓存从：

```python
.t().contiguous()
```

改成了：

```python
.t()
```

- 这在未来启用 Triton / fused 路径时可能影响性能
- 但在当前 `cublas + unfused` 配置下，它还不是已确认的主瓶颈

## 下一步

- 继续检查 decoder 的 first decode step 特殊路径
- 重点看：
  - KV cache 初始化
  - 首步 attention 调度
  - 首步张量重排 / buffer 准备
- 这次已经验证过一次 `generate()` 切到 pre-allocated buffer decode 的实现尝试，但当前版本是负收益，已撤回，不作为当前基线
- 在确认首步 setup 成本后，再决定是否：
  - 优化 attention cache 路径
  - 恢复部分 `.contiguous()` 转置缓存
  - 重新评估 fused/Triton 路径在当前 GPU 上是否值得启用
