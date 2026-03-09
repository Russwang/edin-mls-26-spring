# HW1 Progress Log (English)

## Current Status

- The active track is the Triton path: `glm_asr_triton_template`
- End-to-end benchmarking is working and correctness remains at `100.0%`
- The current focus is no longer benchmark setup; it is decoder decode-path optimization

## Completed Work

### 1. Cluster / benchmark workflow validation

- Testing must run on a compute node; the head node is only for lightweight tasks such as `git pull/push`
- Before testing on a compute node, activate the environment with:

```bash
source /opt/conda/bin/activate mls
```

- The standard command to enter a compute node is:

```bash
srun -p Teaching --gres=gpu:1 --mem=16G --pty bash
```

- In practice, `16G` was not sufficient in the current environment and caused an OOM during weight loading, so later validation/profiling used a larger memory allocation

### 2. Benchmark harness issue identified and fixed

- The main issue in `benchmark_student.py` was identified:
  - it imports modules directly from the template folder
  - therefore the package-level runtime defaults in `__init__.py` were not actually applied
- Before the fix, `glm_asr_triton_template` was effectively running with the wrong configuration:
  - `Linear.BACKEND=torch`
  - `MLP.FUSED=True`
  - `EncoderMLP.FUSED=True`
- That made the template path look much slower than it really was on the same node

### 3. Benchmark / profiling script changes

- `benchmark_student.py` was updated to:
  - apply a consistent Triton runtime configuration explicitly
  - allow environment-variable overrides
- `benchmark_detailed.py` was updated to:
  - apply the same Triton runtime configuration explicitly
  - add warmup so first-call CUDA/cuBLAS initialization does not pollute timings
  - split decode profiling into first decode step vs steady-state decode step

## Current Key Results

### Latest optimization: fast path for `top_k=1`

- A dedicated `top_k == 1` fast path was added in `glm_asr_triton_template/model.py`
- Previously, even when `top_k=1`, generation still performed a full-vocabulary `argsort` and sampling path on every step
- It now uses direct `argmax` when `top_k=1`, avoiding unnecessary sorting work

### Latest optimization: first decode path switched to `past_key_values`

- `generate()` was changed to:
  - run one `use_cache=True` prefill over the full prompt / audio context
  - decode only the new token embedding on later steps
  - reuse KV cache through `past_key_values`
- This is a more direct fix for the decode hot path than re-feeding the entire `inputs_embeds` sequence on every generation step

Measured result:

```bash
python benchmark_student.py glm_asr_triton_template --warmup 0 --runs 1
```

produced:

- `Time: 6378.0 ms`
- `Tokens: 13`
- `Accuracy: 100.0%`
- `Status: PASS`

Compared with the previous `6785.1 ms`, this is another improvement of about `407.1 ms` or roughly `6.0%`

Latest measurement:

```bash
python benchmark_student.py glm_asr_triton_template --warmup 0 --runs 1
```

produced:

- `Time: 6785.1 ms`
- `Tokens: 13`
- `Accuracy: 100.0%`
- `Status: PASS`

Compared with the post-benchmark-fix baseline of `7437.8 ms`, this is an additional improvement of about `652.7 ms` or roughly `8.8%`

### End-to-end result after fixing benchmark configuration

Running on a compute node:

```bash
python benchmark_student.py glm_asr_triton_template --warmup 0 --runs 1
```

produced:

- `Time: 7437.8 ms`
- `Tokens: 13`
- `Accuracy: 100.0%`
- `Status: PASS`

### Template stage breakdown after fixing profiling

Running on a compute node:

```bash
python benchmark_detailed.py glm_asr_triton_template --runs 2 --seq-len 128
```

produced:

- Audio Encoder: `345.10 ms`
- Multi-modal Projector: `1.86 ms`
- Decoder Prefill: `77.66 ms`
- Decode Step (avg): `84.39 ms`
- Decode Step (steady-state): `66.90 ms`
- First decode step: `241.85 ms`

Estimated total:

- Total (estimated for 50 tokens): `4644.18 ms`

### Current conclusion

- The dominant hotspot is now clearly the decoder decode stage
- Most of the decode variance comes from the first decode step
- Steady-state decode is significantly lower, so the main target is first-step setup cost rather than treating every decode step as equally slow

## Implementation Difference Worth Revisiting

- In `glm_asr_triton_template/layers.py`, several cached transpose paths were changed from:

```python
.t().contiguous()
```

to:

```python
.t()
```

- This may matter later when Triton / fused paths are enabled
- Under the current `cublas + unfused` runtime configuration, it is not yet the confirmed primary bottleneck

## Next Step

- Continue investigating the special path taken by the first decode step
- Focus on:
  - KV cache initialization
  - first-step attention scheduling
  - first-step tensor reshaping / buffer preparation
- A pre-allocated buffer-based decode rewrite for `generate()` was tested once, but it was a net regression in the current implementation and was reverted from the working baseline
- After isolating that setup cost, decide whether the next optimization should be:
  - improving the attention cache path
  - restoring selected `.contiguous()` transpose caches
  - re-evaluating whether fused/Triton paths are worth enabling on the current GPU
