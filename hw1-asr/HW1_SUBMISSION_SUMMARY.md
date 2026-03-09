# HW1-ASR Triton Submission Summary

## 1. Setup
- Track: `glm_asr_triton_template`
- Reference baseline: `glm_asr_triton_example`
- Main benchmark command:
  - `./benchmark.sh glm_asr_triton_template`
  - `./benchmark.sh glm_asr_triton_example`

## 2. Final Correctness Result
- Final run (stable config):
  - Time: `1030.9 ms` (`+/- 6.8 ms`)
  - Tokens: `13`
  - Speed: `79.30 ms/token`
  - Transcription: `Concord returned to its place amidst the tents.`
  - Accuracy: `100.0%`
  - Status: `PASS`

## 3. Optimization Evidence

### Requirement A: Tile/Block Tuning (2-3 configs)
Measured on the same benchmark pipeline:

| Config (`TILE_M/N/K`) | Time (ms) | Accuracy | Notes |
|---|---:|---:|---|
| `64/64/32` | `1376.7` | `100.0%` | Best + stable |
| `128/64/32` | `1509.6` | `100.0%` | Slower + unstable |
| `64/128/32` | `1439.2` | `100.0%` | Better than 128/64/32 |

Best config selected: `64/64/32`.

### Requirement B: Kernel Fusion (at least one)
Used fused kernels in `layers.py` (`MLP.FUSED`, `EncoderMLP.FUSED`) and compared ON/OFF:

| MLP.FUSED | EncoderMLP.FUSED | Time (ms) | Accuracy |
|---|---|---:|---:|
| ON | ON | `1381.5` | `100.0%` |
| OFF | OFF | `1684.7` | `100.0%` |

Fusion ON improves latency by about `303.2 ms` (~`18.0%`).

### Requirement C: FlashAttention-style Attention
- Implemented a minimal streaming FlashAttention-style kernel path in `attention.py`:
  - blockwise score accumulation
  - streaming softmax state update (`m`, `l`)
  - weighted value accumulation
- Control flag: `TRITON_USE_FLASH_ATTN=1` to enable.
- For final submission stability, default is kept at `TRITON_USE_FLASH_ATTN=0` (validated path), because current flash path still needs extra numerical/masking refinement for full end-to-end stability.

## 4. Comparison vs Baseline
- Example baseline (`glm_asr_triton_example`): `1488.7 ms`, `100%`, PASS
- Student template (current optimized version): `1030.9 ms`, `100%`, PASS
- Relative gain: about `30.8%` faster than baseline on the same compute node.

## 5. Files Modified (student work)
- `glm_asr_triton_template/layers.py`
- `glm_asr_triton_template/attention.py`
- `glm_asr_triton_template/rope.py`
- `glm_asr_triton_template/model.py`
- `benchmark_student.py`
- Validation scripts:
  - `glm_asr_triton_template/test_elementwise.py`
  - `glm_asr_triton_template/test_norm.py`
  - `glm_asr_triton_template/test_linear.py`
- Experiment log:
  - `OPTIMIZATION_LOG.md`

## 6. Notes
- Current benchmark path explicitly applies Triton runtime config in `benchmark_student.py`.
- The latest speedup also includes decode-path improvements in `model.py`:
  - `top_k == 1` fast path
  - KV-cache-based incremental decode
- Detailed experiment outputs are tracked in `OPTIMIZATION_LOG.md`.
