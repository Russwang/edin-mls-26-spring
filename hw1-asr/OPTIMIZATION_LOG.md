# HW1 Optimization Log (Triton)

## Environment
- Repo: `edin-mls-26-spring`
- Track: `glm_asr_triton_template`
- GPU: (fill on node)  
- Date: (fill)

## Fixed Benchmark Command
```bash
cd hw1-asr
./benchmark.sh glm_asr_triton_template
./benchmark.sh glm_asr_triton_example
```

## Current Baseline (Already Measured)
| Variant | Time (ms) | Tokens | ms/token | Accuracy | Status |
|---|---:|---:|---:|---:|---|
| triton_template (current) | 1533.5 | 13 | 117.96 | 100.0% | PASS |
| triton_example (baseline) | 1620.4 | 13 | 124.65 | 100.0% | PASS |

Improvement vs example: **+86.9 ms (~5.4% faster)**.

## Experiment A: Tile/Block Tuning (Requirement 1)

Run commands (same node, same environment):

```bash
# A1: 64/64/32
TRITON_TILE_M=64 TRITON_TILE_N=64 TRITON_TILE_K=32 ./benchmark.sh glm_asr_triton_template

# A2: 128/64/32
TRITON_TILE_M=128 TRITON_TILE_N=64 TRITON_TILE_K=32 ./benchmark.sh glm_asr_triton_template

# A3: 64/128/32
TRITON_TILE_M=64 TRITON_TILE_N=128 TRITON_TILE_K=32 ./benchmark.sh glm_asr_triton_template
```

### A1
- `Linear.TILE_M/TILE_N/TILE_K` = `64/64/32` (current)
- Result: (fill)

### A2
- `Linear.TILE_M/TILE_N/TILE_K` = `128/64/32`
- Result: (fill)

### A3
- `Linear.TILE_M/TILE_N/TILE_K` = `64/128/32`
- Result: (fill)

### A Summary
| Config | Time (ms) | Accuracy | Notes |
|---|---:|---:|---|
| 64/64/32 |  |  |  |
| 128/64/32 |  |  |  |
| 64/128/32 |  |  |  |

Best config: (fill)

## Experiment B: Fusion On/Off (Requirement 2)

### Flags
- `MLP.FUSED = True/False`
- `EncoderMLP.FUSED = True/False`

### Results
| MLP.FUSED | EncoderMLP.FUSED | Time (ms) | Accuracy | Notes |
|---|---|---:|---:|---|
| True | True |  |  | current default |
| False | False |  |  |  |

Conclusion: (fill)

## Requirement 3: FlashAttention-style Attention
Current status: not implemented yet (current attention is score + softmax + output kernels).

Plan:
1. Implement blockwise QK^T + streaming softmax + V accumulation in `attention.py`.
2. Re-run benchmark and compare with current implementation.
3. Record accuracy and timing.

## Final Submission Notes
- Correctness target: `Accuracy > 80%` (current: 100%).
- Performance target: faster than `glm_asr_triton_example` (current: yes).
- Keep code constraints: do not modify `model.py`, `weight_loader.py`, `conv.py`.
