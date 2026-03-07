"""Numerical checks for Triton GELU/SiLU kernels."""

import torch
import torch.nn.functional as F

import layers


def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def run_case(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> tuple[float, float]:
    x = torch.randn(*shape, device=device, dtype=dtype)

    y_gelu = layers.gelu(x)
    y_gelu_ref = F.gelu(x.to(torch.float32), approximate="tanh")

    y_silu = layers.silu(x)
    y_silu_ref = F.silu(x.to(torch.float32))

    return max_abs_err(y_gelu, y_gelu_ref), max_abs_err(y_silu, y_silu_ref)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this test.")

    device = torch.device("cuda")
    torch.manual_seed(0)

    cases = [
        ((257,), torch.float32),          # tail mask edge case
        ((1023,), torch.float32),         # larger non-multiple length
        ((2, 16, 257), torch.float32),    # model-like 3D shape with tail
        ((4, 33, 129), torch.float16),    # mixed shape + fp16 input
    ]

    print("Elementwise numerical check (Triton vs PyTorch)")
    print("-" * 56)
    overall_gelu = 0.0
    overall_silu = 0.0

    for shape, dtype in cases:
        gelu_err, silu_err = run_case(shape, dtype, device)
        overall_gelu = max(overall_gelu, gelu_err)
        overall_silu = max(overall_silu, silu_err)
        print(f"shape={shape}, dtype={dtype}: gelu_max_abs_err={gelu_err:.6e}, silu_max_abs_err={silu_err:.6e}")

    print("-" * 56)
    print(f"overall max abs err: GELU={overall_gelu:.6e}, SiLU={overall_silu:.6e}")


if __name__ == "__main__":
    main()
