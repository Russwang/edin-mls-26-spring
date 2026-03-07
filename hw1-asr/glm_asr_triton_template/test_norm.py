"""Numerical checks for Triton RMSNorm/LayerNorm kernels."""

import torch

import layers


def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.to(torch.float32)
    var = torch.mean(x32 * x32, dim=-1, keepdim=True)
    return x32 * torch.rsqrt(var + eps) * weight


def layernorm_ref(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float
) -> torch.Tensor:
    x32 = x.to(torch.float32)
    mean = torch.mean(x32, dim=-1, keepdim=True)
    var = torch.var(x32, dim=-1, keepdim=True, unbiased=False)
    x_norm = (x32 - mean) * torch.rsqrt(var + eps)
    return x_norm * weight + bias


def run_case(shape: tuple[int, ...], dtype: torch.dtype, hidden: int, seed: int = 0) -> None:
    device = torch.device("cuda")
    torch.manual_seed(seed)

    x = torch.randn(*shape, device=device, dtype=dtype)

    rms = layers.RMSNorm(hidden, eps=1e-6)
    rms.weight = torch.randn(hidden, dtype=torch.float32)
    y_rms = rms(x)
    y_rms_ref = rmsnorm_ref(x, rms.weight.to(device), rms.eps)

    ln = layers.LayerNorm(hidden, eps=1e-5)
    ln.weight = torch.randn(hidden, dtype=torch.float32)
    ln.bias = torch.randn(hidden, dtype=torch.float32)
    y_ln = ln(x)
    y_ln_ref = layernorm_ref(x, ln.weight.to(device), ln.bias.to(device), ln.eps)

    print(
        f"shape={shape}, dtype={dtype}, hidden={hidden}, triton_path={rms.use_triton and x.is_cuda}: "
        f"rms_max_abs_err={max_abs_err(y_rms, y_rms_ref):.6e}, "
        f"ln_max_abs_err={max_abs_err(y_ln, y_ln_ref):.6e}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this test.")

    print("Norm numerical check (Triton/Fallback vs PyTorch formula)")
    print("-" * 64)

    # Power-of-two hidden sizes exercise Triton kernels.
    run_case((2, 16, 256), torch.float32, hidden=256, seed=0)
    run_case((4, 7, 512), torch.float16, hidden=512, seed=1)

    # Non-power-of-two hidden size exercises fallback path.
    run_case((3, 5, 257), torch.float32, hidden=257, seed=2)

    print("-" * 64)


if __name__ == "__main__":
    main()
