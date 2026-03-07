"""Numerical checks for Triton linear_kernel_tf32 via Linear layer."""

import torch

import layers


def max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def run_case(shape: tuple[int, ...], in_features: int, out_features: int, dtype: torch.dtype, seed: int) -> None:
    torch.manual_seed(seed)
    device = torch.device("cuda")

    x = torch.randn(*shape, in_features, device=device, dtype=dtype)

    linear = layers.Linear(in_features, out_features, bias=True)
    linear.weight = torch.randn(out_features, in_features, dtype=torch.float32)
    linear.bias_param = torch.randn(out_features, dtype=torch.float32)

    old_backend = layers.Linear.BACKEND
    try:
        layers.Linear.BACKEND = "torch"
        y_torch = linear(x)

        layers.Linear.BACKEND = "triton"
        y_triton = linear(x)
    finally:
        layers.Linear.BACKEND = old_backend

    err = max_abs_err(y_triton, y_torch)
    print(
        f"shape={shape}+[{in_features}], out={out_features}, dtype={dtype}: "
        f"max_abs_err={err:.6e}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this test.")

    print("Linear numerical check (Triton vs Torch)")
    print("-" * 56)

    run_case(shape=(2, 16), in_features=256, out_features=512, dtype=torch.float32, seed=0)
    run_case(shape=(4, 7), in_features=257, out_features=513, dtype=torch.float32, seed=1)   # non-multiple of tile sizes
    run_case(shape=(3, 11), in_features=320, out_features=640, dtype=torch.float16, seed=2)

    print("-" * 56)


if __name__ == "__main__":
    main()
