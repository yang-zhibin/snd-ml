#!/usr/bin/env python3
"""
cuda_sanity_check.py

A single script to diagnose common performance regressions after CUDA upgrades.

What it checks:
- nvidia-smi presence + driver/GPU summary
- PyTorch version, CUDA build, GPU availability, device details
- TF32 / cudnn / determinism-related settings
- Imports and basic sanity for PyTorch Geometric + compiled extensions (scatter/sparse/cluster/pyg_lib)
- Quick GPU microbench: matmul throughput (float32)
- DataLoader throughput microbench (CPU -> optionally pinned memory)
- Optional tiny PyG message passing run if PyG is available

Usage:
  python cuda_sanity_check.py
  python cuda_sanity_check.py --dataloader --workers 4 --pin
  python cuda_sanity_check.py --gpu-bench
  python cuda_sanity_check.py --pyg
  python cuda_sanity_check.py --all
"""

import argparse
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple


def run_cmd(cmd: list[str], timeout: int = 10) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"Timeout running: {' '.join(cmd)}"


def header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def kv(k: str, v) -> None:
    print(f"{k:28s}: {v}")


def try_import(name: str):
    try:
        mod = __import__(name)
        return True, mod, None
    except Exception as e:
        return False, None, e


def nvidia_smi_summary() -> None:
    header("NVIDIA / Driver (nvidia-smi)")
    rc, out, err = run_cmd(["nvidia-smi"], timeout=10)
    if rc != 0:
        kv("nvidia-smi", f"NOT OK (rc={rc})")
        kv("error", err or "(no stderr)")
        return
    # Print first ~20 lines to keep it readable
    lines = out.splitlines()
    for line in lines[:20]:
        print(line)
    if len(lines) > 20:
        print("... (truncated)")

    # Also pull machine-readable info if available
    rc2, out2, err2 = run_cmd(
        ["nvidia-smi", "--query-gpu=name,driver_version,cuda_version,compute_cap", "--format=csv,noheader"],
        timeout=10,
    )
    if rc2 == 0 and out2:
        kv("query", out2)


def torch_env_summary() -> dict:
    header("PyTorch / CUDA runtime sanity")
    info = {}
    ok, torch, e = try_import("torch")
    if not ok:
        kv("torch import", f"FAILED: {e}")
        return info

    info["torch_version"] = torch.__version__
    info["torch_cuda_build"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    info["cuda_device_count"] = torch.cuda.device_count()

    kv("python", sys.version.replace("\n", " "))
    kv("platform", f"{platform.platform()} ({platform.machine()})")
    kv("torch.__version__", info["torch_version"])
    kv("torch.version.cuda", info["torch_cuda_build"])
    kv("torch.cuda.is_available()", info["cuda_available"])
    kv("torch.cuda.device_count()", info["cuda_device_count"])

    if info["cuda_available"]:
        dev = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(dev)
        kv("gpu name", props.name)
        kv("compute capability", f"{props.major}.{props.minor}")
        kv("total memory (GB)", f"{props.total_memory/1024**3:.2f}")
        kv("multi_processor_count", props.multi_processor_count)
        kv("cuda runtime", torch.version.cuda)
        kv("cudnn version", torch.backends.cudnn.version())
    else:
        print("\nNOTE: CUDA is not available to PyTorch. Training is likely on CPU.")

    # Backend / determinism / TF32 flags
    header("Backend flags that affect performance")
    kv("cudnn.enabled", getattr(torch.backends.cudnn, "enabled", None))
    kv("cudnn.benchmark", getattr(torch.backends.cudnn, "benchmark", None))
    kv("cudnn.deterministic", getattr(torch.backends.cudnn, "deterministic", None))
    try:
        kv("deterministic_algorithms", torch.are_deterministic_algorithms_enabled())
    except Exception:
        kv("deterministic_algorithms", "unknown")

    # TF32 flags exist on Ampere+
    try:
        kv("matmul.allow_tf32", torch.backends.cuda.matmul.allow_tf32)
        kv("cudnn.allow_tf32", torch.backends.cudnn.allow_tf32)
    except Exception:
        kv("tf32 flags", "not available")

    # Env vars that sometimes force slow paths
    header("Env vars that may affect CUDA performance")
    for k in [
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_TF32_OVERRIDE",
        "CUBLAS_WORKSPACE_CONFIG",
        "TORCH_LOGS",
        "TORCHINDUCTOR_CACHE_DIR",
        "XLA_FLAGS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]:
        if k in os.environ:
            kv(k, os.environ.get(k))

    return info


def pyg_extensions_summary() -> None:
    header("PyTorch Geometric + compiled extension imports")
    ok_torch, torch, _ = try_import("torch")
    if not ok_torch:
        kv("torch", "not importable; skipping PyG checks")
        return

    ok, pyg, e = try_import("torch_geometric")
    if not ok:
        kv("torch_geometric import", f"FAILED: {e}")
        print("If you expected PyG, install/repair torch-geometric + matching wheels for your torch+cuda build.")
        return
    kv("torch_geometric.__version__", getattr(pyg, "__version__", "unknown"))

    # Try the common compiled deps
    for name in ["pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv"]:
        ok2, mod, e2 = try_import(name)
        if ok2:
            ver = getattr(mod, "__version__", "unknown")
            kv(f"{name} import", f"OK (version {ver})")
        else:
            kv(f"{name} import", f"FAILED: {e2}")


def gpu_matmul_bench(iters: int = 50, n: int = 4096, dtype: str = "float32") -> None:
    header("GPU microbenchmark: matmul throughput")
    ok, torch, e = try_import("torch")
    if not ok:
        kv("torch import", f"FAILED: {e}")
        return
    if not torch.cuda.is_available():
        print("CUDA not available; skipping GPU benchmark.")
        return

    dt = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}.get(dtype, torch.float32)
    device = torch.device("cuda:0")

    # Warmup
    a = torch.randn(n, n, device=device, dtype=dt)
    b = torch.randn(n, n, device=device, dtype=dt)
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    sec = t1 - t0
    # FLOPs for matmul: ~2*n^3 operations
    flops = 2.0 * (n ** 3) * iters
    tflops = flops / sec / 1e12

    kv("dtype", str(dt))
    kv("matrix size", f"{n}x{n}")
    kv("iters", iters)
    kv("time (s)", f"{sec:.3f}")
    kv("approx TFLOP/s", f"{tflops:.2f}")

    print("\nTip: If TFLOP/s is extremely low or time is huge, you may be on a slow path.")


def dataloader_bench(samples: int = 20000, batch_size: int = 256, workers: int = 4, pin: bool = False) -> None:
    header("DataLoader throughput microbenchmark")
    ok, torch, e = try_import("torch")
    if not ok:
        kv("torch import", f"FAILED: {e}")
        return

    from torch.utils.data import Dataset, DataLoader

    class Dummy(Dataset):
        def __init__(self, n: int, feat: int = 1024):
            self.n = n
            self.feat = feat

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            # Simulate CPU preprocessing a bit
            x = torch.randn(self.feat)
            y = torch.randint(0, 2, (1,)).item()
            return x, y

    ds = Dummy(samples)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=(workers > 0),
        drop_last=True,
    )

    # iterate for a fixed number of batches
    steps = 200
    t0 = time.perf_counter()
    it = iter(dl)
    for _ in range(steps):
        xb, yb = next(it)
        # Optional host->device copy to see pin_memory benefit
        if torch.cuda.is_available():
            xb = xb.to("cuda", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    sec = t1 - t0
    items = steps * batch_size
    kv("workers", workers)
    kv("pin_memory", pin)
    kv("batch_size", batch_size)
    kv("steps", steps)
    kv("items processed", items)
    kv("time (s)", f"{sec:.3f}")
    kv("items/s", f"{items/sec:.1f}")

    print("\nTip: If items/s is very low and GPU util is low during training, you're input-bound.")


def pyg_tiny_run() -> None:
    header("Optional: tiny PyG run (message passing sanity)")
    ok_torch, torch, e = try_import("torch")
    if not ok_torch:
        kv("torch import", f"FAILED: {e}")
        return
    ok_pyg, pyg, e2 = try_import("torch_geometric")
    if not ok_pyg:
        kv("torch_geometric import", f"FAILED: {e2}")
        return

    try:
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv
    except Exception as e3:
        kv("PyG components", f"FAILED: {e3}")
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kv("device", device)

    # Small random graph
    num_nodes = 20000
    num_edges = 80000
    x = torch.randn(num_nodes, 64, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    data = Data(x=x, edge_index=edge_index)

    conv = GCNConv(64, 64).to(device)
    # Warmup
    for _ in range(5):
        out = conv(data.x, data.edge_index)
        loss = out.pow(2).mean()
        loss.backward()
        conv.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    steps = 30
    for _ in range(steps):
        out = conv(data.x, data.edge_index)
        loss = out.pow(2).mean()
        loss.backward()
        conv.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    kv("nodes", num_nodes)
    kv("edges", num_edges)
    kv("steps", steps)
    kv("time (s)", f"{(t1 - t0):.3f}")

    print("\nIf this is very slow on CUDA, PyG extensions may be missing/mismatched, or you're on CPU.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Run all checks/benches.")
    ap.add_argument("--no-smi", action="store_true", help="Skip nvidia-smi.")
    ap.add_argument("--pyg", action="store_true", help="Run PyG import checks + tiny run.")
    ap.add_argument("--gpu-bench", action="store_true", help="Run GPU matmul microbenchmark.")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="dtype for GPU bench.")
    ap.add_argument("--n", type=int, default=4096, help="matrix size for GPU bench (n x n).")
    ap.add_argument("--iters", type=int, default=50, help="iterations for GPU bench.")
    ap.add_argument("--dataloader", action="store_true", help="Run DataLoader throughput microbenchmark.")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    ap.add_argument("--pin", action="store_true", help="Use pin_memory in DataLoader bench.")
    ap.add_argument("--samples", type=int, default=20000, help="Number of dataset samples in DataLoader bench.")
    ap.add_argument("--batch-size", type=int, default=256, help="Batch size for DataLoader bench.")
    args = ap.parse_args()

    if not args.no_smi:
        nvidia_smi_summary()

    torch_env_summary()

    # Always do PyG import summary if installed, because it's a common slowdown source.
    pyg_extensions_summary()

    run_all = args.all
    if args.gpu_bench or run_all:
        gpu_matmul_bench(iters=args.iters, n=args.n, dtype=args.dtype)

    if args.dataloader or run_all:
        dataloader_bench(samples=args.samples, batch_size=args.batch_size, workers=args.workers, pin=args.pin)

    if args.pyg or run_all:
        pyg_tiny_run()

    header("Done")
    print("If you paste this output, I can tell you what went wrong (CPU fallback, mismatch, input bottleneck, etc.).")


if __name__ == "__main__":
    main()
