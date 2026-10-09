"""Phase 1 check: can PyTorch in the training container use this machine's NVIDIA GPU?"""
import time
import torch

print("PyTorch:", torch.__version__, "| CUDA build:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available inside the container"
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name} | memory: {props.total_memory / 1024**3:.1f} GB | compute capability: {props.major}.{props.minor}")
archs = torch.cuda.get_arch_list()
print("Kernels built for:", archs)
need = f"sm_{props.major}{props.minor}"
assert need in archs, f"This PyTorch build has no kernels for this GPU ({need}); rebuild with another TORCH_INDEX"

# Small half-precision matmul benchmark (what training with mixed precision uses)
for dtype in (torch.float32, torch.float16):
    a = torch.randn(2048, 2048, device="cuda", dtype=dtype)
    b = torch.randn(2048, 2048, device="cuda", dtype=dtype)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(50):
        a @ b
    torch.cuda.synchronize()
    tflops = 50 * 2 * 2048**3 / (time.time() - t) / 1e12
    print(f"{str(dtype):15s} ~{tflops:.2f} TFLOPS")
print("OK: training container is ready")
