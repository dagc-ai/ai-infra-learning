import torch

torch.manual_seed(42)

# Target 1: memory-bound
size = 1 << 22
a = torch.randn(size, device='cuda')
b = torch.randn(size, device='cuda')
_ = a * b
torch.cuda.synchronize()
for _ in range(10):
    c = a * b
torch.cuda.synchronize()

# Target 2: compute-bound
n = 2048
A = torch.randn(n, n, device='cuda', dtype=torch.float16)
B = torch.randn(n, n, device='cuda', dtype=torch.float16)
_ = torch.matmul(A, B)
torch.cuda.synchronize()
for _ in range(10):
    C = torch.matmul(A, B)
torch.cuda.synchronize()

print("Done.")
