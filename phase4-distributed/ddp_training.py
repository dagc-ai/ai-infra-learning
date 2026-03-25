# ddp_training.py
# Exercise 4.1 — PyTorch DistributedDataParallel (DDP)
#
# What this demonstrates:
#   - One process per GPU, each with its own Python interpreter and CUDA context
#   - Each process trains on DIFFERENT data (data parallelism)
#   - After every backward pass, DDP automatically fires AllReduce via NCCL
#   - AllReduce averages gradients across all ranks over NVLink
#   - Every rank takes an identical optimizer step → models stay in sync
#
# The proof: initial param sums match (DDP broadcast at init)
#            final param sums match (AllReduce worked every step)

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time


def setup(rank, world_size):
    # MASTER_ADDR/PORT: where the rendezvous happens
    # All processes connect here to coordinate before training starts
    # localhost = single machine, all processes on same node
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # init_process_group: initializes the distributed communication backend
    # "nccl" = NVIDIA Collective Communications Library
    #   - NCCL is what actually drives NVLink between GPUs
    #   - It handles AllReduce, AllGather, ReduceScatter under the hood
    #   - PyTorch's dist.all_reduce() is just an API call into NCCL
    # rank: this process's unique ID (0, 1, 2, or 3)
    # world_size: total number of processes (4)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Pin this process to its corresponding GPU
    # rank 0 → cuda:0, rank 1 → cuda:1, etc.
    # Without this, all processes would default to cuda:0 and fight over it
    torch.cuda.set_device(rank)


def cleanup():
    # Tear down the process group cleanly
    # Frees NCCL resources and closes communication channels
    dist.destroy_process_group()


def train(rank, world_size):
    setup(rank, world_size)

    # Every rank builds the SAME model architecture with the SAME random seed
    # (PyTorch uses the same default seed across processes spawned by mp.spawn)
    # DDP will broadcast rank 0's weights to all ranks at construction anyway
    # — so even if seeds differed, all ranks would end up with rank 0's weights
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),   # 1024*1024 weights + 1024 bias = ~1M params
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1024),
    ).to(rank)                          # .to(rank) moves model to this rank's GPU

    # Wrap model in DDP
    # This is where the magic happens:
    #   1. DDP broadcasts rank 0's weights to all other ranks (sync at init)
    #   2. DDP registers backward hooks on every parameter
    #   3. When loss.backward() fires, those hooks trigger AllReduce automatically
    #      as each gradient bucket is ready — overlapping comm with computation
    # device_ids=[rank]: tells DDP which GPU this process owns
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # Verify sync at init: all ranks should print the same number
    # If DDP broadcast worked, param sums are identical across all 4 ranks
    param_sum = sum(p.sum().item() for p in ddp_model.parameters())
    print(f"[Rank {rank}] Initial param sum: {param_sum:.6f}")

    NUM_STEPS = 20
    start = time.perf_counter()

    for step in range(NUM_STEPS):
        optimizer.zero_grad()   # clear gradients from previous step

        # KEY: each rank generates DIFFERENT random data
        # This is the whole point of data parallelism —
        # each GPU processes a different slice of the global batch
        # Effective global batch size = 256 * 4 = 1024 samples per step
        inputs  = torch.randn(256, 1024).to(rank)
        targets = torch.randn(256, 1024).to(rank)

        outputs = ddp_model(inputs)         # forward pass (no communication)
        loss = loss_fn(outputs, targets)

        # backward() computes gradients AND triggers AllReduce
        # DDP hooks fire here — as each parameter's gradient is computed,
        # NCCL initiates AllReduce for that gradient bucket over NVLink
        # By the time backward() returns, all gradients are already averaged
        loss.backward()

        # optimizer.step() is called identically on every rank
        # Because gradients are already averaged, every rank takes the same step
        # → model weights remain synchronized without any explicit sync call
        optimizer.step()

        if rank == 0 and step % 5 == 0:
            print(f"[Rank {rank}] Step {step}, Loss: {loss.item():.4f}")

    # synchronize: wait for all CUDA operations on this device to complete
    # before stopping the timer — without this, timing would be wrong
    # because CUDA kernels execute asynchronously
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Verify sync after training: all ranks should STILL print the same number
    # Each rank saw different data → different raw gradients
    # AllReduce averaged those gradients every step → identical optimizer steps
    # → identical final weights. If these differ, AllReduce failed.
    param_sum_after = sum(p.sum().item() for p in ddp_model.parameters())
    print(f"[Rank {rank}] Final param sum: {param_sum_after:.6f}  |  Time: {elapsed:.2f}s")

    cleanup()


if __name__ == "__main__":
    world_size = 4  # one process per GPU

    # mp.spawn launches `world_size` processes, each running train(rank, world_size)
    # rank is assigned automatically: 0, 1, 2, 3
    # This is the standard single-node multi-GPU launch pattern
    # For multi-node: you'd use torchrun instead of mp.spawn
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
