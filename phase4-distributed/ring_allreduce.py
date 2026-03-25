# ring_allreduce.py
# Exercise 4.2 — Ring AllReduce implemented from scratch
#
# What this demonstrates:
#   - The exact algorithm NCCL executes under the hood during DDP training
#   - Built from point-to-point send/recv primitives only — no dist.all_reduce()
#   - Two phases: Reduce-Scatter (accumulate) → AllGather (broadcast)
#   - After completion, every rank holds the identical summed tensor
#
# Key fix vs. naive implementation:
#   - Must use batch_isend_irecv, not isend + recv separately
#   - Naive approach deadlocks: every rank posts isend then blocks on recv
#   - NCCL isend doesn't move data until matching recv is posted on remote end
#   - With all ranks blocked in recv simultaneously, nobody makes progress
#   - batch_isend_irecv posts all sends AND recvs atomically → NCCL matches them

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def ring_allreduce(tensor, rank, world_size):
    """
    Manual Ring AllReduce using point-to-point communication.

    Args:
        tensor: the gradient tensor this rank wants to reduce
        rank: this process's ID
        world_size: total number of processes

    Returns:
        tensor containing the element-wise sum across all ranks
    """

    # Work on a local copy so we don't modify the original
    tensor = tensor.clone()

    # Split tensor into world_size equal chunks
    # Each chunk will be fully reduced and owned by one rank after Phase 1
    chunk_size = tensor.numel() // world_size
    chunks = [tensor[i * chunk_size:(i + 1) * chunk_size] for i in range(world_size)]

    # Ring neighbors — fixed for all steps
    send_to   = (rank + 1) % world_size   # always send forward
    recv_from = (rank - 1) % world_size   # always receive from behind

    # --- Phase 1: Reduce-Scatter ---
    # Each step: send one chunk forward, receive one chunk from behind, accumulate
    # After (world_size - 1) steps: rank r holds fully-summed chunk r

    for step in range(world_size - 1):
        # Which chunk to send this step — rotates so every chunk visits every GPU
        send_idx = (rank - step) % world_size

        # Which chunk we'll receive and accumulate into
        recv_idx = (rank - step - 1) % world_size

        recv_buffer = torch.zeros_like(chunks[recv_idx])

        # batch_isend_irecv: post send AND recv simultaneously as a batch
        # NCCL matches them across ranks without deadlock
        # This is the correct primitive for ring communication patterns
        ops = [
            dist.P2POp(dist.isend, chunks[send_idx], send_to),
            dist.P2POp(dist.irecv, recv_buffer,      recv_from),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()  # wait for both send and recv to complete

        # Accumulate: add received partial gradient into our local chunk
        chunks[recv_idx] += recv_buffer

    # After Phase 1:
    #   chunks[rank] = fully summed value for that chunk position
    #   e.g. rank 0 holds sum of chunk 0 across all 4 GPUs

    # --- Phase 2: AllGather ---
    # Broadcast each rank's fully-summed chunk around the ring
    # Same mechanics as Phase 1 but no accumulation — just overwrite
    # After (world_size - 1) steps: every rank holds all fully-summed chunks

    for step in range(world_size - 1):
        # Start from our own fully-summed chunk, then forward received chunks
        send_idx = (rank - step + 1) % world_size

        # Slot to write incoming fully-summed chunk into
        recv_idx = (rank - step) % world_size

        recv_buffer = torch.zeros_like(chunks[recv_idx])

        ops = [
            dist.P2POp(dist.isend, chunks[send_idx], send_to),
            dist.P2POp(dist.irecv, recv_buffer,      recv_from),
        ]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        # Overwrite — no addition, this chunk is already fully reduced
        chunks[recv_idx].copy_(recv_buffer)

    return tensor


def run(rank, world_size):
    setup(rank, world_size)

    # Each rank starts with a different tensor — intentional
    # This simulates each GPU having computed different gradients
    torch.manual_seed(rank)
    original = torch.randn(1024, device=rank)

    # Run our manual ring AllReduce
    result = ring_allreduce(original, rank, world_size)

    # Verify against PyTorch's built-in dist.all_reduce()
    # If our implementation is correct, results must match exactly
    reference = original.clone()
    dist.all_reduce(reference, op=dist.ReduceOp.SUM)

    max_diff = (result - reference).abs().max().item()
    print(f"[Rank {rank}] Max diff vs dist.all_reduce: {max_diff:.2e}  |  "
          f"First 4 values: {result[:4].tolist()}")

    # All ranks should print identical first 4 values
    # Max diff should be ~0 (floating point epsilon at most)

    cleanup()


if __name__ == "__main__":
    world_size = 4
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
