# CUDA Kernel Optimization — Phase 2 Notes
*RTX 4090 | CUDA 12.1 | Benchmarked March 2026*

---

## The Central Insight

On a GPU, how you move data matters more than how you compute.

A stock RTX 4090 can execute 82 trillion floating point operations per second. The naive
matrix multiply written in Phase 2 used 0.17% of that. The GPU was not broken. The math
was not wrong. The data movement was catastrophically inefficient. Every optimization in
this phase — tiling, shared memory, coalesced access — was entirely about data movement,
not computation. The actual multiply-add instructions were identical across every kernel
version. The arithmetic intensity was the only thing that changed.

This inverts the intuition most people bring to GPU programming. They think of a GPU as a
fast calculator and assume the bottleneck is arithmetic. It is almost never arithmetic. It
is almost always the memory system.

---

## The Memory Hierarchy — With Real Measured Numbers
```
Level           Latency        Bandwidth              Size              Managed by
────────────────────────────────────────────────────────────────────────────────────
Registers       ~1 cycle       ~20 TB/s (est.)        256 KB/SM         Compiler
Shared Memory   ~5 cycles      ~19 TB/s (est.)        128 KB/SM         Programmer
L1/Tex Cache    ~30 cycles     automatic              128 KB/SM         Hardware
L2 Cache        ~200 cycles    ~7 TB/s                72 MB total       Hardware
HBM (GDDR6X)    ~500 cycles    947.5 GB/s measured    24 GB             N/A
```

HBM bandwidth measured in Phase 1: elementwise multiply hitting 94% of peak (947.5 GB/s).
The 100x latency difference between shared memory and HBM is the entire reason tiling exists.

---

## The Warp Execution Model

The GPU scheduler does not see individual threads. It sees warps: groups of 32 threads
that execute in lockstep on a single SM. When a warp issues an HBM load, all 32 threads
stall for ~500 cycles. The SM hides this latency by switching to a different warp that
has its data ready. This is called latency hiding, and it is the primary mechanism that
makes GPUs efficient — when it works.

SM occupancy is the percentage of peak concurrent warps that are active. High occupancy
gives the scheduler more warps to switch between, which means more latency hiding, which
means more of those 500-cycle HBM stalls get covered by useful work from other warps.
When a kernel is memory-bound, it is usually because: there are not enough warps to hide
HBM latency, or the memory access pattern forces serialized transactions.

Key number: RTX 4090 has 128 SMs. Each SM supports up to 64 concurrent warps.
Maximum concurrent warps across the chip: 8,192.

---

## Arithmetic Intensity: The Design Constraint

Arithmetic intensity = FLOPs executed / bytes transferred from HBM.

This single number determines which side of the roofline a kernel lives on and therefore
what its bottleneck is. Compute it before writing a line of code.

RTX 4090 ridge point (FP32): ~82 TFLOPS / ~1,008 GB/s = ~81 FLOP/byte.

Below 81 FLOP/byte: memory bound. Adding more compute does nothing.
Above 81 FLOP/byte: compute bound. Adding more memory bandwidth does nothing.

Phase 2 measurements across the full range:

| Kernel                | Arithmetic Intensity  | Regime         |
|-----------------------|-----------------------|----------------|
| Vec add               | 0.125 FLOP/byte       | Memory bound   |
| Elementwise multiply  | 0.25 FLOP/byte        | Memory bound   |
| Naive matmul          | 0.25 FLOP/byte        | Memory bound   |
| Tiled matmul (T=16)   | ~2.0 FLOP/byte        | Memory bound   |
| cuBLAS (FP32)         | ~50+ FLOP/byte        | Approaching ridge |

Every kernel in this phase is still memory bound. cuBLAS gets close to the ridge point
through three levels of tiling and Tensor Cores. The naive kernel sits 324x below it.

---

## Naive vs. Tiled Matmul: The Math

### Why Naive is Memory Bound

For C = A x B where all matrices are NxN (float32):

FLOPs: N^2 output elements x 2N ops each = 2N^3 total FLOPs.
HBM reads: N^2 threads x 2N floats each = 2N^3 floats = 8N^3 bytes.
Arithmetic intensity: 2N^3 / 8N^3 = 0.25 FLOP/byte.

The same data is read from HBM repeatedly by different threads. Row i of matrix A is
needed by every thread computing any element of output row i — that is N threads all
independently fetching the same N floats from HBM.

### Why Tiling Fixes It

Tiling with tile size T reduces HBM reads by factor T:

A block of T x T threads handles one T x T tile of output C.
They collaboratively load one T x T tile of A and one T x T tile of B into shared memory.
Each HBM value is loaded once, then reused T times from SRAM.
New HBM bytes: 8N^3 / T.
New arithmetic intensity: T/4 FLOP/byte.

At T=16: 4 FLOP/byte — 16x better than naive.
At T=32: 8 FLOP/byte — 32x better than naive.

Still memory bound on the 4090 at these tile sizes, but HBM traffic has dropped by T.
That reduction in traffic IS the speedup. The compute units are not what changed.

---

## Benchmark Results — RTX 4090

### Wall Clock Performance

| Version      | N=1024          | N=2048          | N=4096          | % of cuBLAS |
|--------------|-----------------|-----------------|-----------------|-------------|
| Naive        | 0.43ms / 5.0T   | 3.37ms / 5.1T   | (skipped)       | ~9%         |
| Tiled T=16   | 0.33ms / 6.4T   | 2.61ms / 6.6T   | 19.17ms / 7.2T  | ~13%        |
| cuBLAS       | 0.04ms / 48.7T  | 0.32ms / 54.5T  | 2.55ms / 53.9T  | 100%        |

TFLOPS = trillions of floating point ops per second.

Naive at N=1024 appears faster than predicted because the 4090's 72MB L2 cache absorbs
some redundant loads when all three matrices (12MB total) fit within it. At N=4096
(192MB total — 3x past L2), naive would collapse. The "skipped" row hides the most
dramatic number in the table.

### ncu Hardware Counter Evidence (N=1024)

| Metric                        | Naive      | Tiled T=16 | Ratio  |
|-------------------------------|------------|------------|--------|
| Global load bytes (HBM)       | 4.29 GB    | 537 MB     | 8x less|
| Shared memory wavefronts      | 0          | 50,339,258 | proof  |
| DRAM throughput % of peak     | 1.70%      | 2.20%      | —      |
| SM throughput % of peak       | 97%        | 94%        | —      |

The 8x reduction in global load bytes (vs. predicted 16x) is explained by L2 caching
partially absorbing naive's redundant reads at N=1024. At N=2048 the global load bytes
are identical (34.36 GB for both naive and tiled at that size), confirming L2 is no
longer helping once the working set exceeds it.

The shared memory wavefront count is the key proof: naive shows exactly zero SRAM
accesses. Tiled shows 50 million. Those 50 million are the tile loads — data moving from
HBM into SRAM once, then being reused from SRAM 16 times instead of going back to HBM.

SM throughput is slightly higher for naive (~97%) than tiled (~94%) because the
__syncthreads() barriers in the tiled kernel introduce brief synchronization stalls. This
is a reminder that SM throughput is a composite metric covering all pipelines including
the memory request pipeline — a kernel hammering HBM with load instructions shows high
SM throughput even while its compute units are starving.

### Why cuBLAS Is Still 8x Faster Than Our Tiled Kernel

Our kernel implements one level of tiling in shared memory. cuBLAS implements three:

Level 1 — L2 tiling: problem decomposition so the working set fits in 72MB L2.
Level 2 — Shared memory tiling: what we implemented (cuBLAS uses T=128 or T=256).
Level 3 — Register tiling: each thread accumulates a 4x4 or 8x8 submatrix in registers,
maximizing reuse before touching shared memory at all.

Additionally: cuBLAS uses Tensor Cores (4x TFLOPS for matrix ops vs FP32 FMA),
double-buffering (loads the next tile while computing the current one to hide latency),
and hand-written assembly for the inner loop.

The gap between our kernel and cuBLAS maps exactly to the engineering effort that makes
a production GPU software stack defensible as a product.

---

## Memory Coalescing

Coalescing is how the hardware converts individual thread memory requests into efficient
HBM transactions. When a warp issues a load, the hardware collects all 32 addresses and
groups them into 128-byte cache line transactions.

32 consecutive threads reading 32 consecutive floats = 1 transaction (ideal).
32 consecutive threads reading every 128th float = 32 transactions (32x bandwidth waste).

Rule: consecutive threads (consecutive threadIdx.x) should read consecutive addresses.

In row-major matrix storage, A[row][col] lives at address row*N + col. Threads with
consecutive threadIdx.x mapped to consecutive col values are coalesced. Threads mapped
to consecutive row values (with fixed col) are strided by N — catastrophic for large N.

Access pattern is a first-class design decision. The layout of tensors in memory —
row-major vs column-major, contiguous vs strided — determines whether your kernel
saturates memory bandwidth or wastes most of it.

---

## The __syncthreads() Contract

Every __syncthreads() call is a barrier: every thread in the block stops and waits for
the slowest thread before anyone proceeds.

The tiled matmul uses exactly two barriers per tile loop iteration, which is the minimum
correct placement:

Barrier 1: after loading into shared memory.
Reason: thread (0,0) must not start computing until thread (15,15) has finished loading
its element. Without this barrier, fast threads compute with stale or uninitialized data.

Barrier 2: after computing from shared memory.
Reason: thread (0,0) must not overwrite sA with the next tile until thread (15,15) has
finished reading from sA for the current tile. Without this barrier, data corruption
occurs on the second and later iterations.

Understanding why both barriers are necessary — and what breaks if you remove either —
is the difference between writing correct concurrent code and writing code that works
most of the time.

Barriers are not free: each one serializes the block momentarily. Over-synchronization
reduces the parallelism the SM can exploit. The two-barrier pattern here is both correct
and minimal.

---

## Warp Shuffle: Register-Level Reduction

For operations requiring reduction across threads — max, sum, softmax — warp shuffle
intrinsics allow direct register-to-register communication without shared memory:

__shfl_down_sync(mask, value, offset): thread i receives the value that thread i+offset
holds. Used to implement reduction in log2(32) = 5 steps.

Step 1 (offset=16): thread 0 gets thread 16's value -> holds max(v0, v16)
Step 2 (offset=8):  thread 0 gets thread 8's value  -> holds max(v0, v8, v16, v24)
Step 3 (offset=4):  ...
Step 5 (offset=1):  thread 0 holds max of all 32 values

5 instructions to reduce 32 values. ~1 cycle latency. No shared memory. No barrier.

This pattern appears verbatim in Flash Attention's inner loop as the online softmax
computation. In Phase 3, tl.max() and tl.sum() in Triton compile to these exact
instructions. The warp shuffle is the hardware primitive that makes fused attention
kernels computationally feasible.

---

## Developer Kernel Design Workflow

When a developer sits down to write a new kernel, Phase 2 defines the workflow:

1. Calculate arithmetic intensity first. FLOPs / bytes. Find it on the roofline. Know
   whether the bottleneck will be memory or compute before writing a line.

2. Look for fusion opportunities. A standalone softmax has low arithmetic intensity.
   A fused kernel that keeps intermediate results in registers across multiple logical
   operations has much higher effective intensity. This is why Flash Attention exists.

3. Tile explicitly. Do not rely on L2 cache to absorb redundant loads — it works at
   small sizes and collapses at production sizes. Choose tile size to maximize reuse
   within the 128KB shared memory budget per SM.

4. Enforce coalesced access. Design thread-to-data mapping so consecutive threadIdx.x
   values access consecutive memory addresses. Verify with ncu.

5. Profile at multiple scales. L2 cache effects mask real bottlenecks at small N. The
   kernel that looks fine at N=1024 may be a disaster at N=4096. Always benchmark at
   the actual production problem size.

---

## GTM Insights

### "GPU utilization" is a meaningless metric without context

Naive matmul measured 97% SM throughput while using 0.17% of compute capability. Vec add
measured 8.82% SM throughput while running at maximum efficiency for its workload. Both
are "low GPU utilization" stories depending on who is measuring and what they mean. A
customer saying their GPU utilization is low could be describing either situation, and the
interventions are completely different. The question to ask is: are you compute-bound or
memory-bound, and at what arithmetic intensity? That question separates the conversation
from hardware spec recitation to workload diagnosis.

### The cuBLAS gap is the software moat — and it's what alternative vendors are attacking

It took decades of engineering to reach 54 TFLOPS on a routine FP32 matmul. That software
depth is a primary reason customers stay on NVIDIA even when competing hardware has
comparable raw specs. When any alternative vendor pitches their chip, the sophisticated
buyer immediately asks: what is your software stack maturity? They are asking whether they
will be stuck writing naive kernels or whether they get something equivalent to cuBLAS out
of the box. Understanding that the gap between a tiled kernel and cuBLAS is pure software
engineering — not physics — means you can have a credible conversation about what it
actually costs to close it and why a customer migration timeline must account for it.

### SRAM vs HBM is the architectural choice that defines every non-NVIDIA chip

Every result from Phase 2 points to the same conclusion: fast, on-chip SRAM is the scarce
resource that determines kernel performance. HBM is abundant and relatively slow. NVIDIA's
answer has been to make HBM faster (H100: 3.35 TB/s vs 4090: 1 TB/s) while keeping SRAM
small per SM. Tenstorrent's answer is to dramatically increase the SRAM-to-compute ratio
and move data through a programmable mesh of cores rather than a cache hierarchy.
Cerebras's answer is to make the entire chip one giant SRAM. These are not marketing
differences — they are fundamentally different positions on the memory hierarchy tradeoff
that Phase 2 makes concrete. When a customer asks why they would buy alternative hardware
instead of an H100, the answer is not a spec comparison. It is an explanation of where
their specific workload sits on the roofline and which memory architecture fits it.

### The workload fit question is the real sales conversation

Raw TFLOPS is the wrong unit of comparison for almost every real purchasing decision. The
right question is: at what arithmetic intensity does the customer's workload actually
operate? A workload at 2 FLOP/byte gets the same throughput from a chip with half the
TFLOPS but double the memory bandwidth. A workload running Flash Attention-style tiled
kernels at 50+ FLOP/byte genuinely needs peak TFLOPS. Most enterprise AI infrastructure
sales conversations never reach this question. The person who can ask it — and then
connect the answer to why a specific architecture fits or does not — is selling outcomes
instead of hardware.

---

*Phase 2 complete. Exercises: vec_add, naive matmul, tiled matmul, softmax kernel.*
*Profiled with ncu. Benchmark table filled with real measured numbers.*
*Next: Phase 3 — Triton and Flash Attention.*
