# Distributed Training & Interconnects

## The problem data parallelism solves

Training a large model on a single GPU has two hard limits: it's too slow, and the
model may not fit in memory. Data parallelism addresses the speed problem by splitting
the training batch across N GPUs. Each GPU runs the full model on a different slice of
data simultaneously — an embarrassingly parallel forward and backward pass with zero
communication required.

The catch: each GPU computes gradients from different data. Those gradients must be
averaged before any optimizer step, or the model copies drift apart and training
becomes mathematically meaningless.

## AllReduce — what it does and why it's designed the way it is

AllReduce is the collective operation that averages gradients across all ranks.
Every rank contributes a tensor, every rank receives the element-wise sum (or mean).

The naive approach — send everything to rank 0, average there, broadcast back —
makes rank 0 a bottleneck. It receives N tensors worth of data and sends N tensors
back. Every other GPU waits.

Ring AllReduce eliminates the bottleneck by distributing the work equally. GPUs are
arranged in a logical ring. The algorithm runs in two phases:

**Phase 1 — Reduce-Scatter (N-1 steps)**
Each GPU sends one chunk forward and receives one chunk from behind, accumulating
(summing) received values into its local copy. The send index rotates each step so
every chunk visits every GPU exactly once. After N-1 steps, each GPU holds one
fully-summed chunk — the complete sum of that chunk position across all ranks.

**Phase 2 — AllGather (N-1 steps)**
Each GPU broadcasts its fully-summed chunk around the ring. No accumulation —
just overwrite. After N-1 steps, every GPU holds all fully-summed chunks and
therefore the complete averaged gradient tensor.

Bus bandwidth per GPU: 2 × (N-1)/N × tensor_size / time
At N=4: factor = 1.5. Adding more GPUs doesn't increase per-GPU communication
volume — it approaches 2 × tensor_size asymptotically.

## The synchronization barrier

AllReduce is a collective — every rank must participate before any rank receives
the result. This makes it a hard synchronization barrier. The slowest rank
determines when everyone proceeds. At 1000 GPUs, one straggler taxes all 999 others.
This is why hardware uniformity and load balancing matter at scale.

## Measured bandwidth vs. spec

Benchmark results on 4× A100 SXM (NV12 NVLink) on RunPod:

| Size   | Time (ms) | BW (GB/s) | % of 600 GB/s spec |
|--------|-----------|-----------|-------------------|
| 1 MB   | 0.051     | 31.0      | 5.2%              |
| 16 MB  | 0.202     | 124.3     | 20.7%             |
| 64 MB  | 0.596     | 168.8     | 28.1%             |
| 256 MB | 2.092     | 192.5     | 32.1%             |
| 512 MB | 3.817     | 211.0     | 35.2%             |
| 1 GB   | 7.244     | 222.3     | 37.1%             |
| 2 GB   | 14.152    | 227.6     | 37.9%             |

Peak measured: 228 GB/s (38% of 600 GB/s spec).

The curve shape is the key result. Small tensors are latency-dominated — fixed NCCL
handshake and kernel launch overhead dominates transfer time. Large tensors are
bandwidth-saturated — fixed overhead becomes negligible and throughput approaches
the hardware ceiling. The curve is still rising at 2GB, meaning NCCL overhead
remains non-trivial even at that size.

The 38% vs. spec gap has three causes:
1. RunPod virtualization adds latency on every NCCL operation
2. AllReduce involves synchronization barriers that raw point-to-point benchmarks
   don't pay
3. NCCL optimizes for correctness and generality, not peak benchmark numbers

On bare-metal A100 SXM with dedicated NVLink, 70-85% of spec is typical at
large tensor sizes.

## Why interconnect dominates at scale

The roofline model from Phase 1 applies here — just one level up in the hierarchy.
Instead of compute vs. memory bandwidth on one chip, the constraint is compute vs.
interconnect bandwidth across chips.

A 7B parameter model in BF16 produces ~14GB of gradients per step. At 228 GB/s
measured bandwidth, AllReduce takes ~90ms. A forward+backward pass on 4× A100s
takes ~300ms. Communication is ~23% of step time — significant but not dominant.

At 1000 GPUs the math changes. More ranks = more ring steps = more synchronization
= communication starts to dominate compute. At 10,000 GPUs, cluster network topology
(rack-to-rack InfiniBand, custom interconnect) becomes the primary design constraint
of the entire system.

The same principle from Phases 1-3 applies at every level:
minimize expensive data movement by keeping computation and data as close together
as possible. Registers → SRAM → HBM → NVLink → InfiniBand. Each level is orders
of magnitude slower and more expensive than the one below it.

## The real efficiency metric: Model FLOP Utilization (MFU)

GPU utilization reported by nvidia-smi is misleading in distributed training.
A cluster at 80% GPU utilization might be spending 40% of that time waiting on
AllReduce barriers. The meaningful metric is Model FLOP Utilization (MFU):

  MFU = actual useful compute / theoretical peak compute

Most production training runs achieve 30-50% MFU. The gap between 100% and 50%
is almost entirely communication overhead, memory bandwidth limits, and pipeline
bubbles. Understanding why MFU degrades requires understanding everything in
Phases 1-4. Teams that only watch GPU utilization are measuring the wrong thing.

## Hardware spec numbers are ceilings, not floors

The A100 SXM NVLink spec is 600 GB/s. We measured 228 GB/s — 38% of spec.
That gap is not a failure. It is the real world.

Virtualization, synchronization overhead, algorithm overhead, and straggler effects
all eat into the theoretical maximum. Every spec sheet in AI infrastructure needs
to be read this way: this is the ceiling under ideal conditions you will never
fully achieve in production. The relevant question is never "what is the peak?"
It is "what fraction of peak can I sustain, and what is eating the rest?"

---

## Business, operational, and GTM implications

### For technology leaders and engineering teams

**Interconnect is infrastructure, not a feature.**
When an engineering team says "we need more GPUs," the real question is "what
interconnect topology?" Eight A100s on PCIe is a fundamentally different machine
than eight A100s on NVLink — not incrementally worse, categorically worse for
training workloads. A CTO or VP of infrastructure who doesn't understand this
distinction will make expensive procurement mistakes.

**The cluster design decision happens once and is hard to reverse.**
You commit to a network topology when you buy the hardware. If you spec PCIe-only
nodes because they're cheaper and then need to train a 70B model, you don't
upgrade — you replace. This is a $10M+ decision disguised as a procurement
question.

**Communication overhead is the hidden cost of scale.**
Teams routinely budget for GPU compute and underbudget for networking. In a large
training run, the InfiniBand fabric, the spine switches, and the topology design
can cost as much as the GPUs themselves. The companies that learned this early —
Google with TPU pods, Meta with RoCE fabric, Microsoft with InfiniBand — built
custom networking specifically because commodity networking couldn't keep pace
with compute scaling.

**Data parallelism hits a wall at large model sizes.**
Data parallelism works until the model doesn't fit on a single GPU — which happens
at 70B+ parameters. Beyond that you need tensor parallelism or pipeline parallelism,
both of which have even more demanding communication patterns. Engineering teams
that only understand data parallelism hit a hard ceiling when they try to scale
further. Understanding the full parallelism stack is a prerequisite for serious
large-scale training.

**Operational implication: measure MFU, not GPU utilization.**
A cluster running at 80% GPU utilization might be spending 40% of that time
waiting on AllReduce. Optimizing GPU utilization without understanding the
communication profile is optimizing the wrong metric. The teams winning at
large-scale training are the ones who understand the full system — compute,
memory, and interconnect — simultaneously.

### For vendors and GTM

**NVLink is NVIDIA's most durable moat — more than the GPU itself.**
AMD can build a competitive GPU (the MI300X is genuinely excellent on memory
capacity and bandwidth). What AMD cannot replicate quickly is the NVLink ecosystem —
the switch fabric, the topology, and a software stack that has been co-optimized
with NCCL for over a decade. When a hyperscaler buys H100s, they are not just
buying the compute — they are buying the interconnect architecture. This is a
primary reason NVIDIA's margins are what they are.

**The interconnect creates vendor lock-in at the infrastructure layer.**
Once a training cluster is built around NVLink topology, migration is a forklift
replacement. The software (NCCL), the topology (NVSwitch), and the tooling
(Nsight, nvitop) are all NVIDIA-specific. Competitors must match not just GPU
performance but the entire interconnect ecosystem to be a viable alternative
for large-scale training. This is why Tenstorrent, Groq, and others are making
different architectural bets — they cannot out-NVLink NVIDIA on NVIDIA's terms,
so they are competing on different axes entirely.

**For cloud vendors, interconnect determines which workloads you can sell.**
AWS, Azure, and GCP all offer GPU instances. But not all GPU instances can run
large-scale distributed training efficiently. Hyperscalers that invested in
high-bandwidth cluster networking — Azure NDv5 with InfiniBand, Google TPU pods
with ICI — can sell training workloads that commodity GPU clouds cannot. This
creates significant revenue segmentation: inference is increasingly a commodity,
large-scale training is not.

**The buyer's real pain is not "I need more compute."**
It is "my training runs are slower than they should be and I don't know why."
The answer is almost always interconnect bandwidth, communication overhead, or
MFU degradation at scale. Any vendor who can walk into that conversation with
real numbers — the kind generated in this phase — and explain the gap between
spec and measured performance has a fundamentally different conversation than
one selling spec sheets. Technical credibility at this level of specificity is
a genuine sales differentiator in the AI infrastructure market.

**The distributed systems parallel.**
Distributed databases and distributed training share deep structural similarities.
AllReduce is a consensus problem — every node must agree on the same value before
proceeding. This is structurally analogous to distributed transaction coordination.
The communication overhead, the straggler problem, the topology constraints — these
are familiar problems in a new domain. For anyone coming from the distributed
data systems world, this is not foreign territory. The vocabulary is different;
the engineering tradeoffs are the same.
