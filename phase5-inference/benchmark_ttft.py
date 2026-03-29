import requests
import time
import sys
import json

# ============================================================
# TTFT and Throughput Benchmark
# Designed to produce results comparable to Groq LPU benchmarks
# Methodology: 100 token prompt, 200 token output, single request, 5 runs
#
# Metrics:
#   TTFT (Time to First Token): measured via streaming
#   Decode throughput: tokens generated / decode time
#   Total latency: wall clock from request to final token
# ============================================================

url = "http://localhost:8000/v1/completions"
model = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/Llama-3.1-8B-Instruct"

# Fixed 100-token prompt — verified via tokenizer
# This Llama-tokenized prompt is exactly 100 tokens
PROMPT = (
    "The history of artificial intelligence begins in antiquity, with myths, stories, "
    "and rumors of artificial beings endowed with intelligence or consciousness by master "
    "craftsmen. The seeds of modern AI were planted by philosophers who attempted to "
    "describe the process of human thinking as the mechanical manipulation of symbols. "
    "This work culminated in the invention of the programmable digital computer in the "
    "1940s, a machine based on abstract mathematical reasoning. This device and the ideas "
    "behind it inspired a handful of scientists to begin seriously discussing the "
    "possibility of building an electronic brain. The field of AI research was founded "
    "at a workshop held on the campus of Dartmouth College"
)

NUM_RUNS = 5
MAX_TOKENS = 200

def benchmark_single_run(run_num):
    """
    Run one benchmark iteration using streaming to capture TTFT.
    Streaming means the server sends tokens as they are generated
    rather than waiting for the complete response. This lets us
    measure exactly when the first token arrives vs when generation completes.
    """
    ttft = None
    total_tokens = 0
    first_token_time = None

    request_start = time.perf_counter()

    # stream=True sends tokens incrementally as they are generated
    response = requests.post(url, json={
        "model": model,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,      # greedy decoding — deterministic, cleaner benchmarks
        "stream": True         # critical — enables TTFT measurement
    }, stream=True)

    for line in response.iter_lines():
        if not line:
            continue

        # Strip "data: " prefix from SSE stream format
        line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data_str = line[6:]
        if data_str == "[DONE]":
            break

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # First chunk received = first token generated = TTFT
        if first_token_time is None:
            first_token_time = time.perf_counter()
            ttft = (first_token_time - request_start) * 1000  # convert to ms

        # Count tokens in this chunk
        choices = data.get("choices", [])
        for choice in choices:
            text = choice.get("text", "")
            if text:
                total_tokens += 1

    total_latency = (time.perf_counter() - request_start) * 1000  # ms

    # Decode throughput = tokens after first / time after first token
    # TTFT covers prefill + first token
    # Remaining time covers pure decode
    decode_time_sec = (total_latency - ttft) / 1000
    decode_throughput = (total_tokens - 1) / decode_time_sec if decode_time_sec > 0 else 0

    print(f"  Run {run_num}: TTFT={ttft:.1f}ms | "
          f"Throughput={decode_throughput:.1f} tok/s | "
          f"Total={total_latency:.0f}ms | "
          f"Tokens={total_tokens}")

    return ttft, decode_throughput, total_latency, total_tokens

print(f"\nBenchmarking: {model}")
print(f"Prompt tokens: ~100 | Output tokens: {MAX_TOKENS} | Runs: {NUM_RUNS}")
print(f"{'='*70}")

# Warmup run — not counted in results
# Allows CUDA graphs and caches to initialize
print("Warmup run (not counted)...")
benchmark_single_run(0)
print()

# Timed runs
results = []
for i in range(1, NUM_RUNS + 1):
    r = benchmark_single_run(i)
    results.append(r)

# Calculate means
mean_ttft = sum(r[0] for r in results) / NUM_RUNS
mean_throughput = sum(r[1] for r in results) / NUM_RUNS
mean_latency = sum(r[2] for r in results) / NUM_RUNS
mean_tokens = sum(r[3] for r in results) / NUM_RUNS

print(f"\n{'='*70}")
print(f"RESULTS ({NUM_RUNS} run mean):")
print(f"  TTFT:           {mean_ttft:.1f} ms")
print(f"  Decode tok/s:   {mean_throughput:.1f}")
print(f"  Total latency:  {mean_latency:.0f} ms")
print(f"  Avg tokens out: {mean_tokens:.0f}")
print(f"\nResult row for comparison table:")
print(f"| {model.split('/')[-1][:20]:<20} | BF16 | {mean_ttft:>9.1f} | "
      f"{mean_throughput:>5.1f} | {mean_latency:>18.0f} | A100 SXM4 |")
