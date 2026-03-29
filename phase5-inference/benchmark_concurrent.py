import requests
import time
import threading
import sys

# Accept model path as command line argument
# Falls back to Mistral if no argument provided
# This lets us reuse the same script across models for clean comparisons
url = "http://localhost:8000/v1/completions"
model = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/Mistral-7B-Instruct-v0.3"
prompt = "Explain how a transformer neural network works in detail:"

def single_request(results, idx):
    # Each thread runs this function independently
    # Measures wall-clock time for one complete request — prompt + 200 token generation
    # Results stored by index so the main thread can aggregate after all threads finish
    start = time.perf_counter()
    resp = requests.post(url, json={
        "model": model,
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0  # greedy decoding — deterministic output, cleaner benchmarking
    })
    elapsed = time.perf_counter() - start
    tokens = resp.json()["usage"]["completion_tokens"]
    results[idx] = {"tokens": tokens, "elapsed": elapsed}

# Test four concurrency levels: 1, 2, 4, 8 simultaneous requests
# At concurrency=1: baseline, no batching benefit
# At concurrency=8: vLLM batches all requests into shared forward passes
# If continuous batching works correctly, throughput should scale near-linearly
# while per-request latency stays flat — the GPU was underutilized at concurrency=1
for concurrency in [1, 2, 4, 8]:
    results = [None] * concurrency

    # Create one thread per concurrent request
    # All threads start at roughly the same time, simulating simultaneous user arrivals
    # vLLM's scheduler sees all requests and batches their decode steps together
    threads = [threading.Thread(target=single_request, args=(results, i))
               for i in range(concurrency)]

    # Start all threads, then wait for all to finish before measuring
    # Wall-clock elapsed covers the full duration of the batch
    start = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    total_elapsed = time.perf_counter() - start

    # Total throughput = all tokens generated across all requests / wall-clock time
    # This is the GPU-level efficiency metric — how many tokens per second total
    total_tokens = sum(r["tokens"] for r in results)

    # Average latency = mean time each individual request waited for its response
    # This is the user-level experience metric — how long did I wait?
    avg_latency = sum(r["elapsed"] for r in results) / concurrency

    print(f"Concurrency: {concurrency:2d} | "
          f"Total throughput: {total_tokens/total_elapsed:6.1f} tok/s | "
          f"Avg latency: {avg_latency*1000:.0f}ms")
