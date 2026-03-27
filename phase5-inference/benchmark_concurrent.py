import requests
import time
import threading

url = "http://localhost:8000/v1/completions"
model = "/workspace/models/Mistral-7B-Instruct-v0.3"
prompt = "Explain how a transformer neural network works in detail:"

def single_request(results, idx):
    start = time.perf_counter()
    resp = requests.post(url, json={
        "model": model,
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0
    })
    elapsed = time.perf_counter() - start
    tokens = resp.json()["usage"]["completion_tokens"]
    results[idx] = {"tokens": tokens, "elapsed": elapsed}

for concurrency in [1, 2, 4, 8]:
    results = [None] * concurrency
    threads = [threading.Thread(target=single_request, args=(results, i))
               for i in range(concurrency)]

    start = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    total_elapsed = time.perf_counter() - start

    total_tokens = sum(r["tokens"] for r in results)
    avg_latency = sum(r["elapsed"] for r in results) / concurrency

    print(f"Concurrency: {concurrency:2d} | "
          f"Total throughput: {total_tokens/total_elapsed:6.1f} tok/s | "
          f"Avg latency: {avg_latency*1000:.0f}ms")
