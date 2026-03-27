import requests
import time
import sys

url = "http://localhost:8000/v1/completions"
model = sys.argv[1] if len(sys.argv) > 1 else "/workspace/models/Mistral-7B-Instruct-v0.3"
prompt = "Explain how a transformer neural network works in detail:"

# Warmup
requests.post(url, json={"model": model, "prompt": prompt, "max_tokens": 100, "temperature": 0})

# Benchmark
start = time.perf_counter()
resp = requests.post(url, json={
    "model": model,
    "prompt": prompt,
    "max_tokens": 200,
    "temperature": 0
})
elapsed = time.perf_counter() - start

data = resp.json()
tokens = data["usage"]["completion_tokens"]
prompt_tokens = data["usage"]["prompt_tokens"]

print(f"Model:             {model}")
print(f"Prompt tokens:     {prompt_tokens}")
print(f"Completion tokens: {tokens}")
print(f"Total time:        {elapsed*1000:.0f}ms")
print(f"Throughput:        {tokens/elapsed:.1f} tokens/sec")
print(f"\nOutput:\n{data['choices'][0]['text']}")
