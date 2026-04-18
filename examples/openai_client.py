#!/usr/bin/env python3
"""Minimal OpenAI-compatible smoke test for the deployed Qwen3.6 + DFlash server.

Usage:
    python3 openai_client.py
    python3 openai_client.py --prompt "Write the FFT in Python"
    python3 openai_client.py --base-url http://my-spark.local:8000/v1
    python3 openai_client.py --bench    # rough tok/s measurement
"""
import argparse
import time
import sys

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai", file=sys.stderr)
    sys.exit(1)


def chat(client, model, prompt, max_tokens=512, stream=True):
    t0 = time.perf_counter()
    first_token_t = None
    n_tokens = 0
    full_text = []
    full_reasoning = []

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=stream,
    )

    if stream:
        for chunk in response:
            if first_token_t is None:
                first_token_t = time.perf_counter()
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            reasoning = (
                getattr(delta, "reasoning_content", None)
                or getattr(delta, "reasoning", None)
                or getattr(delta, "reasoning_text", None)
            )
            if content:
                full_text.append(content)
                n_tokens += 1
            if reasoning:
                full_reasoning.append(reasoning)
        elapsed = time.perf_counter() - t0
        ttft = first_token_t - t0 if first_token_t else 0
        decode_t = elapsed - ttft
        decode_tps = (n_tokens / decode_t) if decode_t > 0 else 0
        return {
            "content": "".join(full_text),
            "reasoning": "".join(full_reasoning),
            "n_tokens": n_tokens,
            "ttft_s": ttft,
            "decode_tps": decode_tps,
            "total_s": elapsed,
        }
    else:
        choice = response.choices[0]
        elapsed = time.perf_counter() - t0
        return {
            "content": choice.message.content,
            "reasoning": getattr(choice.message, "reasoning_content", "") or "",
            "n_tokens": response.usage.completion_tokens,
            "ttft_s": None,
            "decode_tps": response.usage.completion_tokens / elapsed,
            "total_s": elapsed,
        }


def bench(client, model):
    """Standard benchmark suite: 5 prompts × 3 runs each."""
    prompts = [
        ("math", "Compute 47 × 89 step by step. Show all work."),
        ("code", "Write an iterative implementation of binary search in Python with comments."),
        ("chat", "Explain the difference between TCP and UDP in 3 paragraphs."),
        ("reason", "Solve: A man has 17 sheep. All but 9 die. How many are left?"),
        ("long", "Summarize the major themes of '1984' by George Orwell in detail."),
    ]
    results = {}
    for tag, prompt in prompts:
        runs = []
        for i in range(3):
            r = chat(client, model, prompt, max_tokens=512)
            runs.append(r)
            print(f"  [{tag} run {i+1}] {r['decode_tps']:.1f} tok/s "
                  f"({r['n_tokens']} tok in {r['total_s']:.1f}s, ttft={r['ttft_s']:.2f}s)")
        avg_tps = sum(r["decode_tps"] for r in runs) / len(runs)
        results[tag] = avg_tps
        print(f"  [{tag}] avg: {avg_tps:.1f} tok/s")
        print()

    print("=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)
    for tag, tps in results.items():
        print(f"  {tag:8s}  {tps:6.1f} tok/s")
    overall = sum(results.values()) / len(results)
    print(f"  {'overall':8s}  {overall:6.1f} tok/s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="qwen36-35b-heretic")
    p.add_argument("--prompt", default="What is 17 × 23? Show your work.")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--bench", action="store_true", help="Run standard benchmark suite")
    args = p.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    if args.bench:
        bench(client, args.model)
        return

    print(f"→ {args.prompt}")
    print()
    r = chat(client, args.model, args.prompt, args.max_tokens, stream=not args.no_stream)

    if r["reasoning"]:
        print("=== REASONING ===")
        print(r["reasoning"])
        print()
    print("=== ANSWER ===")
    print(r["content"])
    print()
    print(f"--- {r['n_tokens']} tokens, {r['total_s']:.2f}s, "
          f"{r['decode_tps']:.1f} tok/s decode"
          + (f", ttft={r['ttft_s']:.2f}s" if r['ttft_s'] is not None else ""))


if __name__ == "__main__":
    main()
