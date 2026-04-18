#!/usr/bin/env python3
"""Concurrency-sweep benchmark for Qwen3.6 + DFlash.

Sweeps concurrent sequences (1, 2, 4, 8, 16, 32, 64, 128) and measures:
  - per-request decode tok/s
  - aggregate decode tok/s across all concurrent sequences
  - TTFT (time to first token)
  - DFlash acceptance rate (from server-side metrics if exposed)

Usage:
  python3 bench_concurrency.py                              # default: 5 prompts × 8 concurrency levels
  python3 bench_concurrency.py --max-tokens 256 --runs 3
  python3 bench_concurrency.py --base-url http://my-spark:8000/v1
"""
import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


PROMPTS = [
    "Compute 47 × 89 step by step. Show all work.",
    "Write an iterative implementation of binary search in Python with comments.",
    "Explain the difference between TCP and UDP in 3 paragraphs.",
    "Solve: A man has 17 sheep. All but 9 die. How many are left? Explain.",
    "List 5 properties of prime numbers and why each is important.",
]


def one_request(client, model, prompt, max_tokens):
    """Use non-streaming so we get the authoritative token count from the server.
    Also captures TTFT via a separate streaming probe (fast, single token)."""
    # TTFT probe (1 token streaming) — same temperature as the real request
    t0 = time.perf_counter()
    first_token_t = None
    probe = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0,  # match real request — DFlash needs consistent sampling
        stream=True,
    )
    for chunk in probe:
        if first_token_t is None:
            first_token_t = time.perf_counter()
            break
    ttft = (first_token_t - t0) if first_token_t else None

    # Full request (non-streaming) — gets accurate completion_tokens count
    t1 = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,  # greedy — DFlash acceptance requires same sampling on target+drafter
        stream=False,
    )
    t_end = time.perf_counter()
    n_tokens = response.usage.completion_tokens if response.usage else 0
    return {
        "n_tokens": n_tokens,
        "ttft": ttft,
        "decode_s": t_end - t1,
        "wall_s": t_end - t0,
    }


def run_concurrency(client, model, concurrency, max_tokens, runs):
    """Run `concurrency` parallel requests, repeat `runs` times. Return aggregated stats."""
    all_metrics = []
    for run_i in range(runs):
        prompts_for_run = (PROMPTS * ((concurrency // len(PROMPTS)) + 1))[:concurrency]
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = [
                ex.submit(one_request, client, model, p, max_tokens)
                for p in prompts_for_run
            ]
            results = [f.result() for f in as_completed(futures)]
        t_total = time.perf_counter() - t_start
        all_metrics.extend(results)
        total_tokens = sum(r["n_tokens"] for r in results)
        agg_tps = total_tokens / t_total if t_total > 0 else 0
        per_req_tps = sum(
            r["n_tokens"] / r["decode_s"] for r in results if r["decode_s"] > 0
        ) / len(results)
        avg_ttft = sum(r["ttft"] for r in results if r["ttft"]) / len(results)
        print(
            f"  run {run_i+1}/{runs}: agg={agg_tps:6.1f} tok/s, "
            f"per-req={per_req_tps:5.1f} tok/s, "
            f"ttft={avg_ttft*1000:6.0f} ms, "
            f"total={total_tokens:5d} tok in {t_total:5.1f}s"
        )

    n = len(all_metrics)
    total_tokens = sum(r["n_tokens"] for r in all_metrics)
    avg_per_req = sum(
        r["n_tokens"] / r["decode_s"] for r in all_metrics if r["decode_s"] > 0
    ) / n
    avg_ttft = sum(r["ttft"] for r in all_metrics if r["ttft"]) / n
    return {
        "concurrency": concurrency,
        "n_requests": n,
        "total_tokens": total_tokens,
        "avg_per_req_tps": avg_per_req,
        "avg_ttft_ms": avg_ttft * 1000,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="qwen36-35b-heretic")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--runs", type=int, default=2, help="runs per concurrency level")
    p.add_argument(
        "--levels",
        type=str,
        default="1,2,4,8,16,32,64,128",
        help="comma-separated concurrency levels",
    )
    args = p.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    # Warmup
    print("=== warmup ===")
    one_request(client, args.model, "Hello", 16)
    print("warmup done\n")

    levels = [int(x) for x in args.levels.split(",")]
    results = []
    for c in levels:
        print(f"=== concurrency={c} ===")
        r = run_concurrency(client, args.model, c, args.max_tokens, args.runs)
        results.append(r)
        print()

    # Summary
    print("=" * 75)
    print(f"{'concurrency':>11s}  {'agg_tps':>10s}  {'per_req_tps':>12s}  {'ttft_ms':>9s}  {'tokens':>8s}")
    print("=" * 75)
    for r in results:
        agg_tps = r["avg_per_req_tps"] * r["concurrency"]
        print(
            f"{r['concurrency']:>11d}  {agg_tps:>10.1f}  "
            f"{r['avg_per_req_tps']:>12.1f}  {r['avg_ttft_ms']:>9.0f}  "
            f"{r['total_tokens']:>8d}"
        )


if __name__ == "__main__":
    main()
