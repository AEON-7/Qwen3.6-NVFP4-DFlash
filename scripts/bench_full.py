#!/usr/bin/env python3
"""Comprehensive Qwen3.6-35B-A3B-heretic + DFlash benchmark on DGX Spark.

Mirrors the supergemma4-26b benchmark methodology — produces a multi-section
markdown report you can paste directly into the HF model card.

Sections:
  1. Single-stream decode (N trials, p50/p95/min/max)
  2. TTFT by prompt length (tiny/short/medium/long)
  3. Decode rate by output length (50/200/500/1000 max tokens)
  4. Sampling: greedy vs stochastic comparison
  5. Long-prompt prefill (1K/4K/16K/32K input)
  6. Concurrent throughput sweep (1...256, agg / per-req p50,min / TTFT p50,p95,max / errors)
  7. TTFT-only scaling (1/4/16/64/256 with 1-token output)
  8. RAG-style concurrent (1K input + 50-token output at 1/4/16/64)

Usage:
  # Quick smoke (sections 1, 6 limited)
  python3 bench_full.py --quick

  # Full bench
  python3 bench_full.py --output bench_results.json

  # Pick sections
  python3 bench_full.py --sections single,concurrent --max-concurrency 64
"""
import argparse
import asyncio
import json
import random
import statistics
import string
import time
from datetime import datetime, timezone

import httpx

DEFAULT_BASE = "http://localhost:8000/v1"
DEFAULT_MODEL = "qwen36-fast"   # greedy alias by convention

# Mixed-domain prompts (code, math, QA, reasoning, creative)
PROMPTS_MIXED = [
    "Write an iterative Python implementation of binary search with comments.",
    "Compute 47 × 89 step by step and show all work.",
    "Explain the difference between TCP and UDP in three paragraphs.",
    "A man has 17 sheep. All but 9 die. How many are left? Explain.",
    "List 5 properties of prime numbers and why each is important.",
    "Write a 4-line haiku about autumn wind through pine trees.",
    "Describe the architecture of a transformer encoder layer.",
    "What's the shortest path from (0,0) to (5,3) on a grid moving only right or up?",
]

PROMPT_TINY = "Hi"                                   # ~2 tokens
PROMPT_SHORT = "What is the capital of France?"      # ~7 tokens
PROMPT_MEDIUM = (                                    # ~50 tokens
    "Write a brief explanation of how transformer self-attention works, "
    "covering the role of queries, keys, and values, and how the softmax-scaled "
    "dot-product produces attention weights. Keep it to two paragraphs."
)
# PROMPT_LONG built dynamically from filler


def filler_tokens(approx_tokens: int, seed: int = 0) -> str:
    """Build prompt that tokenizes to approximately `approx_tokens`."""
    random.seed(seed)
    n_chars = approx_tokens * 4
    words = [
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 9)))
        for _ in range(n_chars // 5)
    ]
    return " ".join(words)


async def stream_one(client, base, model, messages, max_tokens, temperature=0.0,
                     timeout=600.0, enable_thinking=False):
    """Stream one chat completion. Returns (ttft_s, n_tokens, total_s, error_str_or_None).

    By default disables Qwen3.6 thinking for clean decode-rate measurements (matches
    methodology used in supergemma4 bench). Pass enable_thinking=True to measure
    real production behavior including reasoning-token overhead.
    """
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    err = None
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
    }
    try:
        async with client.stream("POST", f"{base}/chat/completions",
                                  json=body, timeout=timeout) as resp:
            if resp.status_code != 200:
                err = f"http_{resp.status_code}"
                return None, 0, time.perf_counter() - t0, err
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                if obj.get("choices"):
                    delta = obj["choices"][0].get("delta", {}) or {}
                    # TTFT fires on first chunk with actual generated content
                    # (vLLM emits an empty role-only delta first; skip that)
                    if ttft is None:
                        c = delta.get("content") or ""
                        r = delta.get("reasoning") or delta.get("reasoning_content") or ""
                        if (c and c != "") or (r and r != ""):
                            ttft = time.perf_counter() - t0
                if obj.get("usage"):
                    tokens = obj["usage"].get("completion_tokens", tokens)
    except (httpx.HTTPError, asyncio.TimeoutError) as e:
        err = type(e).__name__
    total = time.perf_counter() - t0
    return ttft, tokens, total, err


def stats(xs):
    """Return (median, p95, min, max) for a numeric iterable, ignoring None."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, None, None
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    return (
        xs_sorted[n // 2],
        xs_sorted[min(int(n * 0.95), n - 1)],
        xs_sorted[0],
        xs_sorted[-1],
    )


# ── Section 1 — Single-stream decode (10 trials) ──────────────────────────────

async def bench_single_stream(base, model, n_trials=10, max_tokens=200,
                              temperature=0.0):
    print(f"\n── [1] Single-stream decode (greedy, max_tokens={max_tokens}, "
          f"n={n_trials}) ──")
    records = []
    async with httpx.AsyncClient() as client:
        # warmup outside the timing
        await stream_one(client, base, model,
                         [{"role": "user", "content": "hi"}], 8, 0.0)
        for i in range(n_trials):
            ttft, toks, total, err = await stream_one(
                client, base, model,
                [{"role": "user", "content": PROMPTS_MIXED[i % len(PROMPTS_MIXED)]}],
                max_tokens, temperature,
            )
            if err or not toks or not ttft:
                print(f"  trial {i+1}: ERR {err or 'no tokens'}")
                continue
            decode_s = total - ttft
            tps = toks / decode_s if decode_s > 0 else 0
            records.append({"ttft": ttft, "decode_tps": tps, "tokens": toks,
                           "total": total})
            print(f"  trial {i+1:2d}: TTFT {ttft*1000:5.0f} ms  "
                  f"decode {tps:6.1f} tok/s  ({toks} tok in {total:.2f}s)")
    if not records:
        return {"section": "single_stream", "error": "all trials failed"}
    tps_p50, tps_p95, tps_min, tps_max = stats([r["decode_tps"] for r in records])
    ttft_p50, _, ttft_min, ttft_max = stats([r["ttft"] for r in records])
    print(f"  → median {tps_p50:.1f} tok/s, p95 {tps_p95:.1f}, "
          f"min {tps_min:.1f}, max {tps_max:.1f}")
    return {
        "section": "single_stream",
        "n_trials": len(records),
        "decode_tps_p50": tps_p50, "decode_tps_p95": tps_p95,
        "decode_tps_min": tps_min, "decode_tps_max": tps_max,
        "ttft_ms_p50": ttft_p50 * 1000, "ttft_ms_min": ttft_min * 1000,
        "ttft_ms_max": ttft_max * 1000,
    }


# ── Section 2 — TTFT by prompt length ────────────────────────────────────────

async def bench_ttft_by_prompt(base, model, n_trials=5):
    print(f"\n── [2] TTFT by prompt length (n={n_trials}/class) ──")
    classes = [
        ("tiny",   PROMPT_TINY,   2),
        ("short",  PROMPT_SHORT,  7),
        ("medium", PROMPT_MEDIUM, 50),
        ("long",   filler_tokens(450) +
                   "\n\nSummarize the above in one sentence.", 465),
    ]
    out = []
    async with httpx.AsyncClient() as client:
        for name, prompt, approx_in in classes:
            ttfts = []
            for _ in range(n_trials):
                ttft, _toks, _total, err = await stream_one(
                    client, base, model,
                    [{"role": "user", "content": prompt}],
                    16, 0.0,
                )
                if err or not ttft:
                    continue
                ttfts.append(ttft)
            p50, p95, mn, _mx = stats(ttfts)
            row = {
                "class": name,
                "approx_input_tokens": approx_in,
                "ttft_ms_p50": p50 * 1000 if p50 else None,
                "ttft_ms_p95": p95 * 1000 if p95 else None,
                "ttft_ms_min": mn * 1000 if mn else None,
                "effective_prefill_tps": (approx_in / p50) if p50 else None,
            }
            out.append(row)
            if p50 is None:
                print(f"  {name:7s} (~{approx_in:4d} tok): NO DATA (all trials failed)")
            else:
                print(f"  {name:7s} (~{approx_in:4d} tok): "
                      f"p50 {row['ttft_ms_p50']:6.0f} ms  "
                      f"p95 {row['ttft_ms_p95']:6.0f} ms  "
                      f"min {row['ttft_ms_min']:6.0f} ms  "
                      f"eff. prefill {row['effective_prefill_tps']:7.0f} tok/s")
    return {"section": "ttft_by_prompt", "rows": out}


# ── Section 3 — Decode rate by output length ─────────────────────────────────

async def bench_decode_by_output(base, model, n_trials=3):
    print(f"\n── [3] Decode rate by output length (n={n_trials}/length) ──")
    lengths = [50, 200, 500, 1000]
    rows = []
    async with httpx.AsyncClient() as client:
        for max_tok in lengths:
            tps_list, ttft_list, actual_list, total_list = [], [], [], []
            for i in range(n_trials):
                ttft, toks, total, err = await stream_one(
                    client, base, model,
                    [{"role": "user", "content":
                      PROMPTS_MIXED[i % len(PROMPTS_MIXED)]}],
                    max_tok, 0.0,
                )
                if err or not toks or not ttft:
                    continue
                decode_s = total - ttft
                if decode_s <= 0:
                    continue
                tps_list.append(toks / decode_s)
                ttft_list.append(ttft)
                actual_list.append(toks)
                total_list.append(total)
            if not tps_list:
                rows.append({"max_tokens": max_tok, "error": "no successful trials"})
                continue
            tps_p50 = statistics.median(tps_list)
            ttft_p50 = statistics.median(ttft_list) * 1000
            actual_p50 = int(statistics.median(actual_list))
            total_p50 = statistics.median(total_list)
            rows.append({
                "max_tokens": max_tok,
                "actual_tokens_p50": actual_p50,
                "ttft_ms_p50": ttft_p50,
                "decode_tps_p50": tps_p50,
                "total_s_p50": total_p50,
            })
            print(f"  max={max_tok:4d}  actual={actual_p50:4d}  "
                  f"TTFT={ttft_p50:5.0f} ms  decode={tps_p50:5.1f} tok/s  "
                  f"total={total_p50:.2f}s")
    return {"section": "decode_by_output", "rows": rows}


# ── Section 4 — Sampling: greedy vs stochastic ───────────────────────────────

async def bench_sampling(base, model, n_trials=5, max_tokens=200):
    print(f"\n── [4] Sampling: greedy vs stochastic (n={n_trials}/mode) ──")
    results = []
    async with httpx.AsyncClient() as client:
        for label, temp in [("greedy_T0", 0.0), ("stochastic_T0.7", 0.7)]:
            tps_list, ttft_list = [], []
            for i in range(n_trials):
                ttft, toks, total, err = await stream_one(
                    client, base, model,
                    [{"role": "user", "content":
                      PROMPTS_MIXED[i % len(PROMPTS_MIXED)]}],
                    max_tokens, temp,
                )
                if err or not toks or not ttft:
                    continue
                decode_s = total - ttft
                if decode_s <= 0:
                    continue
                tps_list.append(toks / decode_s)
                ttft_list.append(ttft)
            tps_p50, tps_p95, _, _ = stats(tps_list)
            ttft_p50 = statistics.median(ttft_list) * 1000 if ttft_list else None
            row = {
                "mode": label, "temperature": temp,
                "decode_tps_p50": tps_p50, "decode_tps_p95": tps_p95,
                "ttft_ms_p50": ttft_p50,
            }
            results.append(row)
            print(f"  {label:18s}: decode p50 {tps_p50:5.1f} tok/s  "
                  f"p95 {tps_p95:5.1f}  TTFT p50 {ttft_p50:5.0f} ms")
    return {"section": "sampling", "rows": results}


# ── Section 5 — Long-prompt prefill ──────────────────────────────────────────

async def bench_long_prefill(base, model, sizes=(1024, 4096, 16384, 32768)):
    print(f"\n── [5] Long-prompt prefill ──")
    rows = []
    async with httpx.AsyncClient() as client:
        for n_in in sizes:
            filler = filler_tokens(n_in - 50)
            messages = [
                {"role": "user", "content":
                 filler + "\n\nSummarize the above in one sentence."}
            ]
            ttft, toks, total, err = await stream_one(
                client, base, model, messages, 64, 0.0,
            )
            if err or not toks or not ttft:
                rows.append({"input_tokens": n_in, "error": err or "no tokens"})
                print(f"  in={n_in:5d}  ERR {err or 'no tokens'}")
                continue
            decode_tps = toks / (total - ttft) if total > ttft else 0
            prefill_tps = n_in / ttft if ttft > 0 else 0
            rows.append({
                "input_tokens_target": n_in,
                "actual_input_tokens_estimate": n_in,  # from filler
                "ttft_ms": ttft * 1000,
                "prefill_tps": prefill_tps,
                "decode_tps_after_prefill": decode_tps,
                "completion_tokens": toks,
            })
            print(f"  in={n_in:5d}  TTFT={ttft*1000:6.0f} ms  "
                  f"prefill={prefill_tps:7.0f} tok/s  "
                  f"decode={decode_tps:5.1f} tok/s  "
                  f"({toks} out)")
    return {"section": "long_prefill", "rows": rows}


# ── Section 6 — Concurrent throughput sweep ──────────────────────────────────

async def bench_concurrent(base, model, levels=(1, 2, 4, 8, 16, 32, 64, 128, 256),
                           max_tokens=200, temperature=0.7, n_runs=3):
    print(f"\n── [6] Concurrent throughput sweep (max_tokens={max_tokens}, "
          f"T={temperature}, n_runs={n_runs}) ──")
    rows = []
    async with httpx.AsyncClient(limits=httpx.Limits(
            max_connections=512, max_keepalive_connections=512)) as client:
        for n_conc in levels:
            run_metrics = []  # one entry per run
            for run_i in range(n_runs):
                prompts = [PROMPTS_MIXED[i % len(PROMPTS_MIXED)] for i in range(n_conc)]
                t_start = time.perf_counter()
                results = await asyncio.gather(*(
                    stream_one(client, base, model,
                               [{"role": "user", "content": p}],
                               max_tokens, temperature, timeout=900.0)
                    for p in prompts
                ))
                t_total = time.perf_counter() - t_start
                errors = sum(1 for (_t, _n, _w, e) in results if e)
                ok = [(t, n, w) for (t, n, w, e) in results
                      if not e and t is not None and n]
                if not ok:
                    print(f"  conc={n_conc:3d} run {run_i+1}: ALL FAILED")
                    continue
                total_tok = sum(n for (_t, n, _w) in ok)
                agg_tps = total_tok / t_total
                per_req_tps = [
                    n / (w - t) for (t, n, w) in ok if (w - t) > 0
                ]
                pr_p50, _pr_p95, pr_min, _pr_max = stats(per_req_tps)
                ttfts = [t for (t, _n, _w) in ok]
                tt_p50, tt_p95, tt_min, tt_max = stats(ttfts)
                run_metrics.append({
                    "agg_tps": agg_tps,
                    "per_req_p50": pr_p50, "per_req_min": pr_min,
                    "ttft_p50": tt_p50, "ttft_p95": tt_p95,
                    "ttft_min": tt_min, "ttft_max": tt_max,
                    "errors": errors, "ok_count": len(ok),
                    "wall_s": t_total,
                })
                print(f"  conc={n_conc:3d} run {run_i+1}: "
                      f"agg={agg_tps:7.1f} tok/s  "
                      f"per-req p50={pr_p50:5.1f} min={pr_min:5.1f}  "
                      f"TTFT p50={tt_p50*1000:5.0f} p95={tt_p95*1000:5.0f}  "
                      f"err={errors}/{n_conc}  wall={t_total:.1f}s")
            if not run_metrics:
                rows.append({"concurrency": n_conc, "error": "all runs failed"})
                continue
            # Pick median run by aggregate throughput
            run_metrics.sort(key=lambda m: m["agg_tps"])
            med = run_metrics[len(run_metrics) // 2]
            rows.append({
                "concurrency": n_conc,
                "errors_in_median_run": med["errors"],
                "agg_tps_median_of_runs": med["agg_tps"],
                "per_req_decode_tps_p50": med["per_req_p50"],
                "per_req_decode_tps_min": med["per_req_min"],
                "ttft_ms_p50": med["ttft_p50"] * 1000,
                "ttft_ms_p95": med["ttft_p95"] * 1000,
                "ttft_ms_min": med["ttft_min"] * 1000,
                "ttft_ms_max": med["ttft_max"] * 1000,
                "n_runs": len(run_metrics),
            })
    return {"section": "concurrent_throughput", "rows": rows,
            "params": {"max_tokens": max_tokens, "temperature": temperature,
                       "n_runs": n_runs}}


# ── Section 7 — TTFT-only scaling ────────────────────────────────────────────

async def bench_ttft_scaling(base, model, levels=(1, 4, 16, 64, 256), n_runs=3):
    print(f"\n── [7] TTFT-only scaling (1-token output, n={n_runs}) ──")
    rows = []
    async with httpx.AsyncClient(limits=httpx.Limits(
            max_connections=512, max_keepalive_connections=512)) as client:
        for n_conc in levels:
            ttfts_all = []
            for _ in range(n_runs):
                results = await asyncio.gather(*(
                    stream_one(client, base, model,
                               [{"role": "user", "content":
                                 PROMPTS_MIXED[i % len(PROMPTS_MIXED)]}],
                               1, 0.0, timeout=300.0)
                    for i in range(n_conc)
                ))
                ttfts_all.extend(t for (t, _n, _w, e) in results
                                 if not e and t is not None)
            p50, p95, mn, mx = stats(ttfts_all)
            row = {
                "concurrency": n_conc, "n_observations": len(ttfts_all),
                "ttft_ms_p50": p50 * 1000 if p50 else None,
                "ttft_ms_p95": p95 * 1000 if p95 else None,
                "ttft_ms_min": mn * 1000 if mn else None,
                "ttft_ms_max": mx * 1000 if mx else None,
            }
            rows.append(row)
            if p50 is None:
                print(f"  conc={n_conc:3d}: NO DATA (all requests failed)")
            else:
                print(f"  conc={n_conc:3d}: p50={row['ttft_ms_p50']:6.0f} ms  "
                      f"p95={row['ttft_ms_p95']:6.0f}  "
                      f"min={row['ttft_ms_min']:6.0f}  "
                      f"max={row['ttft_ms_max']:6.0f}")
    return {"section": "ttft_scaling", "rows": rows}


# ── Section 8 — RAG-style concurrent (1K input, 50-token output) ─────────────

async def bench_rag_concurrent(base, model, levels=(1, 4, 16, 64), n_runs=2):
    print(f"\n── [8] RAG-style concurrent (1K input, 50-token output, n={n_runs}) ──")
    rag_filler = filler_tokens(950)
    rag_question = "\n\nIn one sentence, summarize the topic of the above text."
    rag_prompt = rag_filler + rag_question
    rows = []
    async with httpx.AsyncClient(limits=httpx.Limits(
            max_connections=512, max_keepalive_connections=512)) as client:
        for n_conc in levels:
            run_metrics = []
            for _ in range(n_runs):
                t_start = time.perf_counter()
                results = await asyncio.gather(*(
                    stream_one(client, base, model,
                               [{"role": "user", "content": rag_prompt}],
                               50, 0.7, timeout=900.0)
                    for _ in range(n_conc)
                ))
                t_total = time.perf_counter() - t_start
                errors = sum(1 for (_t, _n, _w, e) in results if e)
                ok = [(t, n, w) for (t, n, w, e) in results
                      if not e and t is not None and n]
                if not ok:
                    continue
                total_tok = sum(n for (_t, n, _w) in ok)
                per_req = [n / (w - t) for (t, n, w) in ok if (w - t) > 0]
                ttfts = [t for (t, _n, _w) in ok]
                run_metrics.append({
                    "agg_tps": total_tok / t_total,
                    "ttft_p50": statistics.median(ttfts),
                    "ttft_p95": stats(ttfts)[1],
                    "decode_p50": statistics.median(per_req) if per_req else 0,
                    "errors": errors,
                })
            if not run_metrics:
                rows.append({"concurrency": n_conc, "error": "all runs failed"})
                continue
            med = sorted(run_metrics, key=lambda m: m["agg_tps"])[
                len(run_metrics) // 2]
            rows.append({
                "concurrency": n_conc,
                "errors_in_median_run": med["errors"],
                "agg_tps": med["agg_tps"],
                "ttft_ms_p50": med["ttft_p50"] * 1000,
                "ttft_ms_p95": med["ttft_p95"] * 1000,
                "decode_tps_p50": med["decode_p50"],
            })
            print(f"  conc={n_conc:3d}: agg={med['agg_tps']:6.1f} tok/s  "
                  f"TTFT p50={med['ttft_p50']*1000:5.0f} ms  "
                  f"p95={med['ttft_p95']*1000:5.0f}  "
                  f"decode p50={med['decode_p50']:5.1f}  "
                  f"err={med['errors']}/{n_conc}")
    return {"section": "rag_concurrent", "rows": rows}


# ── main ─────────────────────────────────────────────────────────────────────

ALL_SECTIONS = ["single", "ttft_by_prompt", "decode_by_output", "sampling",
                "long_prefill", "concurrent", "ttft_scaling", "rag"]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sections", default="all",
                    help=f"Comma-separated subset of {ALL_SECTIONS} or 'all'")
    ap.add_argument("--max-concurrency", type=int, default=256)
    ap.add_argument("--quick", action="store_true",
                    help="Smaller trials/levels for quick smoke test")
    ap.add_argument("--output", default=None,
                    help="Write JSON results to this path")
    args = ap.parse_args()

    sections = (ALL_SECTIONS if args.sections == "all"
                else [s.strip() for s in args.sections.split(",")])

    # Concurrency levels
    full_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    levels = [c for c in full_levels if c <= args.max_concurrency]
    ttft_levels = [c for c in [1, 4, 16, 64, 256] if c <= args.max_concurrency]
    if args.quick:
        n_trials_single = 3
        n_runs_concurrent = 1
        levels = [1, 4, 16, 64]
        ttft_levels = [1, 16, 64]
        rag_levels = [1, 4, 16]
        long_sizes = (1024, 4096)
    else:
        n_trials_single = 10
        n_runs_concurrent = 3
        rag_levels = [1, 4, 16, 64]
        long_sizes = (1024, 4096, 16384, 32768)

    print(f"# Qwen3.6-35B-A3B-heretic NVFP4 + DFlash — Full benchmark")
    print(f"# base: {args.base_url}  model: {args.model}")
    print(f"# sections: {sections}")
    print(f"# concurrency levels: {levels}")
    print(f"# started: {datetime.now(timezone.utc).isoformat()}")

    out = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": args.model,
        "config": {
            "n_trials_single": n_trials_single,
            "n_runs_concurrent": n_runs_concurrent,
            "concurrency_levels": levels,
            "ttft_levels": ttft_levels,
            "rag_levels": rag_levels,
            "long_prefill_sizes": list(long_sizes),
        },
        "results": {},
    }

    if "single" in sections:
        out["results"]["single"] = await bench_single_stream(
            args.base_url, args.model, n_trials=n_trials_single)
    if "ttft_by_prompt" in sections:
        out["results"]["ttft_by_prompt"] = await bench_ttft_by_prompt(
            args.base_url, args.model)
    if "decode_by_output" in sections:
        out["results"]["decode_by_output"] = await bench_decode_by_output(
            args.base_url, args.model)
    if "sampling" in sections:
        out["results"]["sampling"] = await bench_sampling(
            args.base_url, args.model)
    if "long_prefill" in sections:
        out["results"]["long_prefill"] = await bench_long_prefill(
            args.base_url, args.model, sizes=long_sizes)
    if "concurrent" in sections:
        out["results"]["concurrent"] = await bench_concurrent(
            args.base_url, args.model, levels=levels,
            n_runs=n_runs_concurrent)
    if "ttft_scaling" in sections:
        out["results"]["ttft_scaling"] = await bench_ttft_scaling(
            args.base_url, args.model, levels=ttft_levels)
    if "rag" in sections:
        out["results"]["rag"] = await bench_rag_concurrent(
            args.base_url, args.model, levels=rag_levels)

    out["finished_at"] = datetime.now(timezone.utc).isoformat()
    if args.output:
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[saved] JSON results -> {args.output}")
    else:
        print("\n[results JSON below]")
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
