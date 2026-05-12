"""Small OpenAI-compatible benchmark for deployed LLM servers.

The tool intentionally uses the Python standard library so it can run on
serving nodes without extra dependencies.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROMPTS = [
    "用中文解释 speculative decoding 的核心思想，限制在 200 字以内。",
    "Write a Python function that merges overlapping intervals.",
    "请比较 Kimi、GLM、MiniMax 这类 MoE 模型部署时的显存瓶颈。",
    "Summarize the tradeoffs between vLLM and SGLang for serving large MoE models.",
]


@dataclass(frozen=True)
class RequestResult:
    ok: bool
    latency_s: float
    output_tokens: int
    input_tokens: int
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", choices=["chat", "completion"], default="chat")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--prompts-file", default=None)
    return parser.parse_args()


def load_prompts(path: str | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    text = Path(path).read_text(encoding="utf-8")
    prompts = [line.strip() for line in text.splitlines() if line.strip()]
    if not prompts:
        raise ValueError("prompts-file contains no non-empty prompts.")
    return prompts


def make_payload(args: argparse.Namespace, prompt: str) -> dict[str, object]:
    if args.endpoint == "chat":
        return {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
    return {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }


def post_json(url: str, payload: dict[str, object], api_key: str, timeout: float) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run_one(args: argparse.Namespace, prompt: str) -> RequestResult:
    path = "/v1/chat/completions" if args.endpoint == "chat" else "/v1/completions"
    url = args.base_url.rstrip("/") + path
    started_at = time.perf_counter()
    try:
        response = post_json(url, make_payload(args, prompt), args.api_key, args.timeout)
        latency = time.perf_counter() - started_at
        usage = response.get("usage") or {}
        output_tokens = int(usage.get("completion_tokens") or 0)
        input_tokens = int(usage.get("prompt_tokens") or 0)
        return RequestResult(
            ok=True,
            latency_s=latency,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        latency = time.perf_counter() - started_at
        return RequestResult(ok=False, latency_s=latency, output_tokens=0, input_tokens=0, error=str(exc))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts_file)
    scheduled = [prompts[index % len(prompts)] for index in range(args.requests)]

    started_at = time.perf_counter()
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(run_one, args, prompt) for prompt in scheduled]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = time.perf_counter() - started_at

    ok_results = [result for result in results if result.ok]
    failed_results = [result for result in results if not result.ok]
    latencies = [result.latency_s for result in ok_results]
    output_tokens = sum(result.output_tokens for result in ok_results)

    summary = {
        "requests": len(results),
        "succeeded": len(ok_results),
        "failed": len(failed_results),
        "elapsed_s": round(elapsed, 4),
        "request_qps": round(len(ok_results) / elapsed, 4) if elapsed else 0.0,
        "output_tokens": output_tokens,
        "output_tokens_per_s": round(output_tokens / elapsed, 4) if elapsed else 0.0,
        "latency_p50_s": round(percentile(latencies, 0.50), 4),
        "latency_p95_s": round(percentile(latencies, 0.95), 4),
        "latency_mean_s": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "first_error": failed_results[0].error if failed_results else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
