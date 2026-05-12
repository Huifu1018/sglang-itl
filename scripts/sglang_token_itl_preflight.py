"""Preflight checks for running TOKEN_ITL inside SGLang."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, entry_points, version
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=None, help="Optional target model id/path.")
    parser.add_argument("--draft", default=None, help="Optional draft model id/path.")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--allow-no-cuda", action="store_true")
    parser.add_argument("--skip-model-config", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks: list[dict[str, Any]] = []

    _check_package(checks, "sglang-itl")
    _check_entrypoint(checks)
    _check_sglang_registration(checks)
    _check_cuda(checks, allow_no_cuda=args.allow_no_cuda)

    if not args.skip_model_config:
        if args.target:
            _check_hf_config(checks, "target_config", args.target, args.trust_remote_code)
        if args.draft:
            _check_hf_config(checks, "draft_config", args.draft, args.trust_remote_code)

    ok = all(item["ok"] for item in checks)
    print(json.dumps({"ok": ok, "checks": checks}, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(1)


def _check_package(checks: list[dict[str, Any]], package_name: str) -> None:
    try:
        checks.append(
            {
                "name": f"package:{package_name}",
                "ok": True,
                "detail": version(package_name),
            }
        )
    except PackageNotFoundError as exc:
        checks.append({"name": f"package:{package_name}", "ok": False, "detail": str(exc)})


def _check_entrypoint(checks: list[dict[str, Any]]) -> None:
    matches = [
        ep
        for ep in entry_points(group="sglang.srt.plugins")
        if ep.name == "token_itl" and ep.value == "tokentiming.sglang.plugin:activate"
    ]
    checks.append(
        {
            "name": "sglang_entrypoint",
            "ok": len(matches) == 1,
            "detail": [f"{ep.name}={ep.value}" for ep in matches],
        }
    )


def _check_sglang_registration(checks: list[dict[str, Any]]) -> None:
    try:
        import sglang  # noqa: F401

        sglang_version = _safe_version("sglang")
        from tokentiming.sglang.plugin import activate

        activate()
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        algo = SpeculativeAlgorithm.from_string("TOKEN_ITL")
        checks.append(
            {
                "name": "sglang_registration",
                "ok": True,
                "detail": {
                    "sglang_version": sglang_version,
                    "algorithm": repr(algo),
                    "is_ngram_compatible": bool(algo.is_ngram()),
                    "supports_spec_v2": bool(algo.supports_spec_v2()),
                },
            }
        )
    except Exception as exc:
        checks.append({"name": "sglang_registration", "ok": False, "detail": repr(exc)})


def _check_cuda(checks: list[dict[str, Any]], *, allow_no_cuda: bool) -> None:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        checks.append(
            {
                "name": "cuda",
                "ok": available or allow_no_cuda,
                "detail": {
                    "available": available,
                    "device_count": torch.cuda.device_count() if available else 0,
                },
            }
        )
    except Exception as exc:
        checks.append({"name": "cuda", "ok": allow_no_cuda, "detail": repr(exc)})


def _check_hf_config(
    checks: list[dict[str, Any]],
    name: str,
    model_id: str,
    trust_remote_code: bool,
) -> None:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        checks.append(
            {
                "name": name,
                "ok": True,
                "detail": {
                    "model": model_id,
                    "model_type": getattr(cfg, "model_type", None),
                    "architectures": getattr(cfg, "architectures", None),
                },
            }
        )
    except Exception as exc:
        checks.append({"name": name, "ok": False, "detail": repr(exc)})


def _safe_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
