"""Deployment command builders for production speculative decoding servers."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from typing import Literal


Engine = Literal["vllm", "sglang"]
Mode = Literal["baseline", "peagle", "eagle3", "standalone", "ngram", "token_itl"]


@dataclass(frozen=True)
class ServingProfile:
    """A reproducible serving command profile."""

    engine: Engine
    mode: Mode
    target_model: str
    draft_model: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int | None = None
    max_model_len: int | None = None
    dtype: str | None = None
    quantization: str | None = None
    num_speculative_tokens: int = 5
    speculative_num_steps: int = 4
    speculative_eagle_topk: int = 1
    speculative_num_draft_tokens: int = 6
    parallel_drafting: bool = True
    trust_remote_code: bool = True
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if self.mode in {"peagle", "eagle3", "standalone", "token_itl"} and not self.draft_model:
            raise ValueError(f"{self.mode} mode requires draft_model.")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("port must be in [1, 65535].")
        if self.num_speculative_tokens <= 0:
            raise ValueError("num_speculative_tokens must be positive.")
        if self.speculative_num_steps <= 0:
            raise ValueError("speculative_num_steps must be positive.")
        if self.speculative_eagle_topk <= 0:
            raise ValueError("speculative_eagle_topk must be positive.")
        if self.speculative_num_draft_tokens <= 0:
            raise ValueError("speculative_num_draft_tokens must be positive.")


def build_vllm_command(profile: ServingProfile) -> list[str]:
    """Build a ``vllm serve`` command."""

    profile.validate()
    if profile.engine != "vllm":
        raise ValueError("profile.engine must be 'vllm'.")

    command = [
        "vllm",
        "serve",
        profile.target_model,
        "--host",
        profile.host,
        "--port",
        str(profile.port),
    ]
    if profile.trust_remote_code:
        command.append("--trust-remote-code")
    if profile.tensor_parallel_size is not None:
        command += ["--tensor-parallel-size", str(profile.tensor_parallel_size)]
    if profile.max_model_len is not None:
        command += ["--max-model-len", str(profile.max_model_len)]
    if profile.dtype is not None:
        command += ["--dtype", profile.dtype]

    speculative_config = _vllm_speculative_config(profile)
    if speculative_config is not None:
        command += ["--speculative-config", json.dumps(speculative_config, separators=(",", ":"))]

    command += list(profile.extra_args)
    return command


def build_sglang_command(profile: ServingProfile) -> list[str]:
    """Build an SGLang serving command."""

    profile.validate()
    if profile.engine != "sglang":
        raise ValueError("profile.engine must be 'sglang'.")

    launcher = (
        ["sglang-itl-launch"]
        if profile.mode == "token_itl"
        else ["python3", "-m", "sglang.launch_server"]
    )
    command = [
        *launcher,
        "--model",
        profile.target_model,
        "--host",
        profile.host,
        "--port",
        str(profile.port),
    ]
    if profile.trust_remote_code:
        command.append("--trust-remote-code")
    if profile.tensor_parallel_size is not None:
        command += ["--tp", str(profile.tensor_parallel_size)]
    if profile.max_model_len is not None:
        command += ["--context-length", str(profile.max_model_len)]
    if profile.dtype is not None:
        command += ["--dtype", profile.dtype]
    if profile.quantization is not None:
        command += ["--quantization", profile.quantization]

    if profile.mode == "eagle3" or profile.mode == "peagle":
        command += [
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            str(profile.draft_model),
            "--speculative-num-steps",
            str(profile.speculative_num_steps),
            "--speculative-eagle-topk",
            str(profile.speculative_eagle_topk),
            "--speculative-num-draft-tokens",
            str(profile.speculative_num_draft_tokens),
        ]
    elif profile.mode == "standalone":
        command += [
            "--speculative-algorithm",
            "STANDALONE",
            "--speculative-draft-model-path",
            str(profile.draft_model),
            "--speculative-num-steps",
            str(profile.speculative_num_steps),
            "--speculative-eagle-topk",
            str(profile.speculative_eagle_topk),
            "--speculative-num-draft-tokens",
            str(profile.speculative_num_draft_tokens),
        ]
    elif profile.mode == "ngram":
        command += ["--speculative-algorithm", "NGRAM"]
    elif profile.mode == "token_itl":
        command += [
            "--speculative-algorithm",
            "TOKEN_ITL",
            "--speculative-draft-model-path",
            str(profile.draft_model),
            "--speculative-num-steps",
            str(profile.speculative_num_steps),
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            str(profile.speculative_num_draft_tokens),
            "--speculative-ngram-max-bfs-breadth",
            "1",
            "--disable-overlap-schedule",
            "--disable-cuda-graph",
        ]

    command += list(profile.extra_args)
    return command


def build_command(profile: ServingProfile) -> list[str]:
    if profile.engine == "vllm":
        return build_vllm_command(profile)
    if profile.engine == "sglang":
        return build_sglang_command(profile)
    raise ValueError(f"Unsupported engine: {profile.engine}")


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def minimax_m27_nvfp4_profile(
    *,
    engine: Engine = "vllm",
    mode: Mode = "peagle",
    target_model: str = "nvidia/MiniMax-M2.7-NVFP4",
    draft_model: str | None = "phatv9/p-eagle-minimax-m2.7",
    port: int = 8000,
    tensor_parallel_size: int | None = None,
    max_model_len: int | None = 32768,
) -> ServingProfile:
    """Return a production-oriented MiniMax-M2.7-NVFP4 serving profile."""

    if mode in {"baseline", "ngram"}:
        draft_model = None
    if engine == "sglang" and mode == "peagle":
        mode = "eagle3"
    speculative_num_steps = 4
    speculative_num_draft_tokens = 5 if mode == "token_itl" else 6

    return ServingProfile(
        engine=engine,
        mode=mode,
        target_model=target_model,
        draft_model=draft_model,
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        quantization="modelopt_fp4" if engine == "sglang" else None,
        num_speculative_tokens=5,
        speculative_num_steps=speculative_num_steps,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        parallel_drafting=True,
        trust_remote_code=True,
    )


def _vllm_speculative_config(profile: ServingProfile) -> dict[str, object] | None:
    if profile.mode == "baseline":
        return None
    if profile.mode == "ngram":
        return {
            "method": "ngram",
            "num_speculative_tokens": profile.num_speculative_tokens,
        }
    if profile.mode == "peagle":
        return {
            "method": "eagle3",
            "model": profile.draft_model,
            "num_speculative_tokens": profile.num_speculative_tokens,
            "parallel_drafting": profile.parallel_drafting,
        }
    if profile.mode == "eagle3":
        return {
            "method": "eagle3",
            "model": profile.draft_model,
            "num_speculative_tokens": profile.num_speculative_tokens,
        }
    if profile.mode == "standalone":
        return {
            "method": "draft_model",
            "model": profile.draft_model,
            "num_speculative_tokens": profile.num_speculative_tokens,
        }
    if profile.mode == "token_itl":
        raise ValueError("token_itl mode is currently implemented for SGLang only.")
    raise ValueError(f"Unsupported vLLM speculative mode: {profile.mode}")
