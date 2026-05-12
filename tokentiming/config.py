"""Configuration for TokenTiming decoders."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenTimingConfig:
    """Runtime knobs for TokenTiming-style greedy speculative decoding."""

    max_new_tokens: int = 128
    num_draft_tokens: int = 8
    dtw_window: int | None = 8
    temperature: float = 1.0
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    device: str | None = None
    target_device: str | None = None
    draft_device: str | None = None
    use_cache: bool = True
    add_special_tokens: bool = False
    max_proxy_tokens_per_step: int | None = None

    def validate(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if self.num_draft_tokens <= 0:
            raise ValueError("num_draft_tokens must be positive.")
        if self.dtw_window is not None and self.dtw_window < 0:
            raise ValueError("dtw_window must be non-negative or None.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if self.max_proxy_tokens_per_step is not None and self.max_proxy_tokens_per_step <= 0:
            raise ValueError("max_proxy_tokens_per_step must be positive or None.")

    @property
    def effective_target_device(self) -> str | None:
        return self.target_device or self.device

    @property
    def effective_draft_device(self) -> str | None:
        return self.draft_device or self.device
