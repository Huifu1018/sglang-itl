"""SGLang plugin entry point registering the TOKEN_ITL algorithm."""

from __future__ import annotations

from . import TOKEN_ITL_ALGORITHM
from .validation import validate_server_args


def activate() -> None:
    """Register TOKEN_ITL with SGLang's speculative algorithm registry."""

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    from sglang.srt.speculative.spec_registry import CustomSpecAlgo, get_spec

    if get_spec(TOKEN_ITL_ALGORITHM) is not None:
        return

    class TokenITLSpecAlgo(CustomSpecAlgo):
        def is_ngram(self) -> bool:
            # The worker reuses SGLang's NGRAM verify input and target-only
            # verification kernels, but supplies TokenTiming draft candidates.
            # Returning True also prevents scheduler paths from assuming this
            # custom worker owns a native SGLang draft KV pool.
            return True

        def supports_spec_v2(self) -> bool:
            return False

    @SpeculativeAlgorithm.register(
        TOKEN_ITL_ALGORITHM,
        supports_overlap=False,
        validate_server_args=validate_server_args,
        spec_class=TokenITLSpecAlgo,
    )
    def _factory(server_args: object) -> type:
        from .worker import TokenITLWorker

        return TokenITLWorker
