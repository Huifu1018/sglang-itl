"""SGLang integration for Token-ITL.

The package is intentionally import-light. SGLang, torch, and transformers are
only imported by the plugin/worker modules inside an SGLang runtime.
"""

TOKEN_ITL_ALGORITHM = "TOKEN_ITL"

__all__ = ["TOKEN_ITL_ALGORITHM"]
