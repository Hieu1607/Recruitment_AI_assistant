class ScoringProviderLimitError(RuntimeError):
    """Raised when upstream LLM scoring is blocked by quota or rate limiting."""
