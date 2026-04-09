"""
Quench - Predict benchmark scores from a fraction of the queries

Simple usage::

    import quench

    quench.api_key = "qk_..."  # or set QUENCH_API_KEY env var

    benchmark = quench.Benchmark.load("my-benchmark")
    results = benchmark.predict({"new-model": {...}})

Advanced (thread-safe) usage::

    from quench import QuenchClient

    client = QuenchClient(api_key="qk_...")
    benchmark = client.benchmarks.load("my-benchmark")
"""

from .quench import (
    AuthenticationError,
    BenchmarkNotFoundError,
    NotFoundError,
    QuenchAPIError,
    QuenchClient,
    RateLimitError,
    _DefaultBenchmarkAccessor,
    benchmark_categories,
    embedding_providers,
)
from .quench import (
    Benchmark as _Benchmark,  # noqa: F401 — used via _DefaultBenchmarkAccessor
)

__version__ = "0.2.0"

# Module-level configuration — set these before calling Benchmark.load()
api_key = None  # type: Optional[str]
"""Your Quench API key. Falls back to the ``QUENCH_API_KEY`` environment variable."""

base_url = None  # type: Optional[str]
"""Override the default API base URL."""

# quench.Benchmark.load(...) / .create(...) / .list(...)
Benchmark = _DefaultBenchmarkAccessor()

__all__ = [
    "__version__",
    "api_key",
    "base_url",
    "Benchmark",
    "QuenchClient",
    "QuenchAPIError",
    "AuthenticationError",
    "NotFoundError",
    "RateLimitError",
    "BenchmarkNotFoundError",
    "embedding_providers",
    "benchmark_categories",
]
