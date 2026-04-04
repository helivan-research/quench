# Changelog

## 0.2.0

First public beta of the Quench Python SDK.

### New API

- **`quench.api_key`** / **`QUENCH_API_KEY` env var** — simple, stateless authentication
- **`quench.Benchmark.load(name)`** — load a benchmark
- **`quench.Benchmark.create(name, data)`** — create a benchmark
- **`quench.Benchmark.list()`** — list public benchmarks
- **`quench.Benchmark.list_mine()`** — list your benchmarks
- **`benchmark.predict(responses)`** — predict scores from partial evaluation
- **`benchmark.get_optimal_queries(budget)`** — select most informative queries
- **`benchmark.estimate_query_budget(target_error)`** — estimate queries needed
- **`benchmark.add_model(data)`** / **`benchmark.remove_model(name)`** — manage models
- **`benchmark.delete()`** — delete a benchmark
- **`benchmark.summary()`** — lightweight score summary
- **`benchmark.visualize()`** — interactive MDS visualization
- **`QuenchClient`** — thread-safe client for advanced usage

### Improvements over 0.x

- Replaced global mutable auth state with per-client authentication
- Replaced `print()` calls with `logging` module
- Automatic retry with exponential backoff for rate limits and server errors
- Proper exception hierarchy: `QuenchAPIError`, `AuthenticationError`, `NotFoundError`, `RateLimitError`
- Request timeouts on all HTTP calls
- `py.typed` marker for mypy support
- Modern `pyproject.toml` packaging (replaces `setup.py`)
