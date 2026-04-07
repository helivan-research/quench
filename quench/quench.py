"""
Quench - Efficient evaluation framework for generative models

This module provides the QuenchClient and Benchmark classes for
managing benchmarks and predicting model scores.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("quench")

DEFAULT_BASE_URL = "https://quench.helivan.io/api"
DEFAULT_TIMEOUT = 30
LONG_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class QuenchAPIError(Exception):
    """Base exception for Quench API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[requests.Response] = None):
        self.status_code = status_code
        self.response = response
        super().__init__(message)


class AuthenticationError(QuenchAPIError):
    """Raised when authentication fails or is missing."""
    pass


class NotFoundError(QuenchAPIError):
    """Raised when a resource is not found."""
    pass


class RateLimitError(QuenchAPIError):
    """Raised when the API rate limit is exceeded."""
    pass


class BenchmarkNotFoundError(NotFoundError):
    """Raised when a benchmark doesn't exist."""
    pass


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _build_session() -> requests.Session:
    """Build a requests session with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST", "PATCH", "DELETE"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _raise_for_status(response: requests.Response) -> None:
    """Raise an appropriate QuenchAPIError for non-2xx responses."""
    if response.ok:
        return

    # Try to extract an error message from the JSON body
    message = f"HTTP {response.status_code}"
    try:
        body = response.json()
        if isinstance(body, dict):
            message = body.get("error", body.get("message", message))
    except Exception:
        if response.text:
            message = response.text[:200]

    status = response.status_code
    if status == 401 or status == 403:
        raise AuthenticationError(message, status_code=status, response=response)
    if status == 404:
        raise NotFoundError(message, status_code=status, response=response)
    if status == 429:
        raise RateLimitError(message, status_code=status, response=response)
    raise QuenchAPIError(message, status_code=status, response=response)


# ---------------------------------------------------------------------------
# QuenchClient
# ---------------------------------------------------------------------------

class QuenchClient:
    """Low-level client for the Quench API.

    Handles authentication, retries, and HTTP communication.

    Args:
        api_key: Your Quench API key.  Falls back to the ``QUENCH_API_KEY``
            environment variable when *None*.
        base_url: API base URL.

    Example::

        from quench import QuenchClient

        client = QuenchClient(api_key="qk_...")
        benchmark = client.benchmarks.load("my-benchmark")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("QUENCH_API_KEY")
        self.base_url = (base_url or os.environ.get("QUENCH_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._session = _build_session()
        self._jwt_token: Optional[str] = None
        self._jwt_user_id: Optional[str] = None
        self.benchmarks = _BenchmarkNamespace(self)

    # -- internal helpers ---------------------------------------------------

    def _ensure_auth(self) -> None:
        """Authenticate if we have an API key but no JWT token yet."""
        if self._jwt_token is not None:
            return
        if self.api_key is None:
            raise AuthenticationError(
                "No API key provided. Set quench.api_key, pass api_key= to "
                "QuenchClient, or set the QUENCH_API_KEY environment variable."
            )
        resp = self._session.post(
            f"{self.base_url}/auth/login",
            json={"api_key": self.api_key},
            timeout=DEFAULT_TIMEOUT,
        )
        _raise_for_status(resp)
        data = resp.json()
        self._jwt_token = data.get("token")
        self._jwt_user_id = data.get("user_id")
        logger.debug("Authenticated as user %s", self._jwt_user_id)

    def _auth_headers(self) -> Dict[str, str]:
        """Return Authorization header dict (logs in if needed)."""
        self._ensure_auth()
        return {"Authorization": f"Bearer {self._jwt_token}"}

    def _optional_auth_headers(self) -> Dict[str, str]:
        """Return Authorization header if possible, empty dict otherwise."""
        if self._jwt_token is not None:
            return {"Authorization": f"Bearer {self._jwt_token}"}
        if self.api_key is not None:
            try:
                self._ensure_auth()
                return {"Authorization": f"Bearer {self._jwt_token}"}
            except QuenchAPIError:
                pass
        return {}

    def _get(self, path: str, *, params: Optional[Dict] = None, auth: bool = False, timeout: int = DEFAULT_TIMEOUT) -> Any:
        headers = self._auth_headers() if auth else self._optional_auth_headers()
        resp = self._session.get(f"{self.base_url}{path}", headers=headers, params=params, timeout=timeout)
        _raise_for_status(resp)
        return resp.json()

    def _post(self, path: str, *, json: Optional[Dict] = None, auth: bool = False, timeout: int = DEFAULT_TIMEOUT) -> Any:
        headers = self._auth_headers() if auth else self._optional_auth_headers()
        resp = self._session.post(f"{self.base_url}{path}", headers=headers, json=json, timeout=timeout)
        _raise_for_status(resp)
        return resp.json()

    def _patch(self, path: str, *, json: Optional[Dict] = None, auth: bool = True, timeout: int = DEFAULT_TIMEOUT) -> Any:
        headers = self._auth_headers() if auth else self._optional_auth_headers()
        resp = self._session.patch(f"{self.base_url}{path}", headers=headers, json=json, timeout=timeout)
        _raise_for_status(resp)
        return resp.json()

    def _delete(self, path: str, *, auth: bool = True, timeout: int = DEFAULT_TIMEOUT) -> Any:
        headers = self._auth_headers() if auth else self._optional_auth_headers()
        resp = self._session.delete(f"{self.base_url}{path}", headers=headers, timeout=timeout)
        _raise_for_status(resp)
        # DELETE may return empty body
        if resp.text:
            return resp.json()
        return None

    def _put_raw(self, url: str, data: bytes, content_type: str = "application/json", timeout: int = LONG_TIMEOUT) -> None:
        """PUT raw bytes to an arbitrary URL (e.g. presigned R2 URL)."""
        resp = self._session.put(url, data=data, headers={"Content-Type": content_type}, timeout=timeout)
        _raise_for_status(resp)


# ---------------------------------------------------------------------------
# Benchmark namespace (accessed via client.benchmarks)
# ---------------------------------------------------------------------------

class _BenchmarkNamespace:
    """Namespace for benchmark operations on a client."""

    def __init__(self, client: QuenchClient) -> None:
        self._client = client

    def load(self, benchmark_name: str) -> "Benchmark":
        """Load a benchmark by name.

        Public benchmarks can be loaded without authentication.

        Args:
            benchmark_name: Name of the benchmark.

        Returns:
            A :class:`Benchmark` instance.

        Raises:
            BenchmarkNotFoundError: If the benchmark doesn't exist.
            AuthenticationError: If the benchmark is private and no API key is set.
        """
        try:
            data = self._client._get(f"/benchmarks/{benchmark_name}")
        except NotFoundError:
            raise BenchmarkNotFoundError(f"Benchmark '{benchmark_name}' not found")

        return Benchmark._from_api(self._client, benchmark_name, data)

    # 10 MB threshold for switching to presigned R2 upload
    # (Railway's proxy times out on larger inline payloads)
    _LARGE_FILE_THRESHOLD = 10 * 1024 * 1024

    def create(
        self,
        benchmark_name: str,
        data: Dict,
        *,
        embedding_model: str = "google/gemini-embedding-001",
        **kwargs: Any,
    ) -> "Benchmark":
        """Create a new benchmark.

        For large datasets (>50 MB), the data is automatically uploaded to
        cloud storage via a presigned URL before creating the benchmark.

        Args:
            benchmark_name: Name for the new benchmark.
            data: Dict following the Quench data schema with model responses.
            embedding_model: Embedding provider/model string.
            **kwargs: Extra metadata (description, category, etc.).

        Returns:
            A :class:`Benchmark` instance. Check :attr:`Benchmark.status` —
            it will be ``"processing"`` until embeddings are computed.

        Raises:
            AuthenticationError: If not authenticated.
        """
        _validate_response_data(data)

        # Check if data is too large for inline JSON
        serialized = json.dumps(data)
        use_r2 = len(serialized.encode()) > self._LARGE_FILE_THRESHOLD

        # Extract top-level fields the backend expects outside of metadata
        metadata = dict(kwargs)
        payload: Dict[str, Any] = {
            "benchmark_name": benchmark_name,
            "embedding_model": embedding_model,
        }
        for key in ("organization_id", "category"):
            if key in metadata:
                payload[key] = metadata.pop(key)
        payload["metadata"] = metadata

        if use_r2:
            data_mb = len(serialized) / 1024 / 1024
            logger.info("Benchmark data is large (%.0f MB), uploading per-model to R2", data_mb)

            models = [k for k in data if k != "metadata"]

            # Upload each model's data as a separate R2 object
            model_keys: Dict[str, str] = {}
            for i, model_name in enumerate(models):
                model_json = json.dumps(data[model_name])
                presign_resp = self._client._post(
                    "/upload/presign",
                    json={"filename": f"{benchmark_name}/{model_name}.json", "content_type": "application/json"},
                    auth=True,
                )
                upload_url = presign_resp["upload_url"]
                model_key = presign_resp["storage_key"]
                upload_timeout = max(LONG_TIMEOUT, len(model_json) // (1024 * 1024) * 2)
                self._client._put_raw(upload_url, model_json.encode(), content_type="application/json", timeout=upload_timeout)
                model_keys[model_name] = model_key
                if (i + 1) % 10 == 0 or i == len(models) - 1:
                    logger.info("Uploaded %d/%d models to R2", i + 1, len(models))

            payload["model_storage_keys"] = model_keys

            # Extract lightweight data client-side
            queries: Dict[str, str] = {}
            model_scores: Dict[str, float] = {}
            if models:
                ref = data[models[0]]
                for st_name, st_data in ref.items():
                    if st_name in ("metadata", "score") or not isinstance(st_data, dict):
                        continue
                    for qid, qdata in st_data.items():
                        if qid in ("metadata", "score", "query_score") or not isinstance(qdata, dict):
                            continue
                        q = qdata.get("question", "")
                        if q:
                            queries[f"{st_name}/{qid}"] = q
                for m in models:
                    md = data[m]
                    if isinstance(md, dict) and "score" in md and not isinstance(md["score"], dict):
                        model_scores[m] = float(md["score"])
            payload["queries"] = queries
            payload["model_scores"] = model_scores
        else:
            payload["benchmark_data"] = data

        resp = self._client._post("/benchmarks", json=payload, auth=True, timeout=LONG_TIMEOUT)

        bm = Benchmark(self._client, benchmark_name)
        bm._data = data
        bm._metadata = resp.get("metadata", kwargs)
        bm._benchmark_id = resp.get("benchmark_id")
        bm._status = resp.get("status", "processing")

        if bm._status == "processing":
            logger.info("Created benchmark '%s' (processing — call wait_until_ready() to block)", benchmark_name)
        else:
            logger.info("Created benchmark '%s'", benchmark_name)

        return bm

    def list(self, *, category: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Dict]:
        """List public benchmarks.

        Args:
            category: Filter by category (e.g. ``"math"``, ``"coding"``).
            limit: Max results to return.
            offset: Pagination offset.

        Returns:
            List of benchmark summary dicts.
        """
        params: Dict[str, Any] = {"public_only": "true", "limit": limit, "offset": offset}
        if category:
            params["category"] = category
        resp = self._client._get("/benchmarks", params=params)
        if isinstance(resp, dict) and "benchmarks" in resp:
            return resp["benchmarks"]
        return resp

    def list_mine(self) -> List[Dict]:
        """List benchmarks owned by the authenticated user.

        Returns:
            List of benchmark dicts grouped by organization.

        Raises:
            AuthenticationError: If not authenticated.
        """
        return self._client._get("/my/benchmarks", auth=True)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class Benchmark:
    """A loaded benchmark with model data and prediction capabilities.

    Do not instantiate directly — use :meth:`quench.Benchmark.load` or
    :meth:`quench.Benchmark.create`.
    """

    def __init__(self, client: QuenchClient, name: str) -> None:
        self._client = client
        self.name = name
        self._data: Optional[Dict] = None
        self._metadata: Optional[Dict] = None
        self._embeddings: Optional[Dict] = None
        self._status: Optional[str] = None
        self._benchmark_id: Optional[str] = None
        self._queries: Optional[Dict] = None
        self._model_scores: Optional[Dict] = None

    @classmethod
    def _from_api(cls, client: QuenchClient, name: str, api_data: Dict) -> "Benchmark":
        bm = cls(client, name)
        bm._data = api_data.get("benchmark_data", {})
        bm._metadata = api_data.get("metadata", {})
        bm._embeddings = api_data.get("embeddings", {})
        bm._status = api_data.get("status", "ready")
        bm._queries = api_data.get("queries")
        bm._model_scores = api_data.get("model_scores")
        num_models = len(bm.models)
        logger.info("Loaded benchmark '%s' with %d model(s)", name, num_models)
        return bm

    # -- properties ---------------------------------------------------------

    @property
    def data(self) -> Dict:
        """The full benchmark data dict."""
        if self._data is None:
            raise ValueError("Benchmark data not loaded.")
        return self._data

    @property
    def models(self) -> List[str]:
        """List of model names in this benchmark."""
        # New storage: stub has _models list
        if self._data and "_models" in self._data:
            return self._data["_models"]
        # New storage: model_scores has model names
        if self._model_scores:
            return list(self._model_scores.keys())
        # Legacy: model names are top-level keys
        if self._data is None:
            return []
        return [k for k in self._data if k != "metadata"]

    @property
    def scores(self) -> Dict[str, float]:
        """Model scores dict. Available without loading full data."""
        if self._model_scores:
            return self._model_scores
        # Fallback: extract from legacy data
        if self._data:
            return {
                m: self._data[m].get("score", 0)
                for m in self.models
                if isinstance(self._data.get(m), dict) and "score" in self._data[m]
            }
        return {}

    @property
    def status(self) -> Optional[str]:
        """Benchmark processing status (``processing``, ``ready``, ``failed``)."""
        return self._status

    # -- lifecycle ----------------------------------------------------------

    def wait_until_ready(self, timeout: int = 300, poll_interval: int = 5, on_progress: Any = None) -> "Benchmark":
        """Block until the benchmark is done processing.

        After :meth:`create`, the backend computes embeddings asynchronously.
        Call this method to wait for completion.

        Args:
            timeout: Max seconds to wait (default 300).
            poll_interval: Seconds between status checks (default 5).
            on_progress: Optional callback ``fn(progress_str, pct)`` called on each poll.

        Returns:
            self

        Raises:
            QuenchAPIError: If processing fails.
            TimeoutError: If *timeout* is exceeded.
        """
        if self._status not in (None, "processing"):
            return self

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            resp = self._client._get(f"/benchmarks/{self.name}/summary")
            status = resp.get("status", "ready")
            metadata = resp.get("metadata", {})

            if on_progress:
                progress = metadata.get("embedding_progress", "")
                pct = metadata.get("embedding_pct", 0)
                if progress:
                    on_progress(progress, pct)

            if status == "ready":
                self._status = "ready"
                logger.info("Benchmark '%s' is ready", self.name)
                return self
            if status == "failed":
                error = metadata.get("processing_error", "unknown error")
                raise QuenchAPIError(f"Benchmark '{self.name}' processing failed: {error}")
            logger.debug("Benchmark '%s' still processing...", self.name)
            time.sleep(poll_interval)

        raise TimeoutError(f"Benchmark '{self.name}' not ready after {timeout}s")

    @property
    def progress(self) -> Optional[Dict]:
        """Get current processing progress. Returns None if not processing."""
        if self._status != "processing":
            return None
        try:
            resp = self._client._get(f"/benchmarks/{self.name}/summary")
            metadata = resp.get("metadata", {})
            return {
                "status": resp.get("status"),
                "progress": metadata.get("embedding_progress"),
                "pct": metadata.get("embedding_pct"),
            }
        except Exception:
            return None

    # -- read operations ----------------------------------------------------

    def summary(self) -> Dict:
        """Get a lightweight summary with model scores.

        Returns:
            Dict with model names and their scores.
        """
        return self._client._get(f"/benchmarks/{self.name}/summary")

    def visualize(self) -> str:
        """Get an interactive MDS visualization as HTML.

        Returns:
            HTML string with the Plotly visualization.
        """
        return self._client._get(f"/benchmarks/{self.name}/visualize")

    def get_query_dictionary(self, subtask: Optional[str] = None) -> Dict[str, str]:
        """Get mapping of query IDs to question text.

        Args:
            subtask: Filter to a specific subtask. If *None*, returns all
                queries with ``subtask/query_id`` keys.

        Returns:
            Dict mapping query IDs to question strings.
        """
        # New storage: queries stored separately as {subtask/qid: question}
        if self._queries:
            if subtask:
                prefix = f"{subtask}/"
                return {
                    k[len(prefix):]: v for k, v in self._queries.items()
                    if k.startswith(prefix)
                }
            return dict(self._queries)

        # Legacy: extract from benchmark_data
        if self._data is None:
            raise ValueError("Benchmark data not loaded.")

        model_names = self.models
        if not model_names:
            return {}

        reference = self._data[model_names[0]]
        if not isinstance(reference, dict):
            return {}

        query_dict: Dict[str, str] = {}

        if subtask:
            if subtask not in reference or subtask == "metadata":
                subtasks = [k for k in reference if k != "metadata"]
                raise ValueError(f"Subtask '{subtask}' not found. Available: {subtasks}")
            for qid, qdata in reference[subtask].items():
                if qid in ("query_score", "score", "metadata") or not isinstance(qdata, dict):
                    continue
                query_dict[qid] = qdata.get("question", "")
        else:
            for st_name, st_data in reference.items():
                if st_name == "metadata" or not isinstance(st_data, dict):
                    continue
                for qid, qdata in st_data.items():
                    if qid in ("query_score", "score", "metadata") or not isinstance(qdata, dict):
                        continue
                    query_dict[f"{st_name}/{qid}"] = qdata.get("question", "")

        return query_dict

    def get_model_metadata(self, model_name: str) -> Dict:
        """Get metadata for a specific model.

        Args:
            model_name: Name of the model.

        Returns:
            Model metadata dict.

        Raises:
            ValueError: If the model is not in this benchmark.
        """
        if self._data is None or model_name not in self._data:
            raise ValueError(f"Model '{model_name}' not found in benchmark '{self.name}'")
        return self._data[model_name].get("metadata", {})

    # -- mutating operations ------------------------------------------------

    def add_model(self, response_json: Dict, model_name: Optional[str] = None) -> "Benchmark":
        """Add a model's responses to this benchmark.

        Args:
            response_json: Either a full schema dict with one model key, or
                subtask-level data (requires *model_name*).
            model_name: Required when *response_json* is subtask-level data.

        Returns:
            self (for method chaining).
        """
        if self._data is None:
            raise ValueError("No benchmark data loaded.")

        if model_name and model_name not in response_json:
            final_name = model_name
            model_data = response_json
        else:
            model_keys = [k for k in response_json if k != "metadata"]
            if len(model_keys) != 1:
                raise ValueError(
                    "response_json must contain exactly one model key, "
                    "or pass model_name= with subtask-level data."
                )
            final_name = model_keys[0]
            model_data = response_json[final_name]

        _validate_model_data(model_data, final_name)

        self._client._patch(
            f"/benchmarks/{self.name}/models/{final_name}",
            json={"model_data": model_data},
            auth=True,
            timeout=LONG_TIMEOUT,
        )
        self._data[final_name] = model_data
        logger.info("Added model '%s' to benchmark '%s'", final_name, self.name)
        return self

    def remove_model(self, model_name: str) -> "Benchmark":
        """Remove a model from this benchmark.

        Args:
            model_name: Name of the model to remove.

        Returns:
            self (for method chaining).
        """
        if self._data is None:
            raise ValueError("No benchmark data loaded.")
        if model_name not in self._data:
            raise ValueError(f"Model '{model_name}' not found in benchmark '{self.name}'")

        self._client._delete(f"/benchmarks/{self.name}/models/{model_name}", auth=True)
        del self._data[model_name]
        logger.info("Removed model '%s' from benchmark '%s'", model_name, self.name)
        return self

    def delete(self) -> None:
        """Delete this benchmark permanently.

        Raises:
            AuthenticationError: If not authenticated or not the owner.
        """
        self._client._delete(f"/benchmarks/{self.name}", auth=True)
        logger.info("Deleted benchmark '%s'", self.name)
        self._data = None
        self._metadata = None

    # -- prediction & query selection ---------------------------------------

    def stage(self, query_ids: List) -> Dict:
        """Stage a prediction by preloading cached model embeddings.

        Call this before :meth:`predict` to make prediction instant (~2s
        instead of ~90s). The returned dict contains a ``session_id``
        that must be passed to :meth:`predict`.

        Args:
            query_ids: List of ``(subtask, query_id)`` tuples or
                ``[subtask, query_id]`` lists specifying which queries
                to stage.

        Returns:
            Dict with ``session_id``, ``query_count``, ``model_count``,
            ``expires_in``.

        Example::

            optimal = benchmark.get_optimal_queries(budget=20)
            query_ids = [(q["subtask"], q["query_id"]) for q in optimal["queries"]]
            session = benchmark.stage(query_ids)
            # ... run model on those queries ...
            results = benchmark.predict(responses, session=session)
        """
        result = self._client._post(
            f"/benchmarks/{self.name}/stage",
            json={"query_ids": [list(q) for q in query_ids]},
            timeout=LONG_TIMEOUT,
        )
        logger.info("Staged %d queries for benchmark '%s' (session=%s)",
                     result.get("query_count", 0), self.name, result.get("session_id"))
        return result

    def predict(self, response_json: Dict, session: Optional[Dict] = None) -> Dict:
        """Predict benchmark scores for a partially-evaluated model.

        Args:
            response_json: Model responses (can be partial). Format::

                {
                    "model_name": {
                        "subtask": {
                            "query_id": {
                                "question": "...",
                                "response": ["..."]
                            }
                        }
                    }
                }

            session: Optional staged session dict (from :meth:`stage`).
                When provided, prediction uses preloaded embeddings
                and completes in ~2s instead of ~90s.

        Returns:
            Dict with ``predicted_scores``, ``confidence_interval``,
            ``similar_models``, ``cached_model_scores``, ``overlap_info``,
            and ``mds_coordinates``.
        """
        if self._data is None:
            raise ValueError("No benchmark data loaded. Load a benchmark first.")

        # Use the stored-benchmark endpoint
        if self.name and self._status == "ready":
            payload = {"new_model_data": response_json}
            if session and "session_id" in session:
                payload["session_id"] = session["session_id"]
            result = self._client._post(
                f"/benchmarks/{self.name}/predict",
                json=payload,
                timeout=LONG_TIMEOUT,
            )
        else:
            payload = {
                "benchmark_data": self._data,
                "new_model_data": response_json,
                "benchmark_embeddings": self._embeddings or {},
            }
            result = self._client._post("/evaluate/predict", json=payload, timeout=LONG_TIMEOUT)

        # Normalize predicted_score → predicted_scores
        if "predicted_score" in result and "predicted_scores" not in result:
            result["predicted_scores"] = result.pop("predicted_score")

        logger.info("Prediction complete for benchmark '%s'", self.name)
        return result

    def get_optimal_queries(self, budget: int = 10) -> Dict:
        """Get the most informative queries for a given budget.

        Args:
            budget: Number of queries to select (default 10, max 100).

        Returns:
            Dict with ``queries``, ``total_queries``, ``estimated_error``,
            ``model_count``.
        """
        if self.name is None:
            raise ValueError("No benchmark loaded.")

        result = self._client._get(
            f"/benchmarks/{self.name}/optimal-queries",
            params={"budget": budget},
            timeout=LONG_TIMEOUT,
        )
        logger.debug("Selected %d optimal queries", len(result.get("queries", [])))
        return result

    def estimate_query_budget(self, target_error: float = 0.05) -> Dict:
        """Estimate queries needed for a target prediction error.

        Args:
            target_error: Target mean absolute error (default 0.05).

        Returns:
            Dict with ``estimated_queries``, ``confidence_interval``,
            ``error_curve``, ``total_queries``.
        """
        if self.name is None:
            raise ValueError("No benchmark loaded.")

        result = self._client._get(
            f"/benchmarks/{self.name}/query-budget",
            params={"target_error": target_error},
            timeout=LONG_TIMEOUT,
        )
        logger.debug("Estimated %s queries for %.1f%% error",
                      result.get("estimated_queries", "N/A"), target_error * 100)
        return result


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_response_data(data: Dict) -> None:
    """Validate that a full response dict follows the expected schema."""
    for model_name, model_data in data.items():
        if model_name == "metadata":
            continue
        _validate_model_data(model_data, model_name)


def _validate_model_data(model_data: Dict, model_name: str) -> None:
    """Validate a single model's data structure."""
    if not isinstance(model_data, dict):
        raise ValueError(f"Model '{model_name}' data must be a dict")

    for subtask_name, subtask_data in model_data.items():
        if subtask_name in ("metadata", "score"):
            continue
        if not isinstance(subtask_data, dict):
            raise ValueError(f"Subtask '{subtask_name}' in model '{model_name}' must be a dict")

        for query_id, query_data in subtask_data.items():
            if query_id in ("query_score", "score", "metadata"):
                continue
            if not isinstance(query_data, dict):
                continue
            if "response" not in query_data:
                raise ValueError(f"Missing 'response' in {model_name}/{subtask_name}/{query_id}")


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def embedding_providers(client: Optional[QuenchClient] = None) -> Dict:
    """List available embedding providers and models.

    Returns:
        Dict with ``providers``, ``default_provider``, ``default_model``.

    Args:
        client: Optional client instance. Uses the default client if *None*.
    """
    c = client or _get_default_client()
    return c._get("/embedding-providers")


def benchmark_categories(client: Optional[QuenchClient] = None) -> List[str]:
    """List available benchmark categories.

    Args:
        client: Optional client instance. Uses the default client if *None*.
    """
    c = client or _get_default_client()
    resp = c._get("/benchmark-categories")
    if isinstance(resp, dict) and "categories" in resp:
        return resp["categories"]
    return resp


# ---------------------------------------------------------------------------
# Default client (module-level api_key / base_url)
# ---------------------------------------------------------------------------

_default_client: Optional[QuenchClient] = None


def _get_default_client() -> QuenchClient:
    """Return (and lazily create) the module-level default client."""
    global _default_client
    if _default_client is None:
        # Import here to read module-level api_key / base_url set by user
        import quench as _mod
        _default_client = QuenchClient(
            api_key=getattr(_mod, "api_key", None),
            base_url=getattr(_mod, "base_url", None),
        )
    return _default_client


def _reset_default_client() -> None:
    """Reset the cached default client (used after api_key changes)."""
    global _default_client
    _default_client = None


class _DefaultBenchmarkAccessor:
    """Proxy that forwards Benchmark class methods to the default client.

    This allows ``quench.Benchmark.load(...)`` to work without
    explicitly creating a :class:`QuenchClient`.
    """

    def load(self, benchmark_name: str) -> Benchmark:
        """Load a benchmark using the default client."""
        _reset_default_client()
        return _get_default_client().benchmarks.load(benchmark_name)

    def create(self, benchmark_name: str, data: Dict, **kwargs: Any) -> Benchmark:
        """Create a benchmark using the default client."""
        _reset_default_client()
        return _get_default_client().benchmarks.create(benchmark_name, data, **kwargs)

    def list(self, **kwargs: Any) -> List[Dict]:
        """List benchmarks using the default client."""
        _reset_default_client()
        return _get_default_client().benchmarks.list(**kwargs)

    def list_mine(self) -> List[Dict]:
        """List user's benchmarks using the default client."""
        _reset_default_client()
        return _get_default_client().benchmarks.list_mine()
