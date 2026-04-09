"""
Tests for the Quench SDK.

Covers:
- QuenchClient authentication
- Benchmark load/create/delete
- Model add/remove
- Prediction and query selection
- Helper methods (models, query_dictionary, metadata)
- Validation
- Error handling
"""


import pytest

from quench import AuthenticationError, BenchmarkNotFoundError, QuenchClient
from quench.quench import (
    Benchmark,
    QuenchAPIError,
    RateLimitError,
    _validate_model_data,
    _validate_response_data,
)
from tests.conftest import _make_response

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

class TestAuthentication:

    def test_client_uses_explicit_api_key(self, mock_session):
        client = QuenchClient(api_key="qk_test123")
        assert client.api_key == "qk_test123"

    def test_client_falls_back_to_env_var(self, mock_session, monkeypatch):
        monkeypatch.setenv("QUENCH_API_KEY", "qk_from_env")
        client = QuenchClient()
        assert client.api_key == "qk_from_env"

    def test_ensure_auth_posts_login(self, mock_session):
        mock_session.post.return_value = _make_response(
            {"token": "jwt_abc", "user_id": "user_1"}
        )
        client = QuenchClient(api_key="qk_test")
        client._ensure_auth()

        mock_session.post.assert_called_once()
        call_url = mock_session.post.call_args[0][0]
        assert "/auth/login" in call_url
        assert client._jwt_token == "jwt_abc"
        assert client._jwt_user_id == "user_1"

    def test_ensure_auth_caches_token(self, mock_session):
        mock_session.post.return_value = _make_response(
            {"token": "jwt_abc", "user_id": "user_1"}
        )
        client = QuenchClient(api_key="qk_test")
        client._ensure_auth()
        client._ensure_auth()  # second call should not re-login
        assert mock_session.post.call_count == 1

    def test_ensure_auth_raises_without_key(self, mock_session):
        client = QuenchClient()
        with pytest.raises(AuthenticationError, match="No API key"):
            client._ensure_auth()

    def test_auth_failure_raises(self, mock_session):
        mock_session.post.return_value = _make_response(
            {"error": "Invalid API key"}, status_code=401, ok=False
        )
        client = QuenchClient(api_key="qk_bad")
        with pytest.raises(AuthenticationError):
            client._ensure_auth()


# ---------------------------------------------------------------------------
# Benchmark.load
# ---------------------------------------------------------------------------

class TestBenchmarkLoad:

    def test_load_public_benchmark(self, mock_session, sample_benchmark_data):
        mock_session.get.return_value = _make_response({
            "benchmark_data": sample_benchmark_data,
            "metadata": {"is_public": True},
        })

        client = QuenchClient()  # no API key needed for public
        bm = client.benchmarks.load("public_bench")

        assert bm.name == "public_bench"
        assert "gpt4" in bm.models
        assert "metadata" not in bm.models

    def test_load_not_found(self, mock_session):
        mock_session.get.return_value = _make_response(
            {"error": "Not found"}, status_code=404, ok=False
        )
        client = QuenchClient()
        with pytest.raises(BenchmarkNotFoundError):
            client.benchmarks.load("nonexistent")

    def test_load_private_unauthenticated(self, mock_session):
        mock_session.get.return_value = _make_response(
            {"error": "Forbidden"}, status_code=403, ok=False
        )
        client = QuenchClient()
        with pytest.raises(AuthenticationError):
            client.benchmarks.load("private_bench")


# ---------------------------------------------------------------------------
# Benchmark.create
# ---------------------------------------------------------------------------

class TestBenchmarkCreate:

    def _setup_auth(self, mock_session):
        mock_session.post.return_value = _make_response(
            {"token": "jwt_abc", "user_id": "u1"}
        )

    def test_create_success(self, mock_session, sample_benchmark_data):
        # With threshold=0, all creates go through R2 (1 presign per model + create)
        mock_session.post.side_effect = [
            _make_response({"token": "jwt_abc", "user_id": "u1"}),  # login
            _make_response({
                "upload_url": "https://r2/upload", "storage_key": "uploads/gpt4.json", "expires_in": 3600,
            }),  # presign for gpt4
            _make_response({
                "benchmark_id": "bench_123", "status": "processing", "metadata": {},
            }),  # create
        ]
        mock_session.put.return_value = _make_response(None, text="")
        client = QuenchClient(api_key="qk_test")
        bm = client.benchmarks.create("new_bench", sample_benchmark_data)

        assert bm.name == "new_bench"
        assert bm.status == "processing"
        assert bm._benchmark_id == "bench_123"

    def test_create_requires_auth(self, mock_session, sample_benchmark_data):
        client = QuenchClient()  # no api key
        with pytest.raises(AuthenticationError):
            client.benchmarks.create("new_bench", sample_benchmark_data)

    def test_create_validates_data(self, mock_session):
        self._setup_auth(mock_session)
        client = QuenchClient(api_key="qk_test")

        invalid = {"model1": {"sub1": {"q1": {"question": "x"}}}}  # missing response
        with pytest.raises(ValueError, match="Missing 'response'"):
            client.benchmarks.create("bad_bench", invalid)


# ---------------------------------------------------------------------------
# Benchmark list / list_mine
# ---------------------------------------------------------------------------

class TestBenchmarkList:

    def test_list_public(self, mock_session):
        mock_session.get.return_value = _make_response([
            {"name": "bench1"}, {"name": "bench2"}
        ])
        client = QuenchClient()
        result = client.benchmarks.list()
        assert len(result) == 2

    def test_list_mine(self, mock_session):
        mock_session.post.return_value = _make_response(
            {"token": "jwt", "user_id": "u1"}
        )
        mock_session.get.return_value = _make_response([{"name": "my_bench"}])

        client = QuenchClient(api_key="qk_test")
        result = client.benchmarks.list_mine()
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Benchmark model operations
# ---------------------------------------------------------------------------

class TestModelOperations:

    def _make_loaded_benchmark(self, mock_session, sample_benchmark_data):
        mock_session.get.return_value = _make_response({
            "benchmark_data": sample_benchmark_data,
            "metadata": {},
        })
        mock_session.post.return_value = _make_response(
            {"token": "jwt", "user_id": "u1"}
        )
        client = QuenchClient(api_key="qk_test")
        return client.benchmarks.load("test_bench")

    def test_add_model(self, mock_session, sample_benchmark_data, sample_model_response):
        bm = self._make_loaded_benchmark(mock_session, sample_benchmark_data)
        mock_session.patch.return_value = _make_response({})

        result = bm.add_model(sample_model_response)
        assert result is bm
        assert "claude" in bm.models

    def test_add_model_with_name(self, mock_session, sample_benchmark_data):
        bm = self._make_loaded_benchmark(mock_session, sample_benchmark_data)
        mock_session.patch.return_value = _make_response({})

        subtask_data = {
            "math": {
                "q1": {"question": "2+2?", "response": ["4"]},
            }
        }
        bm.add_model(subtask_data, model_name="new_model")
        assert "new_model" in bm.models

    def test_add_model_no_data_raises(self, mock_session):
        client = QuenchClient(api_key="qk_test")
        bm = Benchmark(client, "test")
        with pytest.raises(ValueError, match="No benchmark data"):
            bm.add_model({"m": {"s": {"q": {"response": ["x"]}}}})

    def test_remove_model(self, mock_session, sample_benchmark_data):
        bm = self._make_loaded_benchmark(mock_session, sample_benchmark_data)
        mock_session.delete.return_value = _make_response(None, text="")

        result = bm.remove_model("gpt4")
        assert result is bm
        assert "gpt4" not in bm.models

    def test_remove_nonexistent_model_raises(self, mock_session, sample_benchmark_data):
        bm = self._make_loaded_benchmark(mock_session, sample_benchmark_data)
        with pytest.raises(ValueError, match="not found"):
            bm.remove_model("nonexistent")


# ---------------------------------------------------------------------------
# Benchmark.delete
# ---------------------------------------------------------------------------

class TestBenchmarkDelete:

    def test_delete(self, mock_session, sample_benchmark_data):
        mock_session.get.return_value = _make_response({
            "benchmark_data": sample_benchmark_data, "metadata": {},
        })
        mock_session.post.return_value = _make_response(
            {"token": "jwt", "user_id": "u1"}
        )
        mock_session.delete.return_value = _make_response(None, text="")

        client = QuenchClient(api_key="qk_test")
        bm = client.benchmarks.load("doomed")
        bm.delete()

        assert bm._data is None


# ---------------------------------------------------------------------------
# Prediction & query selection
# ---------------------------------------------------------------------------

class TestPrediction:

    def _make_loaded_benchmark(self, mock_session, sample_benchmark_data):
        mock_session.get.return_value = _make_response({
            "benchmark_data": sample_benchmark_data,
            "metadata": {},
            "embeddings": {},
        })
        client = QuenchClient()
        return client.benchmarks.load("test_bench")

    def test_predict(self, mock_session, sample_benchmark_data):
        bm = self._make_loaded_benchmark(mock_session, sample_benchmark_data)

        mock_session.post.return_value = _make_response({
            "predicted_scores": {"my_model": 0.85},
            "confidence_interval": {"my_model": [0.80, 0.90]},
        })

        result = bm.predict({"my_model": {"math": {"q1": {"question": "x", "response": ["y"]}}}})
        assert result["predicted_scores"]["my_model"] == 0.85

    def test_predict_no_data_raises(self, mock_session):
        client = QuenchClient()
        bm = Benchmark(client, "empty")
        with pytest.raises(ValueError, match="No benchmark data"):
            bm.predict({"m": {}})

    def test_get_optimal_queries(self, mock_session, sample_benchmark_data):
        bm = self._make_loaded_benchmark(mock_session, sample_benchmark_data)

        # Second GET call (first was load)
        mock_session.get.return_value = _make_response({
            "queries": [{"subtask": "math", "query_id": "q1"}],
            "total_queries": 2,
            "estimated_error": 0.05,
        })

        result = bm.get_optimal_queries(budget=5)
        assert len(result["queries"]) == 1

    def test_estimate_query_budget(self, mock_session, sample_benchmark_data):
        bm = self._make_loaded_benchmark(mock_session, sample_benchmark_data)

        mock_session.get.return_value = _make_response({
            "estimated_queries": 15,
            "confidence_interval": [12, 18],
        })

        result = bm.estimate_query_budget(target_error=0.05)
        assert result["estimated_queries"] == 15


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------

class TestHelperMethods:

    def _make_benchmark(self, data):
        from quench.quench import QuenchClient
        bm = Benchmark(QuenchClient.__new__(QuenchClient), "test")
        bm._data = data
        return bm

    def test_models_property(self, sample_benchmark_data):
        bm = self._make_benchmark(sample_benchmark_data)
        assert "gpt4" in bm.models
        assert "metadata" not in bm.models

    def test_get_query_dictionary_all(self, sample_benchmark_data):
        bm = self._make_benchmark(sample_benchmark_data)
        queries = bm.get_query_dictionary()
        assert "math/q1" in queries
        assert "math/q2" in queries
        assert queries["math/q1"] == "What is 2+2?"

    def test_get_query_dictionary_subtask(self, sample_benchmark_data):
        bm = self._make_benchmark(sample_benchmark_data)
        queries = bm.get_query_dictionary(subtask="math")
        assert "q1" in queries
        assert "math/q1" not in queries

    def test_get_query_dictionary_invalid_subtask(self, sample_benchmark_data):
        bm = self._make_benchmark(sample_benchmark_data)
        with pytest.raises(ValueError, match="not found"):
            bm.get_query_dictionary(subtask="nonexistent")

    def test_get_model_metadata(self, sample_benchmark_data):
        bm = self._make_benchmark(sample_benchmark_data)
        meta = bm.get_model_metadata("gpt4")
        assert meta["version"] == "gpt-4-turbo"

    def test_get_model_metadata_not_found(self, sample_benchmark_data):
        bm = self._make_benchmark(sample_benchmark_data)
        with pytest.raises(ValueError, match="not found"):
            bm.get_model_metadata("nonexistent")

    def test_summary(self, mock_session, sample_benchmark_data):
        mock_session.get.return_value = _make_response({
            "models": {"gpt4": {"score": 1.0}},
        })
        client = QuenchClient()
        bm = Benchmark(client, "test")
        bm._data = sample_benchmark_data
        result = bm.summary()
        assert "models" in result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_validate_missing_response(self):
        invalid = {"model1": {"sub1": {"q1": {"question": "test"}}}}
        with pytest.raises(ValueError, match="Missing 'response'"):
            _validate_response_data(invalid)

    def test_validate_valid_data(self, sample_benchmark_data):
        # Should not raise
        _validate_response_data(sample_benchmark_data)

    def test_validate_model_data_not_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            _validate_model_data("not a dict", "bad_model")

    def test_validate_subtask_not_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            _validate_model_data({"sub1": "not a dict"}, "model1")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:

    def test_rate_limit_error(self, mock_session):
        mock_session.get.return_value = _make_response(
            {"error": "Too many requests"}, status_code=429, ok=False
        )
        client = QuenchClient()
        with pytest.raises(RateLimitError):
            client.benchmarks.load("test")

    def test_generic_api_error(self, mock_session):
        mock_session.get.return_value = _make_response(
            {"error": "Internal server error"}, status_code=500, ok=False
        )
        client = QuenchClient()
        with pytest.raises(QuenchAPIError):
            client.benchmarks.load("test")

    def test_error_includes_message(self, mock_session):
        mock_session.get.return_value = _make_response(
            {"error": "Benchmark limit exceeded"}, status_code=400, ok=False
        )
        client = QuenchClient()
        with pytest.raises(QuenchAPIError, match="Benchmark limit exceeded"):
            client.benchmarks.load("test")


# ---------------------------------------------------------------------------
# Module-level API (quench.Benchmark.load)
# ---------------------------------------------------------------------------

class TestResponseUnwrapping:

    def test_list_unwraps_benchmarks(self, mock_session):
        mock_session.get.return_value = _make_response({
            "benchmarks": [{"name": "b1"}, {"name": "b2"}],
            "total": 2,
        })
        client = QuenchClient()
        result = client.benchmarks.list()
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["name"] == "b1"

    def test_benchmark_categories_unwraps(self, mock_session):
        mock_session.get.return_value = _make_response({
            "categories": ["math", "coding", "safety"],
        })
        from quench.quench import _reset_default_client, benchmark_categories
        _reset_default_client()
        # Use explicit client
        client = QuenchClient()
        result = benchmark_categories(client=client)
        assert isinstance(result, list)
        assert "math" in result

    def test_embedding_providers_returns_dict(self, mock_session):
        mock_session.get.return_value = _make_response({
            "providers": {"google": {}},
            "default_provider": "google",
            "default_model": "gemini-embedding-001",
        })
        from quench.quench import _reset_default_client, embedding_providers
        _reset_default_client()
        client = QuenchClient()
        result = embedding_providers(client=client)
        assert isinstance(result, dict)
        assert "providers" in result


# ---------------------------------------------------------------------------
# Async creation and wait_until_ready
# ---------------------------------------------------------------------------

class TestAsyncCreation:

    def test_wait_until_ready_immediate(self, mock_session, sample_benchmark_data):
        """Benchmark already ready — should return immediately."""
        client = QuenchClient()
        bm = Benchmark(client, "test")
        bm._data = sample_benchmark_data
        bm._status = "ready"

        result = bm.wait_until_ready(timeout=1)
        assert result is bm

    def test_wait_until_ready_polls(self, mock_session, sample_benchmark_data):
        """Benchmark transitions from processing to ready."""
        mock_session.get.side_effect = [
            _make_response({"status": "processing"}),
            _make_response({"status": "ready"}),
        ]
        client = QuenchClient()
        bm = Benchmark(client, "test")
        bm._data = sample_benchmark_data
        bm._status = "processing"

        result = bm.wait_until_ready(timeout=10, poll_interval=0)
        assert result is bm
        assert bm.status == "ready"

    def test_wait_until_ready_failed(self, mock_session, sample_benchmark_data):
        """Benchmark processing fails."""
        mock_session.get.return_value = _make_response({"status": "failed"})

        client = QuenchClient()
        bm = Benchmark(client, "test")
        bm._data = sample_benchmark_data
        bm._status = "processing"

        with pytest.raises(QuenchAPIError, match="processing failed"):
            bm.wait_until_ready(timeout=10, poll_interval=0)


# ---------------------------------------------------------------------------
# Presigned upload
# ---------------------------------------------------------------------------

class TestPresignedUpload:

    def test_all_data_uses_r2(self, mock_session, sample_benchmark_data):
        """All benchmarks use per-model R2 upload (threshold=0)."""
        mock_session.post.side_effect = [
            _make_response({"token": "jwt", "user_id": "u1"}),
            _make_response({
                "upload_url": "https://r2/upload", "storage_key": "uploads/gpt4.json", "expires_in": 3600,
            }),
            _make_response({"benchmark_id": "b1", "status": "processing"}),
        ]
        mock_session.put.return_value = _make_response(None, text="")
        client = QuenchClient(api_key="qk_test")
        client.benchmarks.create("small", sample_benchmark_data)

        # Should have: login + presign + create = 3 posts, 1 put
        assert mock_session.post.call_count == 3
        assert mock_session.put.call_count == 1
        # Create call should have model_storage_keys, not benchmark_data
        create_call = mock_session.post.call_args_list[2]
        payload = create_call[1].get("json", {})
        assert "model_storage_keys" in payload
        assert "benchmark_data" not in payload

    def test_large_data_uses_presigned(self, mock_session):
        """Large benchmarks use presigned upload."""
        # Build data > 50MB threshold
        from quench.quench import _BenchmarkNamespace
        original_threshold = _BenchmarkNamespace._LARGE_FILE_THRESHOLD

        try:
            # Temporarily lower threshold for testing
            _BenchmarkNamespace._LARGE_FILE_THRESHOLD = 10  # 10 bytes

            mock_session.post.side_effect = [
                _make_response({"token": "jwt", "user_id": "u1"}),  # login
                # One presign per model (1 model in this data)
                _make_response({
                    "upload_url": "https://r2.example.com/upload",
                    "storage_key": "uploads/m1.json", "expires_in": 3600,
                }),
                _make_response({"benchmark_id": "b1", "status": "processing"}),
            ]
            mock_session.put.return_value = _make_response(None, text="")

            data = {"m1": {"sub": {"q1": {"question": "x", "response": ["y"]}}}}
            client = QuenchClient(api_key="qk_test")
            client.benchmarks.create("large", data)

            # Should have: login, 1 presign, create = 3 posts + 1 put
            assert mock_session.post.call_count == 3
            assert mock_session.put.call_count == 1

            # Create call should have model_storage_keys, not benchmark_data
            create_call = mock_session.post.call_args_list[2]
            payload = create_call[1].get("json", {})
            assert "model_storage_keys" in payload
            assert "benchmark_data" not in payload
        finally:
            _BenchmarkNamespace._LARGE_FILE_THRESHOLD = original_threshold


# ---------------------------------------------------------------------------
# Module-level API (quench.Benchmark.load)
# ---------------------------------------------------------------------------

class TestModuleLevelAPI:

    def test_module_level_load(self, mock_session, sample_benchmark_data, monkeypatch):
        import quench
        monkeypatch.setattr(quench, "api_key", None)

        mock_session.get.return_value = _make_response({
            "benchmark_data": sample_benchmark_data,
            "metadata": {},
        })

        # Reset cached default client
        from quench.quench import _reset_default_client
        _reset_default_client()

        bm = quench.Benchmark.load("public_bench")
        assert bm.name == "public_bench"
        assert "gpt4" in bm.models
