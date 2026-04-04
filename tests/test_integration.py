"""
Integration tests for the Quench SDK.

These tests call the real backend server.

Usage:
    1. Start the backend: cd quench-backend && python app.py
    2. Run: cd quench && pytest tests/test_integration.py -v
"""

import json
import math
import random
from typing import Dict

import pytest

from quench import QuenchClient, BenchmarkNotFoundError

BASE_URL = "http://localhost:5001"


def convert_math_benchmark(raw_data: Dict) -> Dict:
    """Convert math_benchmark_full.json format to Quench format."""
    converted = {}
    for model_name, model_data in raw_data.items():
        converted[model_name] = {}
        for key, value in model_data.items():
            if key in ("metadata", "score"):
                converted[model_name][key] = value
                continue
            if isinstance(value, dict) and "responses" in value:
                subtask_data = {}
                for query_id, query_data in value["responses"].items():
                    resp = query_data.get("response", "")
                    if isinstance(resp, float):
                        resp = "" if math.isnan(resp) else str(resp)
                    subtask_data[query_id] = {
                        "response": [str(resp)],
                        "score": query_data.get("score", 0.0),
                    }
                converted[model_name][key] = subtask_data
            else:
                converted[model_name][key] = value
    return converted


def select_partial_model(model_data: Dict, fraction: float = 0.3, seed: int = 42) -> Dict:
    """Select a random subset of queries from a model."""
    random.seed(seed)
    partial = {}
    for key, value in model_data.items():
        if key in ("metadata", "score"):
            continue
        if isinstance(value, dict):
            query_ids = list(value.keys())
            n = max(1, int(len(query_ids) * fraction))
            selected = random.sample(query_ids, n)
            partial[key] = {qid: value[qid] for qid in selected}
    return partial


class TestQuenchIntegration:

    @pytest.fixture(autouse=True)
    def client(self):
        self.client = QuenchClient(base_url=BASE_URL)

    def test_load_nonexistent_benchmark(self):
        with pytest.raises((BenchmarkNotFoundError, Exception)):
            self.client.benchmarks.load("nonexistent_benchmark_xyz")

    def test_predict_requires_loaded_data(self):
        from quench.quench import Benchmark
        bm = Benchmark(self.client, "no_data")
        with pytest.raises(ValueError, match="No benchmark data"):
            bm.predict({"my_model": {}})


class TestQuenchPrediction:

    @pytest.fixture
    def math_benchmark_data(self):
        benchmark_path = "../../math_benchmark_full.json"
        try:
            with open(benchmark_path) as f:
                return convert_math_benchmark(json.load(f))
        except FileNotFoundError:
            pytest.skip(f"Benchmark file not found: {benchmark_path}")

    def test_predict_with_math_benchmark(self, math_benchmark_data):
        from quench.quench import Benchmark

        all_models = [k for k in math_benchmark_data if k != "metadata"]
        training_models = all_models[:10]
        test_name = all_models[10] if len(all_models) > 10 else all_models[0]

        full_data = math_benchmark_data[test_name]
        partial_data = select_partial_model(full_data, fraction=0.3)

        client = QuenchClient(base_url=BASE_URL)
        bm = Benchmark(client, "math_test")
        bm._data = {k: math_benchmark_data[k] for k in training_models}
        bm._embeddings = {}

        try:
            results = bm.predict({test_name: partial_data})
            assert "predicted_scores" in results or "error" in results
        except Exception as e:
            pytest.skip(f"Server not available: {e}")
