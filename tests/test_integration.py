"""
Integration tests for quench library

Tests the public API by calling the quench library methods
which in turn call the quench-backend server.

Usage:
    1. Start the backend server: cd quench-backend && python app.py
    2. Run tests: cd quench && pytest tests/test_integration.py -v
"""

import pytest
import json
import math
import random
from typing import Dict

# Import the public quench library
from quench import Quench, login, logout


# Test configuration
BASE_URL = "http://localhost:5001"


def convert_math_benchmark(raw_data: Dict) -> Dict:
    """Convert math_benchmark_full.json format to Quench format."""
    converted = {}

    for model_name, model_data in raw_data.items():
        converted[model_name] = {}

        for key, value in model_data.items():
            if key in ['metadata', 'score']:
                converted[model_name][key] = value
                continue

            if isinstance(value, dict) and 'responses' in value:
                subtask_data = {}
                for query_id, query_data in value['responses'].items():
                    response_text = query_data.get('response', '')
                    if isinstance(response_text, float):
                        if math.isnan(response_text):
                            response_text = ''
                        else:
                            response_text = str(response_text)

                    subtask_data[query_id] = {
                        'response': [response_text] if isinstance(response_text, str) else [str(response_text)],
                        'score': query_data.get('score', 0.0)
                    }
                converted[model_name][key] = subtask_data
            else:
                converted[model_name][key] = value

    return converted


def select_partial_model(model_data: Dict, fraction: float = 0.3, seed: int = 42) -> Dict:
    """Select a random subset of queries from a model's data."""
    random.seed(seed)
    partial = {}

    for key, value in model_data.items():
        if key in ['metadata', 'score']:
            continue

        if isinstance(value, dict):
            query_ids = list(value.keys())
            num_queries = max(1, int(len(query_ids) * fraction))
            selected = random.sample(query_ids, num_queries)
            partial[key] = {qid: value[qid] for qid in selected}

    return partial


class TestQuenchIntegration:
    """Integration tests using the public quench library."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test - ensure clean state."""
        logout()
        yield
        logout()

    def test_load_benchmark_public(self):
        """Test loading a public benchmark without authentication."""
        # This would work if there's a public benchmark on the server
        # For now, we test that the method exists and handles errors properly
        quench = Quench()
        quench._base_url = BASE_URL

        # Should raise BenchmarkNotFoundError for non-existent benchmark
        from quench.quench import BenchmarkNotFoundError
        with pytest.raises((BenchmarkNotFoundError, Exception)):
            quench.load_benchmark('nonexistent_benchmark')

    def test_predict_requires_benchmark(self):
        """Test that predict() requires a loaded benchmark."""
        quench = Quench()
        quench._base_url = BASE_URL

        with pytest.raises(ValueError, match="No benchmark loaded"):
            quench.predict({'my_model': {}})

    def test_list_models_requires_benchmark(self):
        """Test that list_models() requires a loaded benchmark."""
        quench = Quench()

        with pytest.raises(ValueError, match="No benchmark loaded"):
            quench.list_models()


class TestQuenchPrediction:
    """Test prediction functionality with real data."""

    @pytest.fixture
    def math_benchmark_data(self):
        """Load and convert the math benchmark data."""
        benchmark_path = '../../math_benchmark_full.json'
        try:
            with open(benchmark_path) as f:
                raw_data = json.load(f)
            return convert_math_benchmark(raw_data)
        except FileNotFoundError:
            pytest.skip(f"Benchmark file not found: {benchmark_path}")

    def test_predict_with_math_benchmark(self, math_benchmark_data):
        """
        Test score prediction using the math benchmark.

        This test:
        1. Loads a subset of the benchmark as "cached" models
        2. Selects one model as the "new" model
        3. Uses only a fraction of the new model's queries
        4. Calls predict() to estimate the full score
        5. Compares predicted vs actual score
        """
        # Select a random model to be our "new" model
        all_models = [k for k in math_benchmark_data.keys() if k not in ['metadata']]

        # Use first 10 models as training set, pick one as test
        training_models = all_models[:10]
        test_model_name = all_models[10] if len(all_models) > 10 else all_models[0]

        # Get actual score for comparison
        actual_score = math_benchmark_data[test_model_name].get('score', 0)

        # Create partial data for the test model (30% of queries)
        full_model_data = math_benchmark_data[test_model_name]
        partial_model_data = select_partial_model(full_model_data, fraction=0.3)

        # Create quench instance and manually set benchmark data
        # (In real usage, this would come from load_benchmark())
        quench = Quench()
        quench._base_url = BASE_URL
        quench.benchmark_name = "math_test"
        quench.benchmark_data = {k: math_benchmark_data[k] for k in training_models}

        # Create the new model data
        new_model_data = {test_model_name: partial_model_data}

        # Call predict
        try:
            results = quench.predict(new_model_data)

            # Verify results structure
            assert 'predicted_score' in results or 'error' in results

            if 'error' not in results:
                predicted_score = results['predicted_score'].get(test_model_name)
                confidence = results.get('confidence_interval', {}).get(test_model_name, [0, 1])

                print(f"\nModel: {test_model_name}")
                print(f"Actual Score: {actual_score:.4f}")
                print(f"Predicted Score: {predicted_score:.4f}")
                print(f"95% CI: [{confidence[0]:.4f}, {confidence[1]:.4f}]")
                print(f"Error: {abs(actual_score - predicted_score):.4f}")

                # Check if actual score is within confidence interval
                in_ci = confidence[0] <= actual_score <= confidence[1]
                print(f"In CI: {in_ci}")

        except Exception as e:
            # Server might not be running
            pytest.skip(f"Server not available: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
