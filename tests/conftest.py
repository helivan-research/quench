"""
Pytest fixtures for Quench SDK tests.
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def sample_benchmark_data():
    """Sample benchmark data following the Quench schema."""
    return {
        "gpt4": {
            "math": {
                "q1": {
                    "question": "What is 2+2?",
                    "response": ["4"],
                    "reasoning": ["Basic addition"],
                    "response_score": [1.0],
                    "score": 1.0,
                    "metadata": {},
                },
                "q2": {
                    "question": "What is 3*3?",
                    "response": ["9"],
                    "reasoning": ["Basic multiplication"],
                    "response_score": [1.0],
                    "score": 1.0,
                    "metadata": {},
                },
                "query_score": [1.0, 1.0],
                "score": 1.0,
                "metadata": {},
            },
            "metadata": {"version": "gpt-4-turbo"},
        },
        "metadata": {"is_public": True},
    }


@pytest.fixture
def sample_model_response():
    """Sample single-model response data."""
    return {
        "claude": {
            "math": {
                "q1": {
                    "question": "What is 2+2?",
                    "response": ["4"],
                    "reasoning": ["Addition of 2 and 2"],
                    "response_score": [1.0],
                    "score": 1.0,
                    "metadata": {},
                },
                "query_score": [1.0],
                "score": 1.0,
                "metadata": {},
            },
            "metadata": {"version": "claude-3.5"},
        }
    }


@pytest.fixture
def mock_session():
    """Mock requests.Session used by QuenchClient."""
    with patch("quench.quench._build_session") as mock_build:
        session = MagicMock()
        mock_build.return_value = session
        yield session


def _make_response(json_data=None, status_code=200, ok=True, text=""):
    """Helper to create a mock response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = ok
    resp.text = text or ""
    if json_data is not None:
        resp.json.return_value = json_data
    return resp
