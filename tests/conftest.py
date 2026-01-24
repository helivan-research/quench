"""
Pytest fixtures for Quench client library tests
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_auth_state():
    """Reset authentication state before each test"""
    from quench import quench as quench_module

    # Store original state
    original_state = quench_module._AUTH_STATE.copy()

    # Reset to unauthenticated
    quench_module._AUTH_STATE.update({
        'api_key': None,
        'user_id': None,
        'authenticated': False,
        'base_url': None
    })

    yield quench_module._AUTH_STATE

    # Restore original state
    quench_module._AUTH_STATE.update(original_state)


@pytest.fixture
def authenticated_state():
    """Set up authenticated state for tests"""
    from quench import quench as quench_module

    original_state = quench_module._AUTH_STATE.copy()

    quench_module._AUTH_STATE.update({
        'api_key': 'sk_test_key_12345',
        'user_id': 'user_test123',
        'authenticated': True,
        'base_url': 'http://localhost:5000'
    })

    yield quench_module._AUTH_STATE

    quench_module._AUTH_STATE.update(original_state)


@pytest.fixture
def sample_benchmark_data():
    """Sample benchmark data following Quench schema"""
    return {
        'gpt4': {
            'math': {
                'q1': {
                    'question': 'What is 2+2?',
                    'response': ['4'],
                    'reasoning': ['Basic addition'],
                    'response_score': [1.0],
                    'score': 1.0,
                    'metadata': {}
                },
                'q2': {
                    'question': 'What is 3*3?',
                    'response': ['9'],
                    'reasoning': ['Basic multiplication'],
                    'response_score': [1.0],
                    'score': 1.0,
                    'metadata': {}
                },
                'query_score': [1.0, 1.0],
                'score': 1.0,
                'metadata': {}
            },
            'metadata': {'version': 'gpt-4-turbo'}
        },
        'metadata': {'is_public': True}
    }


@pytest.fixture
def sample_model_response():
    """Sample single model response data"""
    return {
        'claude': {
            'math': {
                'q1': {
                    'question': 'What is 2+2?',
                    'response': ['4'],
                    'reasoning': ['Addition of 2 and 2'],
                    'response_score': [1.0],
                    'score': 1.0,
                    'metadata': {}
                },
                'query_score': [1.0],
                'score': 1.0,
                'metadata': {}
            },
            'metadata': {'version': 'claude-3.5'}
        }
    }


@pytest.fixture
def mock_requests():
    """Mock requests library for HTTP calls"""
    with patch('quench.quench.requests') as mock:
        yield mock
