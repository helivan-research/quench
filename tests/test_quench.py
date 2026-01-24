"""
Tests for Quench client library

Tests cover:
- Authentication (login/logout)
- Benchmark operations (create, load, add_model, remove_model)
- Evaluation (predict)
- Helper methods (list_models, get_query_dictionary)
"""

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quench import Quench, login, logout, AuthenticationError, BenchmarkNotFoundError


class TestAuthentication:
    """Tests for login/logout functionality"""

    def test_login_success(self, mock_auth_state, mock_requests):
        """Test successful login"""
        mock_response = MagicMock()
        mock_response.json.return_value = {'user_id': 'user_abc123'}
        mock_response.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_response

        result = login(api_key='sk_test_key')

        assert result is True
        assert mock_auth_state['authenticated'] is True
        assert mock_auth_state['user_id'] == 'user_abc123'
        assert mock_auth_state['api_key'] == 'sk_test_key'

    def test_login_failure(self, mock_auth_state, mock_requests):
        """Test failed login"""
        from requests.exceptions import RequestException
        mock_requests.post.side_effect = RequestException('Connection failed')

        result = login(api_key='sk_invalid_key')

        assert result is False
        assert mock_auth_state['authenticated'] is False

    def test_login_custom_base_url(self, mock_auth_state, mock_requests):
        """Test login with custom base URL"""
        mock_response = MagicMock()
        mock_response.json.return_value = {'user_id': 'user_xyz'}
        mock_response.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_response

        login(api_key='sk_test', base_url='http://custom.api.com')

        mock_requests.post.assert_called_once()
        call_url = mock_requests.post.call_args[0][0]
        assert 'http://custom.api.com' in call_url

    def test_logout(self, authenticated_state):
        """Test logout clears state"""
        logout()

        from quench import quench as quench_module
        assert quench_module._AUTH_STATE['authenticated'] is False
        assert quench_module._AUTH_STATE['api_key'] is None
        assert quench_module._AUTH_STATE['user_id'] is None


class TestBenchmarkOperations:
    """Tests for benchmark CRUD operations"""

    def test_load_benchmark_public(self, mock_auth_state, mock_requests, sample_benchmark_data):
        """Test loading a public benchmark without auth"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'benchmark_data': sample_benchmark_data,
            'metadata': {'is_public': True}
        }
        mock_response.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_response

        quench = Quench().load_benchmark('public_benchmark')

        assert quench.benchmark_name == 'public_benchmark'
        assert quench.benchmark_data is not None
        assert 'gpt4' in quench.benchmark_data

    def test_load_benchmark_private_requires_auth(self, mock_auth_state, mock_requests):
        """Test loading private benchmark without auth raises error"""
        from requests.exceptions import HTTPError
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = HTTPError(response=mock_response)
        mock_requests.get.return_value = mock_response

        with pytest.raises(AuthenticationError):
            Quench().load_benchmark('private_benchmark')

    def test_load_benchmark_not_found(self, mock_auth_state, mock_requests):
        """Test loading non-existent benchmark raises error"""
        from requests.exceptions import HTTPError
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPError(response=mock_response)
        mock_requests.get.return_value = mock_response

        with pytest.raises(BenchmarkNotFoundError):
            Quench().load_benchmark('nonexistent')

    def test_create_benchmark_requires_auth(self, mock_auth_state, sample_benchmark_data):
        """Test creating benchmark without auth raises error"""
        quench = Quench()

        with pytest.raises(AuthenticationError):
            quench.create_benchmark(sample_benchmark_data, 'new_benchmark')

    def test_create_benchmark_success(self, authenticated_state, mock_requests, sample_benchmark_data):
        """Test successful benchmark creation"""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_response

        quench = Quench()
        result = quench.create_benchmark(sample_benchmark_data, 'my_new_benchmark')

        assert result is quench  # Returns self for chaining
        assert quench.benchmark_name == 'my_new_benchmark'
        assert quench.benchmark_data == sample_benchmark_data

    def test_add_model_requires_auth(self, mock_auth_state, sample_benchmark_data, sample_model_response):
        """Test adding model without auth raises error"""
        quench = Quench()
        quench.benchmark_data = sample_benchmark_data
        quench.benchmark_name = 'test_benchmark'

        with pytest.raises(AuthenticationError):
            quench.add_model(sample_model_response)

    def test_add_model_success(self, authenticated_state, mock_requests, sample_benchmark_data, sample_model_response):
        """Test successful model addition"""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_requests.patch.return_value = mock_response

        quench = Quench()
        quench.benchmark_data = sample_benchmark_data.copy()
        quench.benchmark_name = 'test_benchmark'

        result = quench.add_model(sample_model_response)

        assert result is quench
        assert 'claude' in quench.benchmark_data

    def test_add_model_no_benchmark_loaded(self, authenticated_state, sample_model_response):
        """Test adding model when no benchmark is loaded raises error"""
        quench = Quench()

        with pytest.raises(ValueError, match="No benchmark loaded"):
            quench.add_model(sample_model_response)

    def test_remove_model_success(self, authenticated_state, mock_requests, sample_benchmark_data):
        """Test successful model removal"""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_requests.delete.return_value = mock_response

        quench = Quench()
        quench.benchmark_data = sample_benchmark_data.copy()
        quench.benchmark_name = 'test_benchmark'

        result = quench.remove_model('gpt4')

        assert result is quench
        assert 'gpt4' not in quench.benchmark_data


class TestEvaluation:
    """Tests for predict/evaluation functionality"""

    def test_predict_public_benchmark(self, mock_auth_state, mock_requests, sample_benchmark_data):
        """Test evaluation of public benchmark without auth"""
        # Load benchmark
        mock_get = MagicMock()
        mock_get.json.return_value = {
            'benchmark_data': sample_benchmark_data,
            'metadata': {'is_public': True}
        }
        mock_get.raise_for_status = MagicMock()

        # Evaluation response
        mock_post = MagicMock()
        mock_post.json.return_value = {
            'overall_scores': {'accuracy': 1.0},
            'model_scores': {'gpt4': {'accuracy': 1.0}}
        }
        mock_post.raise_for_status = MagicMock()

        mock_requests.get.return_value = mock_get
        mock_requests.post.return_value = mock_post

        quench = Quench().load_benchmark('public_benchmark')
        results = quench.predict(metrics=['accuracy'])

        assert 'overall_scores' in results
        assert results['overall_scores']['accuracy'] == 1.0

    def test_predict_add_model_requires_auth(self, mock_auth_state, sample_benchmark_data, sample_model_response):
        """Test predict with add_model=True requires auth"""
        quench = Quench()
        quench.benchmark_data = sample_benchmark_data
        quench.benchmark_name = 'test'

        with pytest.raises(AuthenticationError):
            quench.predict(sample_model_response, add_model=True)

    def test_predict_no_data(self, mock_auth_state):
        """Test predict without data raises error"""
        quench = Quench()

        with pytest.raises(ValueError, match="No data to evaluate"):
            quench.predict()


class TestHelperMethods:
    """Tests for helper methods"""

    def test_list_models(self, sample_benchmark_data):
        """Test listing models in benchmark"""
        quench = Quench()
        quench.benchmark_data = sample_benchmark_data

        models = quench.list_models()

        assert 'gpt4' in models
        assert 'metadata' not in models

    def test_list_models_no_benchmark(self):
        """Test list_models without loaded benchmark raises error"""
        quench = Quench()

        with pytest.raises(ValueError, match="No benchmark loaded"):
            quench.list_models()

    def test_get_query_dictionary_all(self, sample_benchmark_data):
        """Test getting all queries"""
        quench = Quench()
        quench.benchmark_data = sample_benchmark_data

        queries = quench.get_query_dictionary()

        assert 'math/q1' in queries
        assert 'math/q2' in queries
        assert queries['math/q1'] == 'What is 2+2?'

    def test_get_query_dictionary_subtask(self, sample_benchmark_data):
        """Test getting queries for specific subtask"""
        quench = Quench()
        quench.benchmark_data = sample_benchmark_data

        queries = quench.get_query_dictionary(subtask='math')

        assert 'q1' in queries
        assert 'q2' in queries
        # Should NOT have subtask prefix
        assert 'math/q1' not in queries

    def test_get_query_dictionary_invalid_subtask(self, sample_benchmark_data):
        """Test getting queries for non-existent subtask raises error"""
        quench = Quench()
        quench.benchmark_data = sample_benchmark_data

        with pytest.raises(ValueError, match="not found"):
            quench.get_query_dictionary(subtask='nonexistent')

    def test_get_model_metadata(self, sample_benchmark_data):
        """Test getting model metadata"""
        quench = Quench()
        quench.benchmark_data = sample_benchmark_data

        metadata = quench.get_model_metadata('gpt4')

        assert metadata['version'] == 'gpt-4-turbo'


class TestValidation:
    """Tests for data validation"""

    def test_validate_missing_required_field(self, authenticated_state, mock_requests):
        """Test validation catches missing required fields"""
        invalid_data = {
            'model1': {
                'subtask1': {
                    'q1': {
                        'question': 'test',
                        # Missing: response, reasoning, response_score, score
                    }
                }
            }
        }

        quench = Quench()

        with pytest.raises(ValueError, match="Missing"):
            quench.create_benchmark(invalid_data, 'test')

    def test_validate_mismatched_lengths(self, authenticated_state, mock_requests):
        """Test validation catches mismatched array lengths"""
        invalid_data = {
            'model1': {
                'subtask1': {
                    'q1': {
                        'question': 'test',
                        'response': ['a', 'b'],
                        'reasoning': ['r1'],  # Should have 2 items
                        'response_score': [1.0, 0.8],
                        'score': 0.9,
                        'metadata': {}
                    }
                }
            }
        }

        quench = Quench()

        with pytest.raises(ValueError, match="Mismatched array lengths"):
            quench.create_benchmark(invalid_data, 'test')
