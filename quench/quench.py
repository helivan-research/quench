"""
Quench - Evaluation framework for generative models

This module provides the core Quench class and authentication utilities
for evaluating generative model responses against benchmarks.
"""

import requests
from typing import Dict, List, Optional, Any
from datetime import datetime


class AuthenticationError(Exception):
    """Raised when user is not authenticated"""
    pass


class BenchmarkNotFoundError(Exception):
    """Raised when benchmark doesn't exist"""
    pass


# Global authentication state
_AUTH_STATE = {
    'api_key': None,
    'token': None,
    'user_id': None,
    'authenticated': False,
    'base_url': None
}


def login(api_key: str, base_url: str = "https://quench.helivan.io/api") -> bool:
    """Authenticate user with API key
    
    Args:
        api_key: User's API key
        base_url: Base URL for Quench API (default: https://quench.helivan.io/api)
        
    Returns:
        bool: True if authentication successful, False otherwise
        
    Example:
        >>> from quench import login
        >>> login(api_key="sk_abc123...")
        ✓ Authenticated as user user_xyz
        True
    """
    global _AUTH_STATE
    
    try:
        response = requests.post(
            f"{base_url}/auth/login",
            json={'api_key': api_key}
        )
        response.raise_for_status()

        data = response.json()
        _AUTH_STATE['api_key'] = api_key
        _AUTH_STATE['token'] = data.get('token')
        _AUTH_STATE['user_id'] = data.get('user_id')
        _AUTH_STATE['authenticated'] = True
        _AUTH_STATE['base_url'] = base_url

        print(f"✓ Authenticated as user {_AUTH_STATE['user_id']}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Authentication failed: {e}")
        _AUTH_STATE['authenticated'] = False
        return False


def logout():
    """Clear authentication state

    Example:
        >>> from quench import logout
        >>> logout()
        ✓ Logged out
    """
    global _AUTH_STATE
    _AUTH_STATE = {
        'api_key': None,
        'token': None,
        'user_id': None,
        'authenticated': False,
        'base_url': None
    }
    print("✓ Logged out")


def _require_auth(func):
    """Decorator to enforce authentication for class methods"""
    def wrapper(self, *args, **kwargs):
        if not _AUTH_STATE['authenticated']:
            raise AuthenticationError(
                f"{func.__name__} requires authentication. "
                "Please call login(api_key) first."
            )
        return func(self, *args, **kwargs)
    return wrapper


class Quench:
    """Main class for benchmark management and model evaluation
    
    The Quench class provides methods to create, load, and manage benchmarks,
    as well as evaluate generative model responses against those benchmarks.
    
    All methods require authentication via the login() function.
    
    Example:
        >>> from quench import Quench, login
        >>> login(api_key="sk_abc123...")
        >>> quench = Quench().load_benchmark('my_benchmark')
        >>> models = quench.list_models()
        >>> queries = quench.get_query_dictionary()
    """
    
    def __init__(self):
        """Initialize Quench evaluator

        Note: Authentication must be done via login() before using this class
        """
        self.benchmark_name = None
        self.benchmark_data = None
        self.benchmark_metadata = None
        self.benchmark_embeddings = None  # Pre-computed embeddings for efficiency
        self._base_url = _AUTH_STATE.get('base_url', 'https://quench.helivan.io/api')
    
    def load_benchmark(self, benchmark_name: str):
        """Load a benchmark from remote storage
        
        Public benchmarks can be loaded without authentication.
        Private benchmarks require authentication via login().
        
        Args:
            benchmark_name: Name of the benchmark to load
            
        Returns:
            self (for method chaining)
            
        Raises:
            AuthenticationError: If benchmark is private and user not authenticated
            BenchmarkNotFoundError: If benchmark doesn't exist
            
        Example:
            >>> # Load public benchmark (no auth required)
            >>> quench = Quench().load_benchmark('public_math_eval')
            ✓ Loaded public benchmark 'public_math_eval' with 3 model(s)
            
            >>> # Load private benchmark (auth required)
            >>> login(api_key="sk_abc123...")
            >>> quench = Quench().load_benchmark('my_private_benchmark')
            ✓ Loaded benchmark 'my_private_benchmark' with 2 model(s)
        """
        try:
            # Prepare headers - include auth token if available
            headers = {}
            if _AUTH_STATE['authenticated']:
                headers['Authorization'] = f"Bearer {_AUTH_STATE['token']}"
            
            response = requests.get(
                f"{self._base_url}/benchmarks/{benchmark_name}",
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            self.benchmark_name = benchmark_name
            self.benchmark_data = data.get('benchmark_data', {})
            self.benchmark_metadata = data.get('metadata', {})
            self.benchmark_embeddings = data.get('embeddings', {})  # Reuse stored embeddings
            
            is_public = self.benchmark_metadata.get('is_public', False)
            num_models = len([k for k in self.benchmark_data.keys() if k != 'metadata'])
            
            visibility = "public " if is_public else ""
            print(f"✓ Loaded {visibility}benchmark '{benchmark_name}' with {num_models} model(s)")
            
            return self
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise BenchmarkNotFoundError(
                    f"Benchmark '{benchmark_name}' not found"
                )
            elif e.response.status_code == 401 or e.response.status_code == 403:
                raise AuthenticationError(
                    f"Benchmark '{benchmark_name}' is private. "
                    "Please call login(api_key) first."
                )
            raise
    
    @_require_auth
    def create_benchmark(self, response_json: Dict, benchmark_name: str, **kwargs):
        """Create a new benchmark from initial model responses
        
        Args:
            response_json: Dict following the Quench schema with model(s) responses
            benchmark_name: Name for the new benchmark
            **kwargs: Additional benchmark configuration (e.g., description, tags)
            
        Returns:
            self (for method chaining)
            
        Raises:
            AuthenticationError: If not authenticated
            ValueError: If response_json doesn't follow expected schema
            
        Example:
            >>> benchmark_data = {
            ...     'gpt4': {
            ...         'math': {
            ...             'q1': {
            ...                 'question': '2+2=?',
            ...                 'response': ['4'],
            ...                 'reasoning': ['basic addition'],
            ...                 'response_score': [1.0],
            ...                 'score': 1.0,
            ...                 'metadata': {}
            ...             },
            ...             'query_score': [1.0],
            ...             'score': 1.0,
            ...             'metadata': {}
            ...         },
            ...         'metadata': {'version': 'gpt-4'}
            ...     },
            ...     'metadata': {}
            ... }
            >>> quench = Quench().create_benchmark(benchmark_data, 'my_benchmark')
            ✓ Created benchmark 'my_benchmark' with 1 model(s)
        """
        # Validate schema
        self._validate_response_json(response_json)
        
        # Prepare benchmark metadata
        metadata = response_json.get('metadata', {})
        metadata.update({
            'name': benchmark_name,
            'created_at': datetime.now().isoformat(),
            'created_by': _AUTH_STATE['user_id'],
            **kwargs
        })
        
        # Upload to remote storage
        payload = {
            'benchmark_name': benchmark_name,
            'benchmark_data': response_json,
            'metadata': metadata
        }

        response = requests.post(
            f"{self._base_url}/benchmarks",
            headers={'Authorization': f"Bearer {_AUTH_STATE['token']}"},
            json=payload
        )
        response.raise_for_status()
        
        # Store locally
        self.benchmark_name = benchmark_name
        self.benchmark_data = response_json
        self.benchmark_metadata = metadata
        
        num_models = len([k for k in response_json.keys() if k != 'metadata'])
        print(f"✓ Created benchmark '{benchmark_name}' with {num_models} model(s)")
        
        return self
    
    @_require_auth
    def add_model(self, response_json: Dict, model_name: Optional[str] = None):
        """Add a new model's responses to existing benchmark
        
        Args:
            response_json: Either:
                - Full schema dict with one model
                - Subtask-level dict (requires model_name parameter)
            model_name: Required if response_json is subtask-level data
            
        Returns:
            self (for method chaining)
            
        Raises:
            AuthenticationError: If not authenticated
            ValueError: If no benchmark is loaded or invalid data format
            
        Example:
            >>> new_model = {
            ...     'claude': {
            ...         'math': {
            ...             'q1': {...},
            ...             'query_score': [1.0],
            ...             'score': 1.0,
            ...             'metadata': {}
            ...         },
            ...         'metadata': {'version': 'claude-3.5'}
            ...     }
            ... }
            >>> quench.add_model(new_model)
            ✓ Added model 'claude' to benchmark 'my_benchmark'
        """
        if self.benchmark_data is None:
            raise ValueError(
                "No benchmark loaded. Use load_benchmark() or create_benchmark() first."
            )
        
        # Determine model name and data
        if model_name and model_name not in response_json:
            # User provided subtask-level data
            model_data = response_json
            final_model_name = model_name
        else:
            # User provided full model data
            model_keys = [k for k in response_json.keys() if k != 'metadata']
            if len(model_keys) != 1:
                raise ValueError(
                    "response_json must contain exactly one model "
                    "(or use model_name parameter)"
                )
            final_model_name = model_keys[0]
            model_data = response_json[final_model_name]
        
        # Validate the model data structure
        self._validate_model_data(model_data, final_model_name)
        
        # Update local state
        self.benchmark_data[final_model_name] = model_data
        
        # Sync to remote
        response = requests.patch(
            f"{self._base_url}/benchmarks/{self.benchmark_name}/models/{final_model_name}",
            headers={'Authorization': f"Bearer {_AUTH_STATE['token']}"},
            json={'model_data': model_data}
        )
        response.raise_for_status()
        
        print(f"✓ Added model '{final_model_name}' to benchmark '{self.benchmark_name}'")
        
        return self
    
    @_require_auth
    def remove_model(self, model_name: str):
        """Remove a model from the benchmark
        
        Args:
            model_name: Name of the model to remove
            
        Returns:
            self (for method chaining)
            
        Raises:
            AuthenticationError: If not authenticated
            ValueError: If model doesn't exist
            
        Example:
            >>> quench.remove_model('gpt4')
            ✓ Removed model 'gpt4' from benchmark 'my_benchmark'
        """
        if self.benchmark_data is None:
            raise ValueError("No benchmark loaded")
        
        if model_name not in self.benchmark_data:
            raise ValueError(f"Model '{model_name}' not found in benchmark")
        
        # Remove locally
        del self.benchmark_data[model_name]
        
        # Sync to remote
        response = requests.delete(
            f"{self._base_url}/benchmarks/{self.benchmark_name}/models/{model_name}",
            headers={'Authorization': f"Bearer {_AUTH_STATE['token']}"}
        )
        response.raise_for_status()
        
        print(f"✓ Removed model '{model_name}' from benchmark '{self.benchmark_name}'")
        
        return self
    
    def predict(self, response_json: Dict):
        """Predict the overall benchmark score for a partially-evaluated model

        Given a new model that has only responded to a subset of the benchmark
        queries, predict what its overall benchmark score would be by comparing
        its response embeddings to cached models using MDS and regression.

        Args:
            response_json: New model responses. Can contain responses to only
                          a subset of subtasks/queries. Format:
                          {
                              'model_name': {
                                  'subtask': {
                                      'query_id': {
                                          'question': '...',
                                          'response': ['...'],
                                          ...
                                      }
                                  }
                              }
                          }

        Returns:
            Dict with prediction results:
                - predicted_score: {model_name: score} - the predicted overall score
                - confidence_interval: {model_name: [lower, upper]} - 95% CI
                - similar_models: Most similar cached models by embedding distance
                - cached_model_scores: Known scores of cached models
                - overlap_info: Query overlap statistics (coverage)
                - mds_coordinates: Low-dim model representations

        Raises:
            ValueError: If no benchmark is loaded or response_json is None

        Example:
            >>> # Load benchmark with cached model results
            >>> quench = Quench().load_benchmark('math_eval')
            >>> print(quench.list_models())  # ['gpt4', 'claude', 'gemini']

            >>> # Predict score for new model with partial responses
            >>> partial_responses = {
            ...     'my_model': {
            ...         'algebra': {  # Only answered algebra subtask
            ...             'q1': {'question': '2+2?', 'response': ['4'], ...},
            ...         }
            ...         # Note: No 'geometry' subtask - this is partial
            ...     }
            ... }
            >>> results = quench.predict(partial_responses)
            >>> print(results['predicted_score'])
            {'my_model': 0.87}
            >>> print(results['confidence_interval'])
            {'my_model': [0.82, 0.92]}
        """
        if self.benchmark_data is None:
            raise ValueError(
                "No benchmark loaded. Call load_benchmark() first."
            )

        if response_json is None:
            raise ValueError("response_json is required for prediction.")

        # Prepare headers - include auth token if available
        headers = {}
        if _AUTH_STATE['authenticated']:
            headers['Authorization'] = f"Bearer {_AUTH_STATE['token']}"

        # Build payload for prediction
        payload = {
            'benchmark_data': self.benchmark_data,  # Cached benchmark with scores
            'new_model_data': response_json,  # New model to predict (partial)
            'benchmark_embeddings': self.benchmark_embeddings  # Reuse stored embeddings
        }

        response = requests.post(
            f"{self._base_url}/evaluate/predict",
            headers=headers,
            json=payload
        )
        response.raise_for_status()

        results = response.json()

        print(f"✓ Prediction complete")
        return results

    def get_optimal_queries(self, budget: int = 10) -> Dict:
        """Get the most informative queries for predicting benchmark scores

        Selects queries using a combination of:
        - Correlation: How well query scores correlate with overall model scores
        - Variance: How much the query discriminates between models
        - Diversity: Coverage across different query types

        Args:
            budget: Number of queries to select (default 10, max 100)

        Returns:
            Dict with:
                - queries: List of optimal queries with importance scores
                - total_queries: Total queries in benchmark
                - estimated_error: Estimated prediction error with this budget
                - model_count: Number of models used for analysis

        Raises:
            ValueError: If no benchmark is loaded

        Example:
            >>> quench = Quench().load_benchmark('math_eval')
            >>> optimal = quench.get_optimal_queries(budget=15)
            >>> print(optimal['queries'][:3])
            [{'subtask': 'algebra', 'query_id': 'q42', 'importance': 0.92, ...}, ...]
            >>> print(f"Estimated error: {optimal['estimated_error']:.3f}")
            Estimated error: 0.085
        """
        if self.benchmark_name is None:
            raise ValueError(
                "No benchmark loaded. Call load_benchmark() first."
            )

        # Prepare headers - include auth token if available
        headers = {}
        if _AUTH_STATE['authenticated']:
            headers['Authorization'] = f"Bearer {_AUTH_STATE['token']}"

        response = requests.get(
            f"{self._base_url}/benchmarks/{self.benchmark_name}/optimal-queries",
            headers=headers,
            params={'budget': budget}
        )
        response.raise_for_status()

        results = response.json()
        print(f"✓ Selected {len(results.get('queries', []))} optimal queries")
        return results

    def estimate_query_budget(self, target_error: float = 0.05) -> Dict:
        """Estimate the number of queries needed for a target prediction error

        Uses leave-one-model-out cross-validation to estimate how prediction
        error decreases as more queries are included.

        Args:
            target_error: Target MAE for predictions (default 0.05)

        Returns:
            Dict with:
                - estimated_queries: Estimated number of queries needed
                - confidence_interval: [lower, upper] bound estimates
                - error_curve: List of (budget, MAE, std) tuples
                - total_queries: Total queries in benchmark

        Raises:
            ValueError: If no benchmark is loaded

        Example:
            >>> quench = Quench().load_benchmark('math_eval')
            >>> budget_info = quench.estimate_query_budget(target_error=0.05)
            >>> print(f"Need {budget_info['estimated_queries']} queries for 5% error")
            Need 25 queries for 5% error
            >>> print(f"Confidence interval: {budget_info['confidence_interval']}")
            Confidence interval: [22, 30]
        """
        if self.benchmark_name is None:
            raise ValueError(
                "No benchmark loaded. Call load_benchmark() first."
            )

        # Prepare headers - include auth token if available
        headers = {}
        if _AUTH_STATE['authenticated']:
            headers['Authorization'] = f"Bearer {_AUTH_STATE['token']}"

        response = requests.get(
            f"{self._base_url}/benchmarks/{self.benchmark_name}/query-budget",
            headers=headers,
            params={'target_error': target_error}
        )
        response.raise_for_status()

        results = response.json()
        print(f"✓ Estimated {results.get('estimated_queries', 'N/A')} queries for {target_error:.1%} error")
        return results

    def list_models(self) -> List[str]:
        """Get list of models in current benchmark
        
        Returns:
            List of model names
            
        Raises:
            ValueError: If no benchmark is loaded
            
        Example:
            >>> quench.list_models()
            ['gpt4', 'claude-sonnet', 'gemini']
        """
        if self.benchmark_data is None:
            raise ValueError("No benchmark loaded. Use load_benchmark() first.")
        
        return [k for k in self.benchmark_data.keys() if k != 'metadata']
    
    def get_query_dictionary(self, subtask: Optional[str] = None) -> Dict[str, str]:
        """Get mapping of query IDs to questions
        
        Args:
            subtask: Optional subtask name to filter queries. 
                    If None, returns queries from all subtasks.
        
        Returns:
            Dict mapping query_id to question text.
            - If subtask is None: {subtask/query_id: question}
            - If subtask specified: {query_id: question}
            
        Raises:
            ValueError: If no benchmark is loaded or subtask doesn't exist
            
        Example:
            >>> # Get all queries
            >>> all_queries = quench.get_query_dictionary()
            >>> print(all_queries)
            {'math/q1': '2+2=?', 'math/q2': '3*3=?', 'logic/q1': 'If A then B...'}
            
            >>> # Get queries for specific subtask
            >>> math_queries = quench.get_query_dictionary(subtask='math')
            >>> print(math_queries)
            {'q1': '2+2=?', 'q2': '3*3=?'}
        """
        if self.benchmark_data is None:
            raise ValueError("No benchmark loaded. Use load_benchmark() first.")
        
        query_dict = {}
        
        # Get any model's data (they should all have the same query structure)
        model_names = self.list_models()
        if not model_names:
            return query_dict
        
        # Use first model as reference
        reference_model = self.benchmark_data[model_names[0]]
        
        if subtask:
            # Filter to specific subtask
            if subtask not in reference_model or subtask == 'metadata':
                raise ValueError(
                    f"Subtask '{subtask}' not found. "
                    f"Available subtasks: {self._list_subtasks()}"
                )
            
            subtask_data = reference_model[subtask]
            for query_id, query_data in subtask_data.items():
                if query_id in ['query_score', 'score', 'metadata']:
                    continue
                query_dict[query_id] = query_data['question']
        else:
            # Get all queries across all subtasks
            for subtask_name, subtask_data in reference_model.items():
                if subtask_name == 'metadata':
                    continue
                
                for query_id, query_data in subtask_data.items():
                    if query_id in ['query_score', 'score', 'metadata']:
                        continue
                    
                    # Prefix with subtask name to avoid collisions
                    full_id = f"{subtask_name}/{query_id}"
                    query_dict[full_id] = query_data['question']
        
        return query_dict
    
    def get_model_metadata(self, model_name: str) -> Dict:
        """Get metadata for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model metadata dict
            
        Raises:
            ValueError: If model not found
            
        Example:
            >>> metadata = quench.get_model_metadata('gpt4')
            >>> print(metadata)
            {'version': 'gpt-4-turbo', 'temperature': 0.7}
        """
        if self.benchmark_data is None or model_name not in self.benchmark_data:
            raise ValueError(f"Model '{model_name}' not found")
        
        return self.benchmark_data[model_name].get('metadata', {})
    
    def _list_subtasks(self) -> List[str]:
        """Get list of subtasks in the benchmark
        
        Returns:
            List of subtask names
        """
        if self.benchmark_data is None:
            return []
        
        model_names = self.list_models()
        if not model_names:
            return []
        
        reference_model = self.benchmark_data[model_names[0]]
        return [k for k in reference_model.keys() if k != 'metadata']
    
    # Private validation methods
    def _validate_response_json(self, response_json: Dict):
        """Validate response_json follows expected schema"""
        for model_name, model_data in response_json.items():
            if model_name == 'metadata':
                continue
            
            self._validate_model_data(model_data, model_name)
    
    def _validate_model_data(self, model_data: Dict, model_name: str):
        """Validate a single model's data structure"""
        if not isinstance(model_data, dict):
            raise ValueError(f"Model '{model_name}' data must be dict")

        # Check subtask level
        for subtask_name, subtask_data in model_data.items():
            if subtask_name in ['metadata', 'score']:
                continue

            if not isinstance(subtask_data, dict):
                raise ValueError(
                    f"Subtask '{subtask_name}' in model '{model_name}' must be dict"
                )
            
            # Check query level
            for query_id, query_data in subtask_data.items():
                if query_id in ['query_score', 'score', 'metadata']:
                    continue

                if not isinstance(query_data, dict):
                    continue

                # Only 'response' is strictly required
                if 'response' not in query_data:
                    raise ValueError(
                        f"Missing 'response' in {model_name}/"
                        f"{subtask_name}/{query_id}"
                    )
