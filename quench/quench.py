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
    'user_id': None,
    'authenticated': False,
    'base_url': None
}


def login(api_key: str, base_url: str = "https://api.quench.io") -> bool:
    """Authenticate user with API key
    
    Args:
        api_key: User's API key
        base_url: Base URL for Quench API (default: https://api.quench.io)
        
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
        self._base_url = _AUTH_STATE.get('base_url', 'https://api.quench.io')
    
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
                headers['Authorization'] = f"Bearer {_AUTH_STATE['api_key']}"
            
            response = requests.get(
                f"{self._base_url}/benchmarks/{benchmark_name}",
                headers=headers
            )
            response.raise_for_status()
            
            data = response.json()
            self.benchmark_name = benchmark_name
            self.benchmark_data = data.get('benchmark_data', {})
            self.benchmark_metadata = data.get('metadata', {})
            
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
            headers={'Authorization': f"Bearer {_AUTH_STATE['api_key']}"},
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
            headers={'Authorization': f"Bearer {_AUTH_STATE['api_key']}"},
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
            headers={'Authorization': f"Bearer {_AUTH_STATE['api_key']}"}
        )
        response.raise_for_status()
        
        print(f"✓ Removed model '{model_name}' from benchmark '{self.benchmark_name}'")
        
        return self
    
    def predict(
        self, 
        response_json: Optional[Dict] = None, 
        metrics: Optional[List[str]] = None,
        add_model: bool = False,
        model_name: Optional[str] = None,
        **kwargs
    ):
        """Evaluate model responses and compute scores
        
        Public benchmarks can be evaluated without authentication.
        Adding models to benchmarks (add_model=True) requires authentication.
        
        Args:
            response_json: Optional new responses to evaluate (otherwise uses loaded benchmark)
            metrics: List of metrics to compute (e.g., ['accuracy', 'consistency', 'coherence'])
            add_model: If True, add the evaluated model to the benchmark (default: False, requires auth)
            model_name: Required if add_model=True and response_json is subtask-level data
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dict with evaluation results containing model scores, subtask scores, etc.
            
        Raises:
            AuthenticationError: If add_model=True but not authenticated
            ValueError: If add_model=True but model already exists, or no data to evaluate
            
        Example:
            >>> # Evaluate public benchmark (no auth required)
            >>> quench = Quench().load_benchmark('public_math_eval')
            >>> results = quench.predict(metrics=['accuracy'])
            ✓ Evaluation complete
            
            >>> # Evaluate and add to benchmark (auth required)
            >>> login(api_key="sk_abc123...")
            >>> results = quench.predict(new_responses, add_model=True, model_name='gpt-4o')
            ✓ Evaluation complete
            ✓ Added model 'gpt-4o' to benchmark 'my_benchmark'
        """
        # Check authentication requirement
        if add_model and not _AUTH_STATE['authenticated']:
            raise AuthenticationError(
                "Adding models to benchmarks requires authentication. "
                "Please call login(api_key) first."
            )
        
        # Use provided data or fall back to loaded benchmark
        data = response_json if response_json else self.benchmark_data
        
        if data is None:
            raise ValueError("No data to evaluate. Provide response_json or load a benchmark.")
        
        # If adding model, validate it doesn't already exist
        if add_model and response_json:
            # Determine the model name from response_json
            if model_name and model_name not in response_json:
                eval_model_name = model_name
            else:
                model_keys = [k for k in response_json.keys() if k != 'metadata']
                if len(model_keys) != 1:
                    raise ValueError(
                        "When add_model=True, response_json must contain exactly one model "
                        "(or use model_name parameter)"
                    )
                eval_model_name = model_keys[0]
            
            if self.benchmark_data and eval_model_name in self.benchmark_data:
                raise ValueError(
                    f"Model '{eval_model_name}' already exists in benchmark. "
                    f"Use add_model=False to evaluate without adding, or remove_model() first."
                )
        
        # Prepare headers - include auth token if available
        headers = {}
        if _AUTH_STATE['authenticated']:
            headers['Authorization'] = f"Bearer {_AUTH_STATE['api_key']}"
        
        # Send evaluation request to remote API
        payload = {
            'benchmark_data': data,
            'metrics': metrics or ['accuracy', 'consistency'],
            'parameters': kwargs
        }
        
        response = requests.post(
            f"{self._base_url}/evaluate",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        results = response.json()
        
        # If add_model flag is True, add the model to benchmark
        if add_model and response_json:
            self.add_model(response_json, model_name=model_name)
        
        print(f"✓ Evaluation complete")
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
            if subtask_name == 'metadata':
                continue
            
            if not isinstance(subtask_data, dict):
                raise ValueError(
                    f"Subtask '{subtask_name}' in model '{model_name}' must be dict"
                )
            
            # Check query level
            for query_id, query_data in subtask_data.items():
                if query_id in ['query_score', 'score', 'metadata']:
                    continue
                
                required_fields = ['question', 'response', 'reasoning', 
                                 'response_score', 'score']
                for field in required_fields:
                    if field not in query_data:
                        raise ValueError(
                            f"Missing '{field}' in {model_name}/"
                            f"{subtask_name}/{query_id}"
                        )
                
                # Validate list fields have same length
                r = len(query_data['response'])
                if (len(query_data['reasoning']) != r or 
                    len(query_data['response_score']) != r):
                    raise ValueError(
                        f"Mismatched array lengths in {model_name}/"
                        f"{subtask_name}/{query_id}"
                    )
