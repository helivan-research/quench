"""
Quench - Evaluation framework for generative models

Quench provides a simple API for evaluating generative model responses
against benchmarks stored on remote servers.

Basic usage:
    >>> from quench import login, Quench
    >>> 
    >>> # Authenticate
    >>> login(api_key="your_api_key")
    >>> 
    >>> # Load a benchmark
    >>> quench = Quench().load_benchmark('my_benchmark')
    >>> 
    >>> # Add a new model
    >>> quench.add_model(model_responses)
    >>> 
    >>> # Evaluate
    >>> results = quench.predict(metrics=['accuracy', 'consistency'])
"""

from .quench import (
    Quench,
    login,
    logout,
    AuthenticationError,
    BenchmarkNotFoundError
)

__version__ = "0.1.0"

__all__ = [
    'Quench',
    'login',
    'logout',
    'AuthenticationError',
    'BenchmarkNotFoundError'
]
