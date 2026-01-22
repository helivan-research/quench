# Quench

Quench (**Qu**ery **E**fficient B**ench**marking) is an evaluation framework for generative models that enables efficient benchmarking of model responses against a standardized set of queries.

## Installation

```bash
pip install quench
```

Or install from source:

```bash
git clone https://github.com/yourusername/quench.git
cd quench
pip install -e .
```

## Quick Start

```python
from quench import login, Quench

# 1. Authenticate
login(api_key="your_api_key_here")

# 2. Load an existing benchmark
quench = Quench().load_benchmark('math_eval_v1')

# 3. Add a new model's responses
new_model_data = {
    'gpt-4o': {
        'algebra': {
            'q1': {
                'question': 'Solve x^2 = 4',
                'response': ['x = 2 or x = -2', 'x = ±2'],
                'reasoning': ['quadratic formula', 'square root'],
                'response_score': [1.0, 1.0],
                'score': 1.0,
                'metadata': {}
            },
            'query_score': [1.0],
            'score': 1.0,
            'metadata': {}
        },
        'metadata': {'version': 'gpt-4o-2024-05-13'}
    }
}

quench.add_model(new_model_data)

# 4. Evaluate all models in the benchmark
results = quench.predict(metrics=['accuracy', 'consistency'])

# 5. Or evaluate and add in one step
results = quench.predict(new_model_data, add_model=True, model_name='claude-opus')
```

## Core Concepts

### Authentication

All operations require authentication via an API key:

```python
from quench import login, logout

# Login
login(api_key="sk_abc123...")

# Logout when done
logout()
```

### Benchmarks

Benchmarks are collections of queries and model responses stored remotely. You can:

- **Create** a new benchmark from initial model responses
- **Load** an existing benchmark
- **Add models** to a benchmark
- **Remove models** from a benchmark

```python
quench = Quench()

# Create a new benchmark
quench.create_benchmark(
    response_json=initial_data,
    benchmark_name='my_benchmark',
    description='Custom evaluation benchmark'
)

# Load existing benchmark
quench.load_benchmark('my_benchmark')

# List models in benchmark
models = quench.list_models()
print(models)  # ['gpt4', 'claude-sonnet', 'gemini']

# Remove a model
quench.remove_model('gpt4')
```

### Response Schema

Model responses must follow this schema:

```python
{
    'model_name': {
        'subtask_name': {
            'query_id': {
                'question': str,              # The query question
                'response': [str, ...],       # List of model responses
                'reasoning': [str, ...],      # Reasoning for each response
                'response_score': [float, ...],  # Score for each response
                'score': float,               # Overall query score
                'metadata': dict              # Query-level metadata
            },
            'query_score': [float, ...],     # Scores for all queries
            'score': float,                   # Overall subtask score
            'metadata': dict                  # Subtask-level metadata
        },
        'metadata': dict                      # Model-level metadata
    },
    'metadata': dict                          # Benchmark-level metadata
}
```

### Querying Benchmark Data

```python
# Get all queries across all subtasks
all_queries = quench.get_query_dictionary()
# Returns: {'algebra/q1': 'Solve x^2 = 4', 'calculus/q1': '...', ...}

# Get queries for a specific subtask
algebra_queries = quench.get_query_dictionary(subtask='algebra')
# Returns: {'q1': 'Solve x^2 = 4', 'q2': 'Factor x^2 - 5x + 6', ...}

# Get model metadata
metadata = quench.get_model_metadata('gpt4')
# Returns: {'version': 'gpt-4-turbo', 'temperature': 0.7, ...}
```

### Evaluation

Evaluate model responses and compute metrics:

```python
# Evaluate loaded benchmark
results = quench.predict(
    metrics=['accuracy', 'consistency', 'coherence']
)

# Evaluate new model without adding to benchmark
results = quench.predict(
    response_json=new_model_data,
    metrics=['accuracy']
)

# Evaluate and add to benchmark
results = quench.predict(
    response_json=new_model_data,
    add_model=True,
    model_name='gpt-4o'
)
```

## API Reference

### Authentication Functions

- `login(api_key, base_url)` - Authenticate with API key
- `logout()` - Clear authentication state

### Quench Class

#### Methods

- `load_benchmark(benchmark_name)` - Load a benchmark from remote storage
- `create_benchmark(response_json, benchmark_name, **kwargs)` - Create a new benchmark
- `add_model(response_json, model_name)` - Add a model to the benchmark
- `remove_model(model_name)` - Remove a model from the benchmark
- `predict(response_json, metrics, add_model, model_name, **kwargs)` - Evaluate models
- `list_models()` - Get list of models in benchmark
- `get_query_dictionary(subtask)` - Get mapping of query IDs to questions
- `get_model_metadata(model_name)` - Get metadata for a specific model

### Exceptions

- `AuthenticationError` - Raised when authentication is required but not provided
- `BenchmarkNotFoundError` - Raised when a requested benchmark doesn't exist

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

### Code Formatting

```bash
black quench/
flake8 quench/
mypy quench/
```

## License

MIT License - see LICENSE file for details

## Support

For issues, questions, or contributions, please visit:
https://github.com/helivan-research/quench/issues
