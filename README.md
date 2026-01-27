# Quench

Python SDK for query-efficient LLM benchmark score prediction.

Quench uses behavioral similarity to cached models to predict benchmark scores from a fraction of the queries, dramatically reducing evaluation costs.

**API:** https://quench.helivan.io/api
**Web App:** https://quench.helivan.io
**Documentation:** https://github.com/helivan-research/quench

## Installation

```bash
pip install quench
```

## Quick Start

```python
from quench import Quench, login

# Load a public benchmark (no auth required)
q = Quench().load_benchmark('helm_gsm8k')

# Predict scores for a new model using only a subset of queries
results = q.predict({
    'my_model': {
        'gsm': {
            'q1': {'question': 'What is 2+2?', 'response': ['4']},
            'q2': {'question': 'What is 3*5?', 'response': ['15']}
        }
    }
})

print(f"Predicted score: {results['predicted_scores']['my_model']:.3f}")
print(f"Similar models: {results['similar_models']}")
```

## Authentication

Required for private benchmarks and creating/modifying data:

```python
from quench import login, logout

# Login with API key (uses https://quench.helivan.io/api by default)
login(api_key="qk_abc123...")

# Or specify a custom API URL
login(api_key="qk_abc123...", base_url="https://your-server.com/api")

# Logout when done
logout()
```

Get your API key from **Settings → API Keys** in the [web app](https://quench.helivan.io).

## Core API

### Loading Benchmarks

```python
from quench import Quench

# Load public benchmark (no auth)
q = Quench().load_benchmark('helm_gsm8k')

# Load private benchmark (requires auth)
login(api_key="qk_...")
q = Quench().load_benchmark('my_private_benchmark')
```

### Predicting Scores

Predict the overall benchmark score for a model that has only answered a subset of queries:

```python
# Partial responses - only answered some queries
partial_responses = {
    'my_model': {
        'algebra': {
            'q1': {'question': '2+2?', 'response': ['4']},
            'q2': {'question': '3*3?', 'response': ['9']}
        }
        # Note: other subtasks not included - this is partial evaluation
    }
}

results = q.predict(partial_responses)

print(results['predicted_scores'])       # {'my_model': 0.87}
print(results['confidence_interval'])    # {'my_model': [0.82, 0.92]}
print(results['similar_models'])         # Most similar cached models
print(results['overlap_info'])           # Query coverage statistics
```

### Optimal Query Selection

Find the most informative queries to maximize prediction accuracy with minimal evaluation:

```python
# Get top 15 most informative queries
optimal = q.get_optimal_queries(budget=15)

print(optimal['queries'][:3])
# [
#   {'subtask': 'algebra', 'query_id': 'q42', 'importance': 0.92, ...},
#   {'subtask': 'geometry', 'query_id': 'q17', 'importance': 0.88, ...},
#   ...
# ]

print(f"Estimated error with 15 queries: {optimal['estimated_error']:.3f}")
```

### Budget Estimation

Estimate how many queries you need for a target prediction accuracy:

```python
# How many queries for 5% error?
budget_info = q.estimate_query_budget(target_error=0.05)

print(f"Need {budget_info['estimated_queries']} queries for 5% error")
print(f"Confidence interval: {budget_info['confidence_interval']}")

# View the full error curve
for point in budget_info['error_curve']:
    print(f"  {point['k']} queries -> {point['mae']:.3f} MAE")
```

### Managing Benchmarks

```python
from quench import Quench, login

login(api_key="qk_...")

# Create a new benchmark
benchmark_data = {
    'gpt4': {
        'math': {
            'q1': {
                'question': '2+2=?',
                'response': ['4'],
                'score': 1.0
            }
        },
        'score': 0.95
    },
    'claude': {
        'math': {
            'q1': {
                'question': '2+2=?',
                'response': ['4'],
                'score': 1.0
            }
        },
        'score': 0.92
    }
}

q = Quench().create_benchmark(benchmark_data, 'my_benchmark')

# Add a new model
q.add_model({
    'gemini': {
        'math': {
            'q1': {
                'question': '2+2=?',
                'response': ['4'],
                'score': 1.0
            }
        },
        'score': 0.90
    }
})

# Remove a model
q.remove_model('gemini')

# List all models
print(q.list_models())  # ['gpt4', 'claude']
```

### Exploring Benchmark Data

```python
# List models in benchmark
models = q.list_models()
print(models)  # ['gpt4', 'claude', 'gemini']

# Get all queries
queries = q.get_query_dictionary()
print(queries)
# {'math/q1': '2+2=?', 'math/q2': '3*3=?', 'logic/q1': 'If A then B...'}

# Get queries for specific subtask
math_queries = q.get_query_dictionary(subtask='math')
print(math_queries)  # {'q1': '2+2=?', 'q2': '3*3=?'}

# Get model metadata
metadata = q.get_model_metadata('gpt4')
print(metadata)  # {'version': 'gpt-4-turbo', 'temperature': 0.7}
```

## SDK Reference

### Authentication Functions

| Function | Description |
|----------|-------------|
| `login(api_key, base_url)` | Authenticate with API key |
| `logout()` | Clear authentication state |

### Quench Class Methods

| Method | Auth Required | Description |
|--------|---------------|-------------|
| `load_benchmark(name)` | No* | Load benchmark from server |
| `create_benchmark(data, name)` | Yes | Create new benchmark |
| `add_model(data, model_name)` | Yes | Add model to benchmark |
| `remove_model(name)` | Yes | Remove model from benchmark |
| `predict(data)` | No* | Predict scores for new model |
| `get_optimal_queries(budget)` | No* | Get most informative queries |
| `estimate_query_budget(target_error)` | No* | Estimate queries needed |
| `list_models()` | No | List models in loaded benchmark |
| `get_query_dictionary(subtask)` | No | Get query ID to question mapping |
| `get_model_metadata(model_name)` | No | Get metadata for a model |

\* Auth required for private benchmarks

## REST API Endpoints

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/login` | POST | Login with email/password or API key |
| `/auth/google` | POST | Login with Google OAuth |
| `/auth/register` | POST | Register new account |
| `/auth/me` | GET | Get current user info |

### Benchmarks

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/benchmarks` | GET | List all accessible benchmarks |
| `/benchmarks` | POST | Create new benchmark |
| `/benchmarks/<name>` | GET | Get benchmark details |
| `/benchmarks/<name>` | DELETE | Delete benchmark |
| `/benchmarks/<name>/summary` | GET | Get benchmark summary (lightweight) |
| `/benchmarks/<name>/visualize` | GET | Get MDS visualization HTML |
| `/benchmarks/<name>/models/<model>` | PATCH | Add/update model |
| `/benchmarks/<name>/models/<model>` | DELETE | Remove model |

### Prediction & Analysis

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/evaluate/predict` | POST | Predict scores for new model |
| `/benchmarks/<name>/predict` | POST | Predict using stored benchmark |
| `/benchmarks/<name>/optimal-queries` | GET | Get optimal query selection |
| `/benchmarks/<name>/query-budget` | GET | Estimate queries for target error |

### API Keys

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/api-keys` | GET | List user's API keys |
| `/auth/api-keys` | POST | Create new API key |
| `/auth/api-keys/<id>` | DELETE | Delete API key |

## Data Format

### Benchmark Data Schema

```python
{
    'model_name': {
        'subtask_name': {
            'query_id': {
                'question': str,           # The question/prompt
                'response': [str, ...],    # Model response(s)
                'score': float,            # Optional: query score (0-1)
                'reasoning': [str, ...],   # Optional: reasoning steps
                'metadata': {}             # Optional: additional data
            },
            'score': float,                # Optional: subtask score
            'metadata': {}
        },
        'score': float,                    # Optional: model's overall score
        'metadata': {}                     # Optional: model metadata
    },
    'metadata': {}                         # Optional: benchmark metadata
}
```

### Prediction Response

```python
{
    'predicted_scores': {'model_name': 0.87},
    'confidence_interval': {'model_name': [0.82, 0.92]},
    'similar_models': [
        {'name': 'gpt4', 'score': 0.92, 'similarity': 0.95},
        {'name': 'claude', 'score': 0.89, 'similarity': 0.91}
    ],
    'overlap_info': {
        'queries_evaluated': 15,
        'total_queries': 500,
        'coverage': 0.03
    },
    'visualization_html': '...'  # Interactive Plotly chart
}
```

### Optimal Queries Response

```python
{
    'benchmark_name': 'helm_math',
    'budget': 15,
    'queries': [
        {
            'subtask': 'algebra',
            'query_id': 'q42',
            'question': 'Solve for x: 2x + 5 = 13',
            'importance': 0.92,
            'breakdown': {
                'correlation': 0.95,
                'variance': 0.88,
                'diversity': 1.0
            },
            'rank': 1
        },
        ...
    ],
    'total_queries': 500,
    'estimated_error': 0.08,
    'model_count': 12
}
```

## Examples

### Efficient Model Evaluation

```python
from quench import Quench

# Load benchmark
q = Quench().load_benchmark('helm_math')

# Get optimal queries for your budget
optimal = q.get_optimal_queries(budget=20)

# Evaluate your model on just these queries
queries_to_run = [
    (query['subtask'], query['query_id'], query['question'])
    for query in optimal['queries']
]

# Run your model on the selected queries...
my_responses = run_model_on_queries(queries_to_run)  # Your evaluation code

# Predict full benchmark score
results = q.predict({'my_model': my_responses})
print(f"Predicted score: {results['predicted_scores']['my_model']:.3f}")
print(f"Saved {100 * (1 - 20/optimal['total_queries']):.0f}% of evaluation cost!")
```

### Comparing Models

```python
from quench import Quench, login

login(api_key="qk_...")
q = Quench().load_benchmark('my_benchmark')

# Get scores for all cached models
for model in q.list_models():
    metadata = q.get_model_metadata(model)
    print(f"{model}: {metadata.get('score', 'N/A')}")
```

## Error Handling

```python
from quench import Quench, login, AuthenticationError, BenchmarkNotFoundError

try:
    q = Quench().load_benchmark('private_benchmark')
except AuthenticationError:
    print("Need to login first")
    login(api_key="qk_...")
    q = Quench().load_benchmark('private_benchmark')
except BenchmarkNotFoundError:
    print("Benchmark doesn't exist")
```

## Getting an API Key

1. Sign up at [quench.helivan.io](https://quench.helivan.io)
2. Go to **Settings** (user menu in top-right)
3. Navigate to **API Keys**
4. Click **Create API Key**
5. Copy the key (it starts with `qk_`)

## Feedback & Support

- **Issues:** [github.com/helivan-research/quench/issues](https://github.com/helivan-research/quench/issues)
- **Email:** info@helivan.io

## License

MIT

---

Built by [Helivan Research](https://helivan.io)
