# Quench

Python SDK for query-efficient LLM benchmark score prediction.

## Installation

```bash
pip install quench
```

## Usage

```python
from quench import Quench

# Load a public benchmark
q = Quench().load_benchmark('helm_gsm8k')

# Predict scores for a new model
results = q.predict({
    'my_model': {
        'gsm': {
            'q1': {'question': 'What is 2+2?', 'response': ['4']}
        }
    }
})

print(results['predicted_scores'])
print(results['similar_models'])
```

## Authentication

Required for private benchmarks and creating/modifying data:

```python
from quench import login
login(api_key="your_api_key")
```

## API

| Method | Description |
|--------|-------------|
| `load_benchmark(name)` | Load benchmark from server |
| `predict(data)` | Predict scores for new model |
| `list_models()` | List models in benchmark |
| `add_model(data)` | Add model to benchmark (auth required) |
| `remove_model(name)` | Remove model (auth required) |
| `create_benchmark(data, name)` | Create new benchmark (auth required) |

## Public Benchmarks

| Name | Description |
|------|-------------|
| `helm_gsm8k` | Grade School Math |
| `helm_math` | MATH Competition Problems |
| `helm_mmlu` | MMLU Knowledge Questions |
| `helm_medqa` | Medical QA |
| `helm_legalbench` | Legal Reasoning |
| `helm_narrativeqa` | Story Comprehension |
| `helm_naturalqa` | Natural Questions |
| `helm_openbookqa` | OpenBookQA |
| `helm_wmt14` | WMT14 Translation |

## Data Format

```python
{
    'model_name': {
        'subtask': {
            'query_id': {
                'question': str,
                'response': [str, ...],
                'score': float  # optional
            }
        }
    }
}
```
