# Quench

Predict benchmark scores from a fraction of the queries.

Quench uses behavioral similarity to predict how a model will perform on an entire benchmark after evaluating only a small subset of queries. This dramatically reduces evaluation costs while maintaining accurate predictions.

**Web App:** https://quench.helivan.io
**API:** https://quench.helivan.io/api

## What is a "Model"?

In Quench, a "model" is any configuration that produces responses to queries. This includes:

- **Different LLMs** (GPT-4, Claude, Llama, etc.)
- **Different prompts** (system prompts, few-shot examples, instructions)
- **Different temperatures** or sampling parameters
- **Different retrieval configurations** (RAG pipelines, vector stores)
- **Different fine-tunes** of the same base model
- **Any combination** of the above

If you want to compare how two configurations perform on a benchmark, each configuration is a "model" in Quench.

## Installation

```bash
pip install quench
```

## Quickstart

### 1. Create an Account

Sign up at [quench.helivan.io](https://quench.helivan.io)

### 2. Create an API Key

1. Go to **Settings** (user menu in top-right)
2. Navigate to **API Keys**
3. Click **Create API Key**
4. Copy the key (starts with `qk_`)

### 3. (Optional) Create a Benchmark

If you have responses from multiple models to a shared set of queries, you can create your own benchmark:

```python
from quench import Quench, login

login(api_key="qk_your_key_here")

# Responses from models you've already evaluated
benchmark_data = {
    'gpt4_temp0': {
        'reasoning': {
            'q1': {'question': 'What is 15% of 80?', 'response': ['12'], 'score': 1.0},
            'q2': {'question': 'If x + 5 = 12, what is x?', 'response': ['7'], 'score': 1.0},
            # ... more queries
        },
        'score': 0.92  # Overall score for this model
    },
    'gpt4_temp07': {
        'reasoning': {
            'q1': {'question': 'What is 15% of 80?', 'response': ['12'], 'score': 1.0},
            'q2': {'question': 'If x + 5 = 12, what is x?', 'response': ['x = 7'], 'score': 1.0},
            # ... same queries, different responses
        },
        'score': 0.88
    },
    'claude_with_cot': {
        'reasoning': {
            'q1': {'question': 'What is 15% of 80?', 'response': ['Let me think step by step... 12'], 'score': 1.0},
            'q2': {'question': 'If x + 5 = 12, what is x?', 'response': ['x = 7'], 'score': 1.0},
        },
        'score': 0.95
    }
}

q = Quench().create_benchmark(benchmark_data, 'my-math-benchmark')
```

Or use one of the public benchmarks (HELM, etc.) already available.

### 4. Predict Scores for a New Model

Now evaluate a new configuration on just a subset of queries and predict its full benchmark score:

```python
from quench import Quench, login

login(api_key="qk_your_key_here")

# Load your benchmark (or a public one)
q = Quench().load_benchmark('my-math-benchmark')

# Get the most informative queries to evaluate
optimal = q.get_optimal_queries(budget=20)
print(f"Evaluate these {len(optimal['queries'])} queries:")
for query in optimal['queries']:
    print(f"  {query['subtask']}/{query['query_id']}: {query['question']}")

# After running your new model on those queries...
new_model_responses = {
    'llama_finetuned': {
        'reasoning': {
            'q1': {'question': 'What is 15% of 80?', 'response': ['12']},
            'q2': {'question': 'If x + 5 = 12, what is x?', 'response': ['7']},
            # ... responses to the optimal queries
        }
    }
}

# Predict the full benchmark score
results = q.predict(new_model_responses)

print(f"Predicted score: {results['predicted_scores']['llama_finetuned']:.3f}")
print(f"Confidence interval: {results['confidence_interval']['llama_finetuned']}")
print(f"Most similar to: {results['similar_models'][0]['name']}")
```

## Core Concepts

### Benchmarks

A benchmark is a collection of queries organized into subtasks, with responses from multiple models. Quench learns the behavioral patterns across these models to predict scores for new models.

### Optimal Query Selection

Not all queries are equally informative. Quench identifies which queries best differentiate between models, so you can evaluate the most valuable ones first.

```python
# Get the 15 most informative queries
optimal = q.get_optimal_queries(budget=15)

# Or: how many queries do I need for 5% error?
budget = q.estimate_query_budget(target_error=0.05)
print(f"Need ~{budget['estimated_queries']} queries for 5% prediction error")
```

### Prediction

Given partial responses (a model's answers to a subset of queries), Quench:
1. Computes behavioral similarity to cached models
2. Uses this similarity to predict performance on unseen queries
3. Returns a predicted overall score with confidence intervals

## API Reference

### Authentication

```python
from quench import login, logout

login(api_key="qk_...")  # Required for private benchmarks and creating data
logout()                  # Clear authentication
```

### Quench Class

```python
from quench import Quench

q = Quench()

# Load a benchmark
q.load_benchmark('benchmark_name')

# Create a new benchmark
q.create_benchmark(data, 'new_benchmark_name')

# Add a model to the loaded benchmark
q.add_model(model_data)

# Predict scores for partial responses
results = q.predict(partial_responses)

# Get optimal queries for a budget
optimal = q.get_optimal_queries(budget=20)

# Estimate queries needed for target error
budget = q.estimate_query_budget(target_error=0.05)

# Explore the benchmark
models = q.list_models()
queries = q.get_query_dictionary()
metadata = q.get_model_metadata('model_name')
```

## Embedding Providers

When creating a benchmark, Quench computes embeddings to measure behavioral similarity between models. The platform provides API keys — you just choose a provider and model.

| Provider | Model | Dimensions | Max Tokens | Cost / 1M tokens |
|----------|-------|------------|------------|-------------------|
| **google** (default) | `gemini-embedding-001` | 3072 | 2048 | $0.15 |
| openai | `text-embedding-3-small` | 1536 | 8191 | $0.02 |
| openai | `text-embedding-3-large` | 3072 | 8191 | $0.13 |
| openai | `text-embedding-ada-002` | 1536 | 8191 | $0.10 |

When creating a benchmark through the web app, select the provider and model in the creation form. The default is Google `gemini-embedding-001`.

## Data Format

### Benchmark/Model Data

```python
{
    'model_name': {
        'subtask_name': {
            'query_id': {
                'question': str,        # The prompt/question
                'response': [str],      # Model's response(s)
                'score': float,         # Optional: 0-1 score for this query
            }
        },
        'score': float  # Optional: overall score for this model
    }
}
```

### Prediction Response

```python
{
    'predicted_scores': {'model_name': 0.87},
    'confidence_interval': {'model_name': [0.82, 0.92]},
    'similar_models': [
        {'name': 'gpt4', 'similarity': 0.95, 'score': 0.92},
        ...
    ],
    'overlap_info': {
        'queries_evaluated': 20,
        'total_queries': 500,
        'coverage': 0.04
    }
}
```

## Example: Comparing Prompt Variants

```python
from quench import Quench, login

login(api_key="qk_...")

# You have a benchmark with various prompting strategies
q = Quench().load_benchmark('prompt-comparison')

# Test a new prompt variant on just 25 queries
optimal = q.get_optimal_queries(budget=25)

# Run your new prompt on those queries and collect responses...
new_prompt_responses = {
    'cot_v2_with_examples': {
        'math': {
            'q42': {'question': '...', 'response': ['...']},
            # ... 24 more queries
        }
    }
}

results = q.predict(new_prompt_responses)
print(f"Predicted score for new prompt: {results['predicted_scores']['cot_v2_with_examples']:.1%}")
```

## Feedback & Support

- **Issues:** [github.com/helivan-research/quench/issues](https://github.com/helivan-research/quench/issues)
- **Email:** info@helivan.io

## License

MIT

---

Built by [Helivan Research](https://helivan.io)
