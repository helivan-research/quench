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

### 1. Create an Account & API Key

1. Sign up at [quench.helivan.io](https://quench.helivan.io)
2. Go to the user menu (top-right) and create an **API Key**
3. Copy the key (starts with `qk_`)

### 2. Authenticate

```python
import quench

# Option A: set in code
quench.api_key = "qk_your_key_here"

# Option B: set via environment variable
# export QUENCH_API_KEY="qk_your_key_here"
```

### 3. Create a Benchmark (Optional)

If you have responses from multiple models to a shared set of queries:

```python
benchmark_data = {
    "gpt4_temp0": {
        "reasoning": {
            "q1": {"question": "What is 15% of 80?", "response": ["12"], "score": 1.0},
            "q2": {"question": "If x + 5 = 12, what is x?", "response": ["7"], "score": 1.0},
        },
        "score": 0.92,
    },
    "claude_with_cot": {
        "reasoning": {
            "q1": {"question": "What is 15% of 80?", "response": ["Let me think... 12"], "score": 1.0},
            "q2": {"question": "If x + 5 = 12, what is x?", "response": ["x = 7"], "score": 1.0},
        },
        "score": 0.95,
    },
}

benchmark = quench.Benchmark.create("my-math-benchmark", benchmark_data)
```

Or use one of the public benchmarks (HELM, etc.) already available.

### 4. Predict Scores for a New Model

Evaluate a new configuration on just a subset of queries and predict its full score:

```python
# Load your benchmark (or a public one)
benchmark = quench.Benchmark.load("my-math-benchmark")

# Get the most informative queries to evaluate
optimal = benchmark.get_optimal_queries(budget=20)
print(f"Evaluate these {len(optimal['queries'])} queries:")
for q in optimal["queries"]:
    print(f"  {q['subtask']}/{q['query_id']}: {q['question']}")

# After running your new model on those queries...
new_model_responses = {
    "llama_finetuned": {
        "reasoning": {
            "q1": {"question": "What is 15% of 80?", "response": ["12"]},
            "q2": {"question": "If x + 5 = 12, what is x?", "response": ["7"]},
        }
    }
}

results = benchmark.predict(new_model_responses)

print(f"Predicted score: {results['predicted_scores']['llama_finetuned']:.3f}")
print(f"Confidence interval: {results['confidence_interval']['llama_finetuned']}")
print(f"Most similar to: {results['similar_models'][0]['name']}")
```

## Core Concepts

### Benchmarks

A benchmark is a collection of queries organized into subtasks, with responses from multiple models. Quench learns the behavioral patterns across these models to predict scores for new ones.

### Optimal Query Selection

Not all queries are equally informative. Quench identifies which queries best differentiate between models, so you can evaluate the most valuable ones first.

```python
# Get the 15 most informative queries
optimal = benchmark.get_optimal_queries(budget=15)

# Or: how many queries do I need for 5% error?
budget = benchmark.estimate_query_budget(target_error=0.05)
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
import quench

# Simple (module-level)
quench.api_key = "qk_..."

# Advanced (thread-safe, multiple keys)
from quench import QuenchClient
client = QuenchClient(api_key="qk_...")
benchmark = client.benchmarks.load("my-benchmark")
```

### Benchmark Operations

```python
# Load a benchmark
benchmark = quench.Benchmark.load("benchmark_name")

# Create a new benchmark
benchmark = quench.Benchmark.create("new_name", data)

# List public benchmarks
benchmarks = quench.Benchmark.list(category="math")

# List your benchmarks
mine = quench.Benchmark.list_mine()

# Delete a benchmark
benchmark.delete()
```

### Model Management

```python
# Add a model
benchmark.add_model(model_data)
benchmark.add_model(subtask_data, model_name="my_model")

# Remove a model
benchmark.remove_model("model_name")
```

### Prediction & Query Selection

```python
# Predict scores for partial responses
results = benchmark.predict(partial_responses)

# Get optimal queries for a budget
optimal = benchmark.get_optimal_queries(budget=20)

# Estimate queries needed for target error
budget = benchmark.estimate_query_budget(target_error=0.05)
```

### Inspection

```python
benchmark.models                              # List of model names
benchmark.summary()                           # Lightweight scores
benchmark.visualize()                         # Interactive MDS visualization HTML
benchmark.get_query_dictionary()              # {subtask/query_id: question}
benchmark.get_query_dictionary(subtask="math")  # {query_id: question}
benchmark.get_model_metadata("gpt4")          # Model metadata dict
```

### Utilities

```python
quench.embedding_providers()    # Available embedding models
quench.benchmark_categories()   # Available categories
```

## Embedding Providers

When creating a benchmark, Quench computes embeddings to measure behavioral similarity between models. The platform provides API keys — you just choose a provider and model.

| Provider | Model | Dimensions | Max Tokens |
|----------|-------|------------|------------|
| **google** (default) | `gemini-embedding-001` | 3072 | 2048 |
| openai | `text-embedding-3-small` | 1536 | 8191 |
| openai | `text-embedding-3-large` | 3072 | 8191 |
| openai | `text-embedding-ada-002` | 1536 | 8191 |

## Data Format

### Benchmark/Model Data

```python
{
    "model_name": {
        "subtask_name": {
            "query_id": {
                "question": str,        # The prompt/question
                "response": [str],      # Model's response(s)
                "score": float,         # Optional: 0-1 score for this query
            }
        },
        "score": float  # Optional: overall score for this model
    }
}
```

### Prediction Response

```python
{
    "predicted_scores": {"model_name": 0.87},
    "confidence_interval": {"model_name": [0.82, 0.92]},
    "similar_models": [
        {"name": "gpt4", "similarity": 0.95, "score": 0.92},
    ],
    "overlap_info": {
        "queries_evaluated": 20,
        "total_queries": 500,
        "coverage": 0.04
    }
}
```

## Example: Comparing Prompt Variants

```python
import quench

quench.api_key = "qk_..."

# You have a benchmark with various prompting strategies
benchmark = quench.Benchmark.load("prompt-comparison")

# Test a new prompt variant on just 25 queries
optimal = benchmark.get_optimal_queries(budget=25)

# Run your new prompt on those queries and collect responses...
new_prompt_responses = {
    "cot_v2_with_examples": {
        "math": {
            "q42": {"question": "...", "response": ["..."]},
            # ... 24 more queries
        }
    }
}

results = benchmark.predict(new_prompt_responses)
print(f"Predicted score: {results['predicted_scores']['cot_v2_with_examples']:.1%}")
```

## Feedback & Support

- **Issues:** [github.com/helivan-research/quench/issues](https://github.com/helivan-research/quench/issues)
- **Email:** info@helivan.io

## License

MIT

---

Built by [Helivan Research](https://helivan.io)
