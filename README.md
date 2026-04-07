# Quench

Predict benchmark scores from a fraction of the queries.

Quench uses behavioral similarity to predict how a model will perform on an entire benchmark after evaluating only a small subset of queries. This dramatically reduces evaluation costs while maintaining accurate predictions.

**Web App:** https://quench.helivan.io
**API:** https://quench.helivan.io/api

## What is a "Model"?

In Quench, a "model" is any configuration that produces responses to queries:

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

### 3. Load a Benchmark

```python
benchmark = quench.Benchmark.load("jailbreak-safety-v1")

print(f"Models: {len(benchmark.models)}")
print(f"Scores: {benchmark.scores}")
```

### 4. Get Optimal Queries

Not all queries are equally informative. Quench identifies which queries best differentiate between models:

```python
optimal = benchmark.get_optimal_queries(budget=20)
print(f"Evaluate these {len(optimal['queries'])} queries:")
for q in optimal["queries"]:
    print(f"  {q['subtask']}/{q['query_id']}")
```

### 5. Stage a Prediction

Before running your model, stage the prediction to preload cached model embeddings. This makes the actual prediction instant:

```python
query_ids = [(q["subtask"], q["query_id"]) for q in optimal["queries"]]
session = benchmark.stage(query_ids)
```

### 6. Predict

After running your model on those queries:

```python
results = benchmark.predict(
    {
        "my-model": {
            "subtask_name": {
                "query_id": {
                    "question": "...",
                    "response": ["model's response here"],
                }
            }
        }
    },
    session=session,  # Uses pre-staged embeddings — prediction completes in ~2s
)

print(f"Predicted score: {results['predicted_scores']['my-model']:.3f}")
print(f"Confidence interval: {results['confidence_interval']['my-model']}")
print(f"Most similar to: {results['similar_models']['my-model'][0]['model']}")
```

## Core Concepts

### Benchmarks

A benchmark is a collection of queries organized into subtasks, with responses from multiple models. Quench learns the behavioral patterns across these models to predict scores for new ones.

### Optimal Query Selection

Quench identifies which queries best differentiate between models using leave-one-out cross-validation, so you can evaluate the most valuable ones first.

```python
# Get the 15 most informative queries
optimal = benchmark.get_optimal_queries(budget=15)

# Or: how many queries do I need for 5% error?
budget = benchmark.estimate_query_budget(target_error=0.05)
print(f"Need ~{budget['estimated_queries']} queries for 5% prediction error")
```

### Staged Prediction

For large benchmarks (80+ models), prediction requires comparing your model to every cached model. Staging preloads the necessary embeddings so the prediction itself is instant:

```python
session = benchmark.stage(query_ids)         # ~60-90s (preloads embeddings)
results = benchmark.predict(data, session=session)  # ~2s (uses preloaded data)
```

Without staging, prediction still works but may take longer.

### Prediction

Given partial responses (a model's answers to a subset of queries), Quench:
1. Embeds the new model's responses
2. Computes behavioral similarity to cached models via embedding distances
3. Applies classical MDS to get low-dimensional model representations
4. Trains Ridge regression on MDS coordinates to predict the overall benchmark score

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
benchmark = quench.Benchmark.create("new_name", data,
    embedding_model="openai/text-embedding-3-small",
    category="safety")

# Wait for processing (embeddings computed asynchronously)
benchmark.wait_until_ready(
    timeout=600,
    on_progress=lambda progress, pct: print(f"Processing: {progress} ({pct}%)")
)

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
# Get optimal queries for a budget
optimal = benchmark.get_optimal_queries(budget=20)

# Stage for fast prediction
query_ids = [(q["subtask"], q["query_id"]) for q in optimal["queries"]]
session = benchmark.stage(query_ids)

# Predict scores
results = benchmark.predict(partial_responses, session=session)

# Estimate queries needed for target error
budget = benchmark.estimate_query_budget(target_error=0.05)
```

### Inspection

```python
benchmark.models                                # List of model names
benchmark.scores                                # {model: score} dict
benchmark.status                                # "processing", "ready", "failed"
benchmark.progress                              # Processing progress (when processing)
benchmark.summary()                             # Lightweight model summaries
benchmark.visualize()                           # Interactive MDS visualization HTML
benchmark.get_query_dictionary()                # {subtask/query_id: question}
benchmark.get_query_dictionary(subtask="math")  # {query_id: question}
benchmark.get_model_metadata("gpt4")            # Model metadata dict
```

### Utilities

```python
quench.embedding_providers()    # Available embedding models
quench.benchmark_categories()   # Available categories
```

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
    "similar_models": {
        "model_name": [
            {"model": "gpt4", "similarity": 0.95, "distance": 1.2},
        ]
    },
    "overlap_info": {
        "query_count": 20,
        "total_benchmark_queries": 500,
        "coverage": 0.04
    },
    "mds_coordinates": {"model_name": [0.1, -0.3], ...}
}
```

## Example: Full Workflow

```python
import quench

quench.api_key = "qk_..."

# Load benchmark
benchmark = quench.Benchmark.load("jailbreak-safety-v1")
print(f"{len(benchmark.models)} models, {len(benchmark.get_query_dictionary())} queries")

# Get optimal queries
optimal = benchmark.get_optimal_queries(budget=20)
query_ids = [(q["subtask"], q["query_id"]) for q in optimal["queries"]]
queries = benchmark.get_query_dictionary()

# Stage prediction
session = benchmark.stage(query_ids)

# Run your model on the optimal queries
responses = {}
for subtask, qid in query_ids:
    key = f"{subtask}/{qid}"
    question = queries.get(key, "")
    answer = my_model(question)  # Your model here
    if subtask not in responses:
        responses[subtask] = {}
    responses[subtask][qid] = {
        "question": question,
        "response": [answer],
    }

# Predict
results = benchmark.predict({"my-model": responses}, session=session)
print(f"Predicted score: {results['predicted_scores']['my-model']:.1%}")
```

## Feedback & Support

- **Issues:** [github.com/helivan-research/quench/issues](https://github.com/helivan-research/quench/issues)
- **Email:** info@helivan.io

## License

MIT

---

Built by [Helivan Research](https://helivan.io)
