"""
End-to-end test for the Quench SDK against a local backend.

Usage:
    1. Start the backend: cd quench-backend && python app.py
    2. Run: python tests/test_e2e_local.py

Subsamples the jailbreak dataset to 5 training models + 1 holdout,
2 subtasks, 50 queries each — small enough to complete in <60s.
"""

import json
import logging
import os
import random
import sys
import time

# Add parent to path for local dev
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import quench
from quench import QuenchClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = os.environ.get("QUENCH_TEST_URL", "http://localhost:5001")
JAILBREAK_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..",
    "projects", "jailbreak-dkps", "data", "model_results.json",
)
# Resolve relative path
JAILBREAK_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "projects", "jailbreak-dkps", "data", "model_results.json")
)
# Fallback to environment variable
if not os.path.exists(JAILBREAK_PATH):
    JAILBREAK_PATH = os.environ.get("JAILBREAK_DATA_PATH", "data/model_results.json")

BENCHMARK_NAME = f"e2e_test_{int(time.time())}"
TRAIN_MODELS = 5
HOLDOUT_INDEX = 5
SUBTASKS = ["non_harmful", "violence_and_harm"]
QUERIES_PER_SUBTASK = 50

# Test user credentials (set via environment or defaults for local testing)
TEST_EMAIL = os.environ.get("QUENCH_TEST_EMAIL", "sdk_e2e_test@example.com")
TEST_PASSWORD = os.environ.get("QUENCH_TEST_PASSWORD", "TestPass_E2E_123!")


def subsample(data, model_names, subtasks, max_queries):
    """Extract a small subset of the full dataset."""
    out = {}
    for name in model_names:
        model = data[name]
        out[name] = {}
        for st in subtasks:
            st_data = model[st]
            qids = [k for k in st_data if k not in ("metadata", "score", "query_score")]
            selected = qids[:max_queries]
            out[name][st] = {qid: st_data[qid] for qid in selected}
        if "score" in model:
            out[name]["score"] = model["score"]
    return out


def step(name):
    """Print step header and return timer."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    return time.time()


def elapsed(t0):
    dt = time.time() - t0
    print(f"  [{dt:.2f}s]")
    return dt


def main():
    timings = {}

    # -- Load and subsample data -------------------------------------------
    t0 = step("0. Load and subsample jailbreak data")
    if not os.path.exists(JAILBREAK_PATH):
        print(f"  SKIP: {JAILBREAK_PATH} not found")
        sys.exit(1)

    with open(JAILBREAK_PATH) as f:
        full_data = json.load(f)

    all_models = [k for k in full_data if k != "metadata"]
    random.seed(42)
    random.shuffle(all_models)
    train_models = all_models[:TRAIN_MODELS]
    holdout_model = all_models[HOLDOUT_INDEX]

    train_data = subsample(full_data, train_models, SUBTASKS, QUERIES_PER_SUBTASK)
    holdout_data = subsample(full_data, [holdout_model], SUBTASKS, QUERIES_PER_SUBTASK)

    n_queries = sum(
        len([k for k in train_data[m][st] if k not in ("metadata", "score", "query_score")])
        for m in train_models for st in SUBTASKS
    )
    payload_size = len(json.dumps(train_data))
    print(f"  Train: {len(train_models)} models, {n_queries} queries, {payload_size/1024:.0f} KB")
    print(f"  Holdout: {holdout_model}")
    timings["subsample"] = elapsed(t0)

    # -- Authenticate ------------------------------------------------------
    t0 = step("1. Authenticate")
    client = QuenchClient(base_url=BASE_URL)

    # Login to get a token
    resp = client._session.post(
        f"{BASE_URL}/auth/login",
        json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
        timeout=10,
    )
    if not resp.ok:
        print(f"  Login failed: {resp.json()}")
        sys.exit(1)
    token_data = resp.json()
    client._jwt_token = token_data["token"]
    client._jwt_user_id = token_data["user_id"]
    print(f"  Logged in as {token_data['username']} ({token_data['user_id']})")
    timings["auth"] = elapsed(t0)

    # -- Create org for test user ------------------------------------------
    t0 = step("1b. Create organization")
    org_name = f"sdk_e2e_test99/e2e-{int(time.time())}"
    resp = client._session.post(
        f"{BASE_URL}/organizations",
        headers=client._auth_headers(),
        json={"organization_name": org_name, "display_name": "E2E Test Org"},
        timeout=10,
    )
    if not resp.ok:
        print(f"  Org creation failed ({resp.status_code}): {resp.json()}")
        sys.exit(1)
    org_data = resp.json()
    org_id = org_data.get("organization_id")
    print(f"  Created org: {org_name} ({org_id})")
    elapsed(t0)

    # -- Create benchmark --------------------------------------------------
    t0 = step("2. Create benchmark")
    # Pass organization_id so the benchmark goes into our org
    bm = client.benchmarks.create(
        BENCHMARK_NAME,
        train_data,
        embedding_model="google/gemini-embedding-001",
        category="safety",
        organization_id=org_id,
    )
    print(f"  Name: {bm.name}")
    print(f"  Status: {bm.status}")
    print(f"  Benchmark ID: {bm._benchmark_id}")
    timings["create"] = elapsed(t0)

    # -- Wait for processing -----------------------------------------------
    t0 = step("3. Wait until ready")
    bm.wait_until_ready(timeout=300, poll_interval=5)
    print(f"  Status: {bm.status}")
    timings["wait"] = elapsed(t0)

    # -- Load benchmark back -----------------------------------------------
    t0 = step("4. Load benchmark")
    bm2 = client.benchmarks.load(BENCHMARK_NAME)
    print(f"  Models: {bm2.models}")
    print(f"  Status: {bm2.status}")
    assert len(bm2.models) == TRAIN_MODELS, f"Expected {TRAIN_MODELS} models, got {len(bm2.models)}"
    timings["load"] = elapsed(t0)

    # -- Summary -----------------------------------------------------------
    t0 = step("5. Summary")
    summary = bm2.summary()
    print(f"  Models in summary: {len(summary.get('models', []))}")
    print(f"  Status: {summary.get('status')}")
    for m in summary.get("models", [])[:3]:
        print(f"    {m.get('name', '?')}: score={m.get('score', '?')}")
    timings["summary"] = elapsed(t0)

    # -- Predict -----------------------------------------------------------
    t0 = step("6. Predict (holdout model, stored benchmark)")
    holdout_responses = holdout_data[holdout_model]
    # Use 50% of holdout queries
    partial = {}
    for st, st_data in holdout_responses.items():
        if st == "score":
            continue
        qids = list(st_data.keys())
        n = max(1, len(qids) // 2)
        partial[st] = {qid: st_data[qid] for qid in qids[:n]}

    total_q = sum(len(v) for v in partial.values())
    actual = full_data[holdout_model].get("score")
    print(f"  Holdout: {holdout_model} (actual score: {actual})")
    print(f"  Using {total_q} queries")

    results = bm2.predict({holdout_model: partial})
    predicted = results.get("predicted_scores", {}).get(holdout_model)
    ci = results.get("confidence_interval", {}).get(holdout_model)
    print(f"  Predicted: {predicted}")
    print(f"  CI: {ci}")
    if predicted is not None and actual is not None:
        print(f"  Error: {abs(predicted - actual):.4f}")
    assert predicted is not None, f"predicted_scores[{holdout_model}] is None — check response key mismatch"
    assert ci is not None, "confidence_interval is None"
    timings["predict"] = elapsed(t0)

    # -- Add model ---------------------------------------------------------
    t0 = step("7. Add model")
    add_data = holdout_data[holdout_model]
    bm2.add_model(add_data, model_name=holdout_model)
    print(f"  Models after add: {bm2.models}")
    assert holdout_model in bm2.models
    timings["add_model"] = elapsed(t0)

    # -- Remove model ------------------------------------------------------
    t0 = step("8. Remove model")
    bm2.remove_model(holdout_model)
    print(f"  Models after remove: {bm2.models}")
    assert holdout_model not in bm2.models
    timings["remove_model"] = elapsed(t0)

    # -- Delete benchmark --------------------------------------------------
    t0 = step("9. Delete benchmark")
    bm2.delete()
    print(f"  Deleted {BENCHMARK_NAME}")

    # Verify it's gone
    try:
        client.benchmarks.load(BENCHMARK_NAME)
        print("  ERROR: benchmark still exists!")
    except quench.BenchmarkNotFoundError:
        print("  Confirmed deleted")
    timings["delete"] = elapsed(t0)

    # -- Cleanup org -------------------------------------------------------
    t0 = step("10. Cleanup organization")
    resp = client._session.delete(
        f"{BASE_URL}/organizations/{org_id}",
        headers=client._auth_headers(),
        timeout=10,
    )
    print(f"  Delete org: {resp.status_code}")
    elapsed(t0)

    # -- Summary -----------------------------------------------------------
    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    total = sum(timings.values())
    for name, dt in timings.items():
        print(f"  {name:20s} {dt:6.2f}s")
    print(f"  {'TOTAL':20s} {total:6.2f}s")
    print()
    print("  ALL E2E TESTS PASSED")
    print()


if __name__ == "__main__":
    main()
