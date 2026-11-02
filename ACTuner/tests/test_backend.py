"""Measure one HNSW configuration without running the tuner."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from lrrl import (
    HNSWEnv,
    configure_hnsw_threads,
    load_gist1m,
    project_path,
    validate_target_recall,
)


SUPPORTED_TARGETS = (0.90, 0.92, 0.94, 0.96, 0.98, 0.99)


def main():
    parser = argparse.ArgumentParser(
        description="Build and evaluate exactly one GIST1M HNSW graph"
    )
    parser.add_argument("--target-recall", type=float, default=0.99)
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=400)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    target = validate_target_recall(args.target_recall)
    if not any(abs(target - value) <= 1e-12 for value in SUPPORTED_TARGETS):
        raise ValueError(
            f"Unsupported target {target}; supported={SUPPORTED_TARGETS}"
        )
    if not 4 <= args.M <= 100:
        raise ValueError("M must be in [4, 100]")
    if not max(20, args.M) <= args.ef_construction <= 800:
        raise ValueError("efConstruction must be in [20, 800] and >= M")

    project_dir = PROJECT_DIR
    output_dir = project_path(args.output_dir) if args.output_dir else (
        project_dir / "runs" / time.strftime("backend_smoke_%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    config_path = project_dir / "config.yaml"
    base, query, groundtruth = load_gist1m(config_path)
    from lrrl import CURRENT_CONFIG

    cfg = CURRENT_CONFIG
    configure_hnsw_threads(cfg.get("runtime", {}))
    hnsw_cfg = cfg["hnsw"]
    runtime_cfg = cfg["runtime"]
    env = HNSWEnv(
        base,
        query,
        groundtruth,
        R_target=target,
        efSearch_train=hnsw_cfg["efSearch_train"],
        efSearch_full=hnsw_cfg["efSearch_full"],
        n_query_sample=runtime_cfg["train_query_sample"],
        search_repeat=runtime_cfg["search_repeat"],
        recall_k=hnsw_cfg["recall_k"],
    )

    started = time.time()
    try:
        reached, max_recall, qps, _, interp, selected_rows = (
            env.build_and_evaluate(args.M, args.ef_construction)
        )
    except Exception as exc:
        error_path = output_dir / "smoke_error.json"
        with error_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "target_recall": target,
                    "M": args.M,
                    "efConstruction": args.ef_construction,
                    "wall_time_s": time.time() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.flush()
            os.fsync(handle.fileno())
        print(f"Smoke failure record saved to: {error_path}")
        raise
    elapsed = time.time() - started
    metrics = dict(env.last_evaluation or {})
    payload = {
        "target_recall": target,
        "M": args.M,
        "efConstruction": args.ef_construction,
        "target_reached": bool(reached),
        "max_recall": float(max_recall),
        "qps_at_target": float(qps),
        "wall_time_s": elapsed,
        "build_time_s": metrics.get("build_time_s"),
        "search_time_s": metrics.get("search_time_s"),
        "internal_build_time_s": metrics.get("internal_build_time_s"),
        "queries": len(query),
        "interp": interp,
        "selected_curve": selected_rows,
        "full_curve": metrics.get("raw_results_full", []),
        "binary_efsearch": metrics.get("binary_efsearch", []),
    }

    output_path = output_dir / "smoke_result.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Smoke result saved to: {output_path}")


if __name__ == "__main__":
    main()
