"""Write parameter evaluations, timing records, and run summaries."""

import os
import re

import numpy as np
import pandas as pd


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def _ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)
    per_param_dir = os.path.join(LOG_DIR, "per_param")
    os.makedirs(per_param_dir, exist_ok=True)
    return per_param_dir


def _has_rows(value):
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return False


def _first_present(mapping, names, default=None):
    if not mapping:
        return default
    for name in names:
        if name in mapping and mapping[name] is not None and mapping[name] != "":
            return mapping[name]
    return default


def _optional_float(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _optional_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _excel_value(value):
    return "N/A" if value is None else value


def _safe_component(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return text or "unknown"


def _unique_filepath(directory, filename):
    stem, extension = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    suffix = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{stem}_repeat{suffix:02d}{extension}")
        suffix += 1
    return candidate


def _infer_phase(source=None, phase=None):
    if phase:
        phase_text = str(phase)
    else:
        source_text = str(source or "").lower()
        if "final" in source_text or "remeasure" in source_text:
            phase_text = "final"
        elif "warmup" in source_text or "warm-up" in source_text:
            phase_text = "warmup"
        else:
            phase_text = "phase2_real"
    if "final" in phase_text.lower() or "remeasure" in phase_text.lower():
        return "final"
    return phase_text


def _timing_values(record=None, fallback_eval_wall_time=None):
    record = record or {}
    flattened = {}
    for nested_name in ("evaluation_metrics", "timing", "metrics"):
        nested = record.get(nested_name)
        if isinstance(nested, dict):
            flattened.update(nested)
    flattened.update(record)
    record = flattened
    build_time_s = _optional_float(_first_present(
        record,
        (
            "build_wall_time_s", "build_time_s", "index_build_time_s",
            "backend_build_time_s",
        ),
    ))
    search_time_s = _optional_float(_first_present(
        record,
        (
            "search_wall_time_s", "search_time_s", "query_wall_time_s",
            "query_time_s", "backend_search_time_s",
        ),
    ))
    eval_time_s = _optional_float(_first_present(
        record,
        (
            "eval_wall_time_s", "evaluation_wall_time_s",
            "evaluation_time_s", "eval_time_s", "wall_time_s", "time",
        ),
    ))
    if eval_time_s is None:
        eval_time_s = _optional_float(fallback_eval_wall_time)
    if eval_time_s is None:
        eval_time_s = _optional_float(_first_present(
            record, ("total_eval_time_s", "total_evaluation_time_s")
        ))
    if eval_time_s is None and (build_time_s is not None or search_time_s is not None):
        eval_time_s = (build_time_s or 0.0) + (search_time_s or 0.0)

    cum_time_s = _optional_float(_first_present(
        record, ("cum_time_s", "cumulative_time_s", "elapsed_time_s")
    ))
    cum_time_min = _optional_float(_first_present(
        record, ("cum_time_min", "cumulative_time_min", "elapsed_time_min")
    ))
    if cum_time_s is None and cum_time_min is not None:
        cum_time_s = cum_time_min * 60.0
    if cum_time_min is None and cum_time_s is not None:
        cum_time_min = cum_time_s / 60.0

    return {
        "build_wall_time_s": build_time_s,
        "search_wall_time_s": search_time_s,
        "eval_wall_time_s": eval_time_s,
        "cum_time_s": cum_time_s,
        "cum_time_min": cum_time_min,
    }


def _curve_dataframe(results):
    rows = []
    if not _has_rows(results):
        return pd.DataFrame(columns=["efSearch", "Recall@10", "QPS"])

    if isinstance(results, pd.DataFrame):
        iterable = results.to_dict("records")
    else:
        iterable = results

    for item in iterable:
        if isinstance(item, dict):
            ef_search = _first_present(
                item, ("efSearch", "ef_search", "efs", "ef")
            )
            recall = _first_present(
                item, ("Recall@10", "recall@10", "recall", "Recall")
            )
            qps = _first_present(item, ("QPS", "qps", "search_qps"))
        else:
            try:
                ef_search, recall, qps = item[0], item[1], item[2]
            except (TypeError, IndexError):
                continue
        ef_value = _optional_int(ef_search)
        recall_value = _optional_float(recall)
        qps_value = _optional_float(qps)
        if ef_value is None or recall_value is None or qps_value is None:
            continue
        rows.append({
            "efSearch": ef_value,
            "Recall@10": recall_value,
            "QPS": qps_value,
        })
    return pd.DataFrame(rows, columns=["efSearch", "Recall@10", "QPS"])


def _record_timing_row(record, fallback_eval_wall_time=None):
    timing = _timing_values(record, fallback_eval_wall_time)
    return {
        "build_wall_time_s": _excel_value(timing["build_wall_time_s"]),
        "search_wall_time_s": _excel_value(timing["search_wall_time_s"]),
        "eval_wall_time_s": _excel_value(timing["eval_wall_time_s"]),
        "cum_time_s": _excel_value(timing["cum_time_s"]),
        "cum_time_min": _excel_value(timing["cum_time_min"]),
    }


def save_param_result(M, efC, recall, qps_at_target, reward, interp, raw_results,
                      supplied_full_results=None, build_time=None, source="unknown",
                      step=None, survived=True,
                      surrogate_mu=None, surrogate_sigma=None, acq_score=None,
                      llm_anchor=None, action_std=None, trust=None,
                      target_recall=0.99, cum_time_min=None, **metadata):
    per_param_dir = _ensure_dirs()
    phase = _infer_phase(source, metadata.get("phase"))
    global_eval_index = _optional_int(_first_present(
        metadata, ("global_eval_index", "real_eval_index", "eval_index")
    ))

    if phase == "final":
        global_eval_index = None

    ppo_step = _first_present(metadata, ("ppo_step", "step"), step)
    timing_metadata = dict(metadata)
    if cum_time_min is not None and "cum_time_min" not in timing_metadata:
        timing_metadata["cum_time_min"] = cum_time_min
    timing = _timing_values(timing_metadata, fallback_eval_wall_time=build_time)

    if global_eval_index is not None:
        identity = f"eval_{global_eval_index:03d}"
    elif phase == "final":
        identity = "final"
    else:
        identity = f"step_{_safe_component(ppo_step)}"
    filename = (
        f"M_{int(M)}_EFC_{int(efC)}_"
        f"{_safe_component(phase)}_{identity}.xlsx"
    )
    filepath = _unique_filepath(per_param_dir, filename)
    qps_label = f"QPS@{target_recall}"

    if isinstance(llm_anchor, dict):
        anchor_M = llm_anchor.get("M", "N/A")
        anchor_efC = llm_anchor.get(
            "efC", llm_anchor.get("efConstruction", "N/A")
        )
    elif isinstance(llm_anchor, (tuple, list)) and len(llm_anchor) >= 2:
        anchor_M, anchor_efC = llm_anchor[0], llm_anchor[1]
    else:
        anchor_M = anchor_efC = "N/A"

    evaluation_metrics = metadata.get("evaluation_metrics")
    if not isinstance(evaluation_metrics, dict):
        evaluation_metrics = {}
    effective_raw_results = (
        raw_results
        if _has_rows(raw_results)
        else evaluation_metrics.get("raw_results")
    )
    effective_full_results = metadata.get("full_results")
    if not _has_rows(effective_full_results):
        effective_full_results = supplied_full_results
    if not _has_rows(effective_full_results):
        effective_full_results = evaluation_metrics.get("raw_results_full")
    training_curve = _curve_dataframe(effective_raw_results)
    full_curve = _curve_dataframe(effective_full_results)
    performance_curve = full_curve if not full_curve.empty else training_curve
    curve_values = (
        ",".join(str(int(value)) for value in performance_curve["efSearch"].tolist())
        if not performance_curve.empty else "N/A"
    )

    summary_items = [
        ("M", int(M)),
        ("efConstruction", int(efC)),
        ("Recall@10", float(recall)),
        (qps_label, float(qps_at_target)),
        ("Reward", float(reward)),
        (f"Survived (recall>={target_recall})", bool(survived)),
        ("Source", str(source)),
        ("Phase", phase),
        ("Global Eval Index", _excel_value(global_eval_index)),
        ("Counts Toward Tuning Budget", phase != "final"),
        ("PPO Step", _excel_value(ppo_step)),
        ("Build Wall Time (s)", _excel_value(timing["build_wall_time_s"])),
        ("Search Wall Time (s)", _excel_value(timing["search_wall_time_s"])),
        ("Evaluation Wall Time (s)", _excel_value(timing["eval_wall_time_s"])),
        ("Cumulative Time (s)", _excel_value(timing["cum_time_s"])),
        ("Cumulative Time (min)", _excel_value(timing["cum_time_min"])),
        ("Curve Point Count", int(len(performance_curve))),
        ("Curve Complete (12 points)", len(performance_curve) == 12),
        ("Curve efSearch", curve_values),
        ("Surrogate Mu", _excel_value(_optional_float(surrogate_mu))),
        ("Surrogate Sigma", _excel_value(_optional_float(surrogate_sigma))),
        ("Acquisition Score", _excel_value(_optional_float(acq_score))),
        ("LLM Anchor M", anchor_M),
        ("LLM Anchor efC", anchor_efC),
        ("Action Std", _excel_value(_optional_float(action_std))),
        ("Trust", _excel_value(_optional_float(trust))),
    ]

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        pd.DataFrame(summary_items, columns=["Parameter", "Value"]).to_excel(
            writer, sheet_name="Summary", index=False
        )

        if not training_curve.empty:
            training_curve.to_excel(
                writer, sheet_name="Training_Sweep", index=False
            )

        if interp:
            interp_rows = []
            for recall_target in sorted(interp, key=float):
                value = _optional_float(interp[recall_target])
                if value is not None:
                    interp_rows.append({
                        "Recall Target": float(recall_target),
                        "QPS": value,
                    })
            if interp_rows:
                pd.DataFrame(interp_rows).to_excel(
                    writer, sheet_name="Interpolated_QPS", index=False
                )

        if not full_curve.empty:
            full_curve.to_excel(
                writer, sheet_name="Full_Evaluation", index=False
            )

        if not performance_curve.empty:


            performance_curve.to_excel(
                writer, sheet_name="Performance_Curve", index=False
            )

    return filepath


def _real_eval_row(record, global_eval_index, default_phase, qps_label,
                   default_ppo_step=None):
    phase = _infer_phase(record.get("source"), record.get("phase", default_phase))
    ppo_step = _first_present(
        record, ("ppo_step", "step", "timestep"), default_ppo_step
    )
    timing = _timing_values(record)
    efc = _first_present(record, ("efConstruction", "efC"))
    row = {


        "Step": global_eval_index,
        "global_eval_index": global_eval_index,
        "phase": phase,
        "ppo_step": _excel_value(ppo_step),
        "M": int(record["M"]),
        "efC": int(efc),
        "Recall@10": float(record["recall"]),
        qps_label: float(record["qps_at_target"]),
        "Reward": float(record["reward"]),
        "Source": record.get("source", default_phase),
        "Surrogate Mu": (
            float(record["surrogate_mu"])
            if record.get("surrogate_mu") is not None else ""
        ),
        "Surrogate Sigma": (
            float(record["surrogate_sigma"])
            if record.get("surrogate_sigma") is not None else ""
        ),
        "build_wall_time_s": _excel_value(timing["build_wall_time_s"]),
        "search_wall_time_s": _excel_value(timing["search_wall_time_s"]),
        "eval_wall_time_s": _excel_value(timing["eval_wall_time_s"]),
        "cum_time_s": _excel_value(timing["cum_time_s"]),
        "cum_time_min": _excel_value(timing["cum_time_min"]),
    }
    return row


def _assigned_global_index(record, used_indices, next_index):
    requested = _optional_int(_first_present(
        record, ("global_eval_index", "real_eval_index", "eval_index")
    ))
    if requested is not None and requested > 0 and requested not in used_indices:
        assigned = requested
    else:
        assigned = next_index
        while assigned in used_indices:
            assigned += 1
    used_indices.add(assigned)
    return assigned, max(next_index, assigned + 1)


def _write_final_remeasure(best_result, qps_label):
    if not best_result or "final_remeasure_recall" not in best_result:
        return
    final_metrics = _first_present(
        best_result,
        (
            "final_remeasure_metrics", "final_evaluation_metrics",
            "final_metrics",
        ),
        {},
    )
    timing_record = dict(final_metrics) if isinstance(final_metrics, dict) else {}
    timing_record.update({
        "build_wall_time_s": _first_present(
            best_result,
            ("final_remeasure_build_wall_time_s",
             "final_remeasure_build_time_s"),
        ),
        "search_wall_time_s": _first_present(
            best_result,
            ("final_remeasure_search_wall_time_s",
             "final_remeasure_search_time_s"),
        ),
        "eval_wall_time_s": _first_present(
            best_result,
            ("final_remeasure_eval_wall_time_s",
             "final_remeasure_time_s"),
        ),
        "cum_time_s": _first_present(
            best_result, ("final_remeasure_cum_time_s",)
        ),
        "cum_time_min": _first_present(
            best_result, ("final_remeasure_cum_time_min",)
        ),
    })
    timing = _timing_values(timing_record)
    row = {
        "global_eval_index": "",
        "phase": "final",
        "validation_mode": best_result.get(
            "final_validation_mode", "remeasure"
        ),
        "counts_toward_tuning_budget": False,
        "selection_ppo_step": best_result.get("step", ""),
        "M": int(best_result["M"]),
        "efC": int(_first_present(best_result, ("efConstruction", "efC"))),
        "Recall@10": float(best_result["final_remeasure_recall"]),
        qps_label: float(best_result.get("final_remeasure_qps", 0.0)),
        "survived": bool(best_result.get("final_remeasure_survived", False)),
        "build_wall_time_s": _excel_value(timing["build_wall_time_s"]),
        "search_wall_time_s": _excel_value(timing["search_wall_time_s"]),
        "eval_wall_time_s": _excel_value(timing["eval_wall_time_s"]),
        "cum_time_s": _excel_value(timing["cum_time_s"]),
        "cum_time_min": _excel_value(timing["cum_time_min"]),
    }
    pd.DataFrame([row]).to_excel(
        os.path.join(LOG_DIR, "final_validation.xlsx"), index=False
    )


def save_aggregated_results(all_history, warmup_results, best_result,
                            surrogate_train_history, llm_suggestions,
                            source_stats, stats, total_time, llm_call_count,
                            final_trust, final_action_std, n_real_evals,
                            total_surrogate_evals, efSearch_train, efSearch_full,
                            llm_usage=None, target_recall=0.99, **metadata):
    _ensure_dirs()
    qps_label = f"QPS@{target_recall}"


    merged_records = []
    for local_index, record in enumerate(warmup_results or [], start=1):
        merged_records.append((record, "warmup", local_index))
    for record in all_history or []:
        if record.get("recall") is None:
            continue
        phase = _infer_phase(record.get("source"), record.get("phase", "phase2_real"))
        if phase == "final":
            continue
        merged_records.append((record, phase, None))

    real_rows = []
    used_indices = set()
    next_index = 1
    for record, default_phase, default_ppo_step in merged_records:
        index, next_index = _assigned_global_index(
            record, used_indices, next_index
        )
        real_rows.append(_real_eval_row(
            record, index, default_phase, qps_label, default_ppo_step
        ))
    if real_rows:
        pd.DataFrame(real_rows).to_excel(
            os.path.join(LOG_DIR, "all_real_evals.xlsx"), index=False
        )


    surrogate_rows = []
    for index, record in enumerate(all_history or [], start=1):
        timing = _timing_values(record)
        surrogate_rows.append({
            "Index": index,
            "phase": _infer_phase(
                record.get("source"), record.get("phase", "phase2")
            ),
            "ppo_step": _excel_value(_first_present(
                record, ("ppo_step", "step", "timestep")
            )),
            "M": int(record["M"]),
            "efC": int(_first_present(record, ("efConstruction", "efC"))),
            "Real Recall": (
                float(record["recall"])
                if record.get("recall") is not None else "N/A"
            ),
            f"Real {qps_label}": (
                float(record["qps_at_target"])
                if record.get("qps_at_target") is not None else "N/A"
            ),
            "Real Reward": (
                float(record["reward"])
                if record.get("reward") is not None else ""
            ),
            "Surrogate Mu": (
                float(record["surrogate_mu"])
                if record.get("surrogate_mu") is not None else ""
            ),
            "Surrogate Sigma": (
                float(record["surrogate_sigma"])
                if record.get("surrogate_sigma") is not None else ""
            ),
            "Source": record.get("source", "phase2"),
            "eval_wall_time_s": _excel_value(timing["eval_wall_time_s"]),
            "cum_time_s": _excel_value(timing["cum_time_s"]),
            "cum_time_min": _excel_value(timing["cum_time_min"]),
        })
    if surrogate_rows:
        pd.DataFrame(surrogate_rows).to_excel(
            os.path.join(LOG_DIR, "all_surrogate_predictions.xlsx"),
            index=False,
        )


    if surrogate_train_history:
        training_rows = [
            {
                "Retrain Step": item["step"],
                "Samples": item["n_samples"],
                "Recall MAE": item["recall_mae"],
                "Train Loss": item.get("train_loss", ""),
            }
            for item in surrogate_train_history
        ]
        pd.DataFrame(training_rows).to_excel(
            os.path.join(LOG_DIR, "surrogate_training_history.xlsx"),
            index=False,
        )


    if llm_suggestions:
        suggestion_rows = []
        for suggestion in llm_suggestions:
            suggestion_rows.append({
                "Epoch": suggestion.get("epoch", ""),
                "Suggested M": suggestion.get("M", ""),
                "Suggested efC": suggestion.get(
                    "efConstruction", suggestion.get("efC", "")
                ),
                "Reasoning": str(suggestion.get("reasoning", ""))[:500],
                "Fallback": suggestion.get("fallback", False),
            })
        pd.DataFrame(suggestion_rows).to_excel(
            os.path.join(LOG_DIR, "llm_suggestions.xlsx"), index=False
        )


    if warmup_results:
        warmup_rows = []
        for index, record in enumerate(warmup_results, start=1):
            warmup_rows.append(_real_eval_row(
                record, index, "warmup", qps_label, index
            ))
        pd.DataFrame(warmup_rows).to_excel(
            os.path.join(LOG_DIR, "warmup_results.xlsx"), index=False
        )


    _write_final_remeasure(best_result, qps_label)


    llm_usage = llm_usage or {}
    summary_rows = [
        ["Total Time (s)", total_time],
        ["Total Time (min)", total_time / 60],
        ["Real Evaluations", n_real_evals],
        ["Tuning Real Evaluations", n_real_evals],
        ["Final Remeasure Counts Toward Tuning Budget", False],
        ["Surrogate Evaluations", total_surrogate_evals],
        ["Efficiency Gain", total_surrogate_evals / max(n_real_evals, 1)],
        ["LLM API Calls", llm_call_count],
        ["LLM Calls with Token Usage", llm_usage.get("calls_with_usage", 0)],
        ["LLM Prompt Chars", llm_usage.get("prompt_chars", 0)],
        ["LLM Prompt Tokens", llm_usage.get("prompt_tokens", 0)],
        ["LLM Completion Tokens", llm_usage.get("completion_tokens", 0)],
        ["LLM Total Tokens", llm_usage.get("total_tokens", 0)],
        ["LLM Max Tokens Requested", llm_usage.get("max_tokens_requested", 0)],
        ["Final Trust", final_trust],
        ["Final Action Std", final_action_std],
        ["Source Distribution", str(source_stats)],
        ["Best M", best_result["M"] if best_result else "N/A"],
        ["Best efC", best_result["efC"] if best_result else "N/A"],
        ["Best Recall", best_result["recall"] if best_result else "N/A"],
        [f"Best {qps_label}",
         best_result["qps_at_target"] if best_result else "N/A"],
        ["Final Remeasure Reached",
         best_result.get("final_remeasure_survived", "N/A")
         if best_result else "N/A"],
        ["Final Remeasure Recall",
         best_result.get("final_remeasure_recall", "N/A")
         if best_result else "N/A"],
        [f"Final Remeasure {qps_label}",
         best_result.get("final_remeasure_qps", "N/A")
         if best_result else "N/A"],
        ["Target Recall", target_recall],
        ["Training efSearch", str(efSearch_train)],
        ["Full efSearch", str(efSearch_full)],
    ]
    for name, value in stats.items():
        try:
            formatted_value = f"{float(value):.1f}s"
        except (TypeError, ValueError):
            formatted_value = str(value)
        summary_rows.append([f"Time: {name}", formatted_value])
    pd.DataFrame(summary_rows, columns=["Metric", "Value"]).to_excel(
        os.path.join(LOG_DIR, "run_summary.xlsx"), index=False
    )

    return LOG_DIR
