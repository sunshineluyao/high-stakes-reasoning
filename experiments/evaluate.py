"""Aggregate case-level PHQ-8 results without seed-level pseudoreplication."""

from __future__ import annotations

import argparse
import math
import os
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover - documented normal approximation fallback
    stats = None


REQUIRED_COLUMNS = {
    "configuration",
    "participant_id",
    "seed",
    "true_score",
    "predicted_score",
}


def validate_schema(results: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(results.columns))
    if missing:
        raise ValueError(f"Missing required result columns: {missing}")


def mean_absolute_error(values: pd.DataFrame) -> float:
    return float((values["true_score"] - values["predicted_score"]).abs().mean())


def bootstrap_ci(
    participant_errors: np.ndarray,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap participants, the study's independent evaluation unit."""

    if len(participant_errors) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bootstrap_means = [
        float(np.mean(rng.choice(participant_errors, size=len(participant_errors), replace=True)))
        for _ in range(iterations)
    ]
    alpha = 1.0 - confidence
    return (
        float(np.quantile(bootstrap_means, alpha / 2.0)),
        float(np.quantile(bootstrap_means, 1.0 - alpha / 2.0)),
    )


def paired_t_test(left: np.ndarray, right: np.ndarray) -> Dict[str, float]:
    """Paired test over one aggregated error per participant."""

    if len(left) != len(right) or len(left) < 2:
        return {"mean_paired_difference": float("nan"), "t_statistic": float("nan"), "p_value": float("nan")}
    differences = left - right
    mean_difference = float(np.mean(differences))
    std_difference = float(np.std(differences, ddof=1))
    if std_difference == 0:
        if mean_difference == 0:
            return {
                "mean_paired_difference": mean_difference,
                "t_statistic": 0.0,
                "p_value": 1.0,
            }
        return {
            "mean_paired_difference": mean_difference,
            "t_statistic": math.copysign(float("inf"), mean_difference),
            "p_value": 0.0,
        }
    statistic = mean_difference / (std_difference / math.sqrt(len(differences)))
    if stats is not None:
        p_value = float(stats.t.sf(abs(statistic), df=len(differences) - 1) * 2)
    else:
        p_value = float(math.erfc(abs(statistic) / math.sqrt(2)))
    return {
        "mean_paired_difference": mean_difference,
        "t_statistic": float(statistic),
        "p_value": p_value,
    }


def participant_error_table(group: pd.DataFrame) -> pd.DataFrame:
    values = group.dropna(subset=["predicted_score"]).copy()
    values["absolute_error"] = (values["true_score"] - values["predicted_score"]).abs()
    return (
        values.groupby("participant_id", as_index=False)
        .agg(
            true_score=("true_score", "first"),
            participant_mean_prediction=("predicted_score", "mean"),
            participant_mean_absolute_error=("absolute_error", "mean"),
            completed_seeds=("seed", "nunique"),
        )
    )


def compute_process_proxies(group: pd.DataFrame) -> Dict[str, float]:
    """Return transparent implementation proxies, not interpretability claims."""

    valid = group.dropna(subset=["predicted_score"]).copy()
    if valid.empty:
        return {
            "audit_correction_rate_model_proxy": float("nan"),
            "audit_unsupported_rate_model_proxy": float("nan"),
            "knowledge_reference_coverage_proxy": float("nan"),
            "rationale_presence_proxy": float("nan"),
            "mean_prediction_sd_across_seeds": float("nan"),
        }

    audit_mask = valid.get("audit_decision", pd.Series("", index=valid.index)).fillna("").ne("")
    audited = valid[audit_mask]
    audit_correction = (
        float(audited["audit_corrected"].astype(bool).mean())
        if not audited.empty and "audit_corrected" in audited
        else float("nan")
    )
    audit_unsupported = (
        float(audited["audit_unsupported_inference"].astype(bool).mean())
        if not audited.empty and "audit_unsupported_inference" in audited
        else float("nan")
    )
    reference_text = valid.get("knowledge_reference_ids", pd.Series("", index=valid.index)).fillna("").astype(str)
    rationale_text = valid.get("rationale", pd.Series("", index=valid.index)).fillna("").astype(str)
    prediction_sd = valid.groupby("participant_id")["predicted_score"].std(ddof=1)

    return {
        "audit_correction_rate_model_proxy": audit_correction,
        "audit_unsupported_rate_model_proxy": audit_unsupported,
        "knowledge_reference_coverage_proxy": float(reference_text.str.len().gt(0).mean()),
        "rationale_presence_proxy": float(rationale_text.str.strip().str.len().gt(0).mean()),
        "mean_prediction_sd_across_seeds": float(prediction_sd.dropna().mean()) if prediction_sd.notna().any() else 0.0,
    }


def summarize_results(results: pd.DataFrame, bootstrap_iterations: int) -> pd.DataFrame:
    validate_schema(results)
    summaries = []
    for configuration, group in results.groupby("configuration"):
        valid = group.dropna(subset=["predicted_score"]).copy()
        valid["absolute_error"] = (valid["true_score"] - valid["predicted_score"]).abs()
        seed_mae = valid.groupby("seed")["absolute_error"].mean()
        participant_errors = participant_error_table(valid)
        ci_lower, ci_upper = bootstrap_ci(
            participant_errors["participant_mean_absolute_error"].to_numpy(dtype=float),
            iterations=bootstrap_iterations,
        )
        summaries.append(
            {
                "configuration": configuration,
                "participant_count": int(participant_errors["participant_id"].nunique()),
                "completed_runs": int(len(valid)),
                "seeds": int(valid["seed"].nunique()),
                "mae_mean_across_seeds": float(seed_mae.mean()) if len(seed_mae) else float("nan"),
                "mae_sd_across_seeds": float(seed_mae.std(ddof=1)) if len(seed_mae) > 1 else 0.0,
                "participant_bootstrap_ci_lower": ci_lower,
                "participant_bootstrap_ci_upper": ci_upper,
                **compute_process_proxies(valid),
            }
        )
    return pd.DataFrame(summaries).sort_values("configuration")


def pairwise_comparisons(results: pd.DataFrame) -> pd.DataFrame:
    validate_schema(results)
    valid = results.dropna(subset=["predicted_score"]).copy()
    tables = {
        configuration: participant_error_table(group)[
            ["participant_id", "participant_mean_absolute_error"]
        ]
        for configuration, group in valid.groupby("configuration")
    }

    rows = []
    for left, right in combinations(sorted(tables), 2):
        merged = tables[left].merge(tables[right], on="participant_id", suffixes=("_left", "_right"))
        left_errors = merged["participant_mean_absolute_error_left"].to_numpy(dtype=float)
        right_errors = merged["participant_mean_absolute_error_right"].to_numpy(dtype=float)
        rows.append(
            {
                "left_configuration": left,
                "right_configuration": right,
                "paired_participants": int(len(merged)),
                "left_participant_mean_mae": float(left_errors.mean()) if len(merged) else float("nan"),
                "right_participant_mean_mae": float(right_errors.mean()) if len(merged) else float("nan"),
                **paired_t_test(left_errors, right_errors),
            }
        )
    return pd.DataFrame(rows)


def load_result_files(input_paths: List[str]) -> pd.DataFrame:
    frames = []
    for path in input_paths:
        csv_path = os.path.join(path, "case_results.csv") if os.path.isdir(path) else path
        if os.path.exists(csv_path):
            frames.append(pd.read_csv(csv_path))
    if not frames:
        raise ValueError("No case result files were found.")
    results = pd.concat(frames, ignore_index=True)
    validate_schema(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PHQ-8 experiment results.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results_summary")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_result_files(args.inputs)
    summarize_results(results, args.bootstrap_iterations).to_csv(
        os.path.join(args.output_dir, "metric_summary.csv"), index=False
    )
    pairwise_comparisons(results).to_csv(
        os.path.join(args.output_dir, "pairwise_participant_tests.csv"), index=False
    )
    print(f"Saved summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
