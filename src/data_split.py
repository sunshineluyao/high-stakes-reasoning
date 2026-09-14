"""Create deterministic, non-overlapping session splits from authorized metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


DEFAULT_SPLIT_COUNTS = {"development": 128, "validation": 19, "test": 37}


def _resolve_columns(df: pd.DataFrame) -> Tuple[str, str]:
    id_candidates = ["Participant_ID", "participant_id", "session_id", "Session_ID"]
    score_candidates = ["PHQ8_Score", "phq8_score", "PHQ_Score", "score"]
    id_column = next((column for column in id_candidates if column in df.columns), None)
    score_column = next((column for column in score_candidates if column in df.columns), None)
    if id_column is None or score_column is None:
        raise ValueError(
            "The index must contain a participant/session identifier and PHQ-8 score. "
            f"Available columns: {list(df.columns)}"
        )
    return id_column, score_column


def load_valid_sessions(index_csv: str, data_root: str = "", require_files: bool = False) -> pd.DataFrame:
    sessions = pd.read_csv(index_csv)
    id_column, score_column = _resolve_columns(sessions)
    valid = sessions.dropna(subset=[id_column, score_column]).copy()
    valid = valid.drop_duplicates(subset=[id_column], keep="first")
    valid[id_column] = valid[id_column].astype(int)

    if require_files:
        if not data_root:
            raise ValueError("--data-root is required with --require-files")
        root = Path(data_root)
        valid = valid[
            valid[id_column].apply(
                lambda participant_id: (root / f"{int(participant_id)}_TRANSCRIPT.csv").exists()
                and (root / f"{int(participant_id)}_CLNF_AUs.csv").exists()
            )
        ].copy()

    return valid.sort_values(id_column).reset_index(drop=True)


def split_sessions(
    sessions: pd.DataFrame,
    seed: int = 42,
    split_counts: Dict[str, int] | None = None,
) -> Dict[str, pd.DataFrame]:
    counts = dict(split_counts or DEFAULT_SPLIT_COUNTS)
    shuffled = sessions.sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(shuffled) != sum(counts.values()):
        development = round(len(shuffled) * 0.70)
        validation = round(len(shuffled) * 0.10)
        counts = {
            "development": development,
            "validation": validation,
            "test": len(shuffled) - development - validation,
        }

    development_end = counts["development"]
    validation_end = development_end + counts["validation"]
    splits = {
        "development": shuffled.iloc[:development_end].reset_index(drop=True),
        "validation": shuffled.iloc[development_end:validation_end].reset_index(drop=True),
        "test": shuffled.iloc[validation_end:].reset_index(drop=True),
    }
    validate_session_level_split(splits)
    return splits


def validate_session_level_split(splits: Dict[str, pd.DataFrame]) -> None:
    if not splits:
        raise ValueError("No splits supplied")
    first = next(iter(splits.values()))
    id_column, _ = _resolve_columns(first)
    id_sets = {name: set(split[id_column].astype(int)) for name, split in splits.items()}
    names = list(id_sets)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = id_sets[left_name].intersection(id_sets[right_name])
            if overlap:
                raise ValueError(
                    f"Data leakage between {left_name} and {right_name}: {sorted(overlap)}"
                )


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_splits(splits: Dict[str, pd.DataFrame], output_dir: str, seed: int, source_csv: str) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    validate_session_level_split(splits)

    manifest = {
        "source_filename": Path(source_csv).name,
        "source_sha256": file_sha256(source_csv),
        "seed": seed,
        "counts": {name: len(split) for name, split in splits.items()},
        "restricted_outputs": True,
        "files": {},
    }
    for name, split in splits.items():
        output_path = target / f"{name}.csv"
        split.to_csv(output_path, index=False)
        manifest["files"][name] = output_path.name

    with open(target / "split_manifest.json", "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create session-level PHQ-8 evaluation splits.")
    parser.add_argument("--index-csv", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-files", action="store_true")
    args = parser.parse_args()

    valid = load_valid_sessions(args.index_csv, args.data_root, args.require_files)
    splits = split_sessions(valid, seed=args.seed)
    write_splits(splits, args.output_dir, args.seed, args.index_csv)
    print(f"Created session-level splits: { {name: len(split) for name, split in splits.items()} }")


if __name__ == "__main__":
    main()
