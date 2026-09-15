"""Validate and copy the study's fixed splits from authorized metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


EXPECTED_SPLIT_COUNTS = {"development": 128, "validation": 19, "test": 37}
EXPECTED_SPLIT_SHA256 = {
    "development": "c3a17956928e1b16cde8adc0b47a2ab5ca59347ddf4701e92175ddc01bd9351e",
    "validation": "ab2e803f378cb08a6a6c5564471429caa2556bf4cd2141afe9e8f13ed903364d",
    "test": "10973c99f1aefc16e37db9eec229858afb62b19e7ce0b25a2666756a25dd33f1",
}


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


def load_fixed_splits(
    split_paths: Dict[str, str],
    data_root: str = "",
    require_files: bool = False,
    expected_counts: Dict[str, int] | None = None,
    expected_hashes: Dict[str, str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Load the original assignments and verify their privacy-safe fingerprints."""

    counts = dict(expected_counts or EXPECTED_SPLIT_COUNTS)
    hashes = dict(expected_hashes or EXPECTED_SPLIT_SHA256)
    if set(split_paths) != set(counts) or set(hashes) != set(counts):
        raise ValueError("Split paths, counts, and hashes must name the same partitions.")

    splits = {}
    for name in counts:
        path = split_paths[name]
        observed_hash = file_sha256(path)
        if observed_hash != hashes[name]:
            raise ValueError(
                f"{name} split fingerprint does not match the archived study assignment."
            )
        split = load_valid_sessions(path, data_root, require_files)
        if len(split) != counts[name]:
            raise ValueError(
                f"{name} split has {len(split)} valid sessions; expected {counts[name]}."
            )
        splits[name] = split

    validate_session_level_split(splits)
    return splits


def write_splits(
    splits: Dict[str, pd.DataFrame],
    output_dir: str,
    source_paths: Dict[str, str],
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    validate_session_level_split(splits)

    manifest = {
        "assignment": "preserved_from_fixed_authorized_split_metadata",
        "counts": {name: len(split) for name, split in splits.items()},
        "restricted_outputs": True,
        "source_files": {
            name: {
                "filename": Path(path).name,
                "sha256": file_sha256(path),
            }
            for name, path in source_paths.items()
        },
        "files": {},
    }
    for name, split in splits.items():
        output_path = target / f"{name}.csv"
        split.to_csv(output_path, index=False)
        manifest["files"][name] = {
            "filename": output_path.name,
            "sha256": file_sha256(str(output_path)),
        }

    with open(target / "split_manifest.json", "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fixed PHQ-8 evaluation splits.")
    parser.add_argument("--development-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--require-files", action="store_true")
    args = parser.parse_args()

    split_paths = {
        "development": args.development_csv,
        "validation": args.validation_csv,
        "test": args.test_csv,
    }
    splits = load_fixed_splits(split_paths, args.data_root, args.require_files)
    write_splits(splits, args.output_dir, split_paths)
    print(f"Validated fixed splits: { {name: len(split) for name, split in splits.items()} }")


if __name__ == "__main__":
    main()
