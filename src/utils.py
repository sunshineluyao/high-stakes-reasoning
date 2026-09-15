"""Load participant-derived DAIC-WOZ inputs from an authorized local copy."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def _read_transcript(path: Path) -> pd.DataFrame:
    transcript = pd.read_csv(path, sep="\t").fillna("")
    if "speaker" not in transcript.columns or "value" not in transcript.columns:
        transcript = pd.read_csv(path).fillna("")
    required = {"speaker", "value"}
    if not required.issubset(transcript.columns):
        raise ValueError(f"Transcript {path.name} is missing columns {sorted(required)}")
    return transcript


def _summarize_action_units(path: Path) -> str:
    action_units = pd.read_csv(path)
    required_columns = {"success", "AU12_r", "AU04_c"}
    missing = sorted(required_columns.difference(action_units.columns))
    if missing:
        raise ValueError(f"Action-unit file {path.name} is missing columns {missing}")
    valid = action_units[action_units["success"] == 1]
    if valid.empty:
        raise ValueError(f"Action-unit file {path.name} contains no successfully tracked frames")

    au12_mean = float(valid["AU12_r"].mean())
    au04_frequency = float(valid["AU04_c"].eq(1).mean())
    return (
        f"AU12 smile intensity mean: {au12_mean:.2f}; "
        f"AU04 brow-furrow activation frequency: {au04_frequency:.2%}"
    )


def load_participant_data(
    data_root: str,
    participant_id: int,
    include_au: bool = True,
) -> Tuple[str, str]:
    """Return participant-only transcript text and an optional two-value AU summary.

    The function does not remove lexical fillers or interviewer-dependent context
    from within participant turns. It only excludes rows whose speaker is not the
    participant, matching the documented study input boundary.
    """

    root = Path(data_root)
    transcript_path = root / f"{participant_id}_TRANSCRIPT.csv"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    transcript = _read_transcript(transcript_path)
    participant_turns = transcript.loc[
        transcript["speaker"].astype(str).str.casefold().eq("participant"), "value"
    ].astype(str)
    participant_text = " ".join(turn.strip() for turn in participant_turns if turn.strip())
    if not participant_text:
        raise ValueError(f"Transcript {transcript_path.name} has no participant text")

    if not include_au:
        return participant_text, ""

    action_unit_path = root / f"{participant_id}_CLNF_AUs.csv"
    if not action_unit_path.exists():
        raise FileNotFoundError(f"Action-unit file not found: {action_unit_path}")
    return participant_text, _summarize_action_units(action_unit_path)
