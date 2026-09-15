# Auditing Multi-Agent LLM Workflows for Depression-Severity Estimation

**Anonymous review artifact for a bounded DAIC-WOZ study**

This repository implements an inference-only workflow that separates evidence
summarization, retrieval of compact PHQ-8 criterion summaries, item-level score
estimation, and a final model-based audit. This double-blind review snapshot
deliberately distinguishes implemented behavior from clinical validation.

## Scientific question and reported result

The study asks whether modular task decomposition lowers PHQ-8 prediction error
relative to direct and chain-of-thought prompting under matched local LLM
backbones.

The archived aggregate table reports a best mean absolute error (MAE) reduction
from **5.35 to 5.02** for Qwen2.5 across three inference seeds. These values are
transcribed in [`artifact/reported_metrics.csv`](artifact/reported_metrics.csv).
Case-level model outputs are not distributed, so the aggregate values should be
treated as reported results rather than an independently reproduced release.

The artifact does **not** establish:

- a mental-health diagnosis;
- clinical utility, calibration, or decision-threshold performance;
- faithful or clinician-validated explanations;
- fairness across demographic groups; or
- transfer from structured interviews to social-media language.

## Implemented study design

```mermaid
flowchart LR
    A[Participant transcript<br/>plus two AU summaries] --> B[Evidence summary]
    B --> C[PHQ-8 criterion retrieval]
    C --> D[Item-level score estimate]
    D --> E[Model-based audit]
    E --> F[PHQ-8 total<br/>and inspectable rationale]
```

The input loader uses participant-only transcript turns and, in the default
`transcript_au_summary` condition, two precomputed OpenFace statistics: mean
AU12 smile intensity and AU04 brow-furrow activation frequency. Raw audio and
video are not used. A `text` condition is also implemented, but no result is
claimed for it unless separately evaluated.

All generative roles use the same selected backbone in a reported backbone
condition. Retrieval uses `nomic-embed-text` with top-$k=3$ when available and a
deterministic lexical fallback otherwise. The workflow uses composable Python
role functions in a fixed sequence and local Ollama inference.

## Repository map

| Path | Purpose |
|---|---|
| `main.py` | Command-line entry point |
| `experiments/runner.py` | Direct, CoT, multi-agent, and ablation execution |
| `experiments/evaluate.py` | Descriptive aggregation and participant-level comparisons |
| `src/utils.py` | Restricted local DAIC-WOZ input loading |
| `src/knowledge_base.py` | Compact PHQ-8 reference retrieval |
| `src/data_split.py` | Session-level split generation and validation |
| `artifact/reported_metrics.csv` | Manuscript-reported aggregate values |
| `artifact/result_index.json` | Machine-readable evidence and replication map |
| `tests/` | Offline unit tests for splitting and evaluation logic |

## Data access

DAIC-WOZ is not included. It must be requested from the
[official USC distribution page](https://dcapswoz.ict.usc.edu/) and used under
its access agreement. Participant identifiers, labels, transcripts, facial
features, and case-level rationales are intentionally excluded from this review
artifact.

After authorized access, provide the three fixed split-metadata files used by
the study and validate them locally:

```bash
python -m src.data_split \
  --development-csv /path/to/fixed_development.csv \
  --validation-csv /path/to/fixed_validation.csv \
  --test-csv /path/to/fixed_test.csv \
  --data-root /path/to/DAIC-WOZ \
  --require-files \
  --output-dir data/splits
```

The fixed 184-session assignment contains 128 development, 19 validation, and
37 held-out test sessions. The script does not reshuffle or repartition them: it
checks the archived SHA-256 fingerprints in `src/data_split.py`, verifies that
the partitions do not overlap, and writes a local manifest. The assignment
files are not distributed because they contain participant identifiers and
labels.

## Installation and offline checks

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m compileall -q main.py experiments src tests
```

Install Ollama separately and obtain the three study backbones:

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull deepseek-r1:8b
ollama pull nomic-embed-text
```

## Running the reported comparison design

Use the validated local test CSV containing `Participant_ID` and `PHQ8_Score`.
The examples below show Qwen2.5; replace `--backbone` for the other conditions.

```bash
python main.py \
  --data-root /path/to/DAIC-WOZ \
  --split-csv data/splits/test.csv \
  --split-name test \
  --mode direct \
  --backbone qwen2.5:7b \
  --input-modality transcript_au_summary \
  --output-dir results/qwen_direct

python main.py \
  --data-root /path/to/DAIC-WOZ \
  --split-csv data/splits/test.csv \
  --split-name test \
  --mode cot \
  --backbone qwen2.5:7b \
  --input-modality transcript_au_summary \
  --output-dir results/qwen_cot

python main.py \
  --data-root /path/to/DAIC-WOZ \
  --split-csv data/splits/test.csv \
  --split-name test \
  --mode multi_agent \
  --config full_pipeline \
  --backbone qwen2.5:7b \
  --input-modality transcript_au_summary \
  --output-dir results/qwen_multi
```

The default inference seeds are `11,22,33`, and the shared generative
temperature is `0.2`.

### Ablations

The archived ablation values and the implemented configurations use the
Qwen2.5 7B backbone:

| Configuration | Perception | Retrieved knowledge | Audit |
|---|:---:|:---:|:---:|
| `full_pipeline` | Yes | Yes | Yes |
| `without_perception` | No | Yes | Yes |
| `without_knowledge` | Yes | No | Yes |
| `without_audit` | Yes | Yes | No |
| `without_knowledge_audit` | Yes | No | No |

Select one with `--config CONFIGURATION` while using `--mode multi_agent`.

## Evaluation

```bash
python -m experiments.evaluate \
  --inputs results/qwen_direct results/qwen_cot results/qwen_multi \
  --output-dir results_summary
```

The evaluator refuses missing predictions, duplicate runs, or unequal
participant--seed panels. It reports mean MAE across seeds, bootstrap intervals
over participant-level mean absolute errors, and optional paired tests. For
each comparison, runs are first matched by participant and seed and then
averaged within participant. Model-generated audit flags and rationale-presence
checks are labeled as process proxies; they are not human or clinical
validation.

## Evidence and replication boundary

The result-to-command map is in
[`artifact/result_index.json`](artifact/result_index.json). A complete numerical
reproduction requires authorized DAIC-WOZ access, locally installed model
weights, and case-level reruns. The review snapshot provides code, configuration,
aggregate reported values, and offline tests, but no restricted participant data
or generated text derived from those data.

## Responsible-use notice

This software is for research evaluation only. PHQ-8 is a screening severity
measure, and model output must not be used as a diagnosis, treatment
recommendation, or autonomous triage decision. Local execution reduces
third-party transmission but does not by itself establish privacy, security,
regulatory, or institutional compliance.

## License

The anonymous review snapshot is provided under the MIT License. Public
attribution metadata will be restored after peer review.
