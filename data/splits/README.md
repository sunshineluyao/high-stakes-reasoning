# Local split outputs

Validate and copy the study's fixed development, validation, and test metadata
locally with `python -m src.data_split`. The command preserves the archived
assignments and verifies their SHA-256 fingerprints; it never reshuffles them.
Outputs are excluded from version control because they contain restricted
DAIC-WOZ participant identifiers and labels.
