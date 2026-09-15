"""Run matched direct, chain-of-thought, multi-agent, and ablation conditions."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from src.knowledge_base import KnowledgeRetriever
from src.utils import load_participant_data


PHQ_ITEMS = {
    "1_interest": "Anhedonia",
    "2_mood": "Depressed mood",
    "3_sleep": "Sleep disturbance",
    "4_energy": "Fatigue or low energy",
    "5_appetite": "Appetite change",
    "6_self_esteem": "Low self-esteem or worthlessness",
    "7_concentration": "Concentration difficulty",
    "8_movement": "Psychomotor agitation or retardation",
}


# These names intentionally match the aggregate ablation table in the paper.
ABLATION_CONFIGS = {
    "full_pipeline": {
        "use_perception": True,
        "use_knowledge": True,
        "use_retrieval": True,
        "use_audit": True,
    },
    "without_perception": {
        "use_perception": False,
        "use_knowledge": True,
        "use_retrieval": True,
        "use_audit": True,
    },
    "without_knowledge": {
        "use_perception": True,
        "use_knowledge": False,
        "use_retrieval": False,
        "use_audit": True,
    },
    "without_audit": {
        "use_perception": True,
        "use_knowledge": True,
        "use_retrieval": True,
        "use_audit": False,
    },
    "without_knowledge_audit": {
        "use_perception": True,
        "use_knowledge": False,
        "use_retrieval": False,
        "use_audit": False,
    },
}


@dataclass(frozen=True)
class ExperimentModels:
    """Matched-backbone condition used by every generative workflow role."""

    backbone: str
    temperature: float = 0.2


def create_llm(model: str, temperature: float, seed: int) -> ChatOllama:
    """Construct an Ollama chat model while supporting older wrappers."""

    from langchain_ollama import ChatOllama

    try:
        return ChatOllama(model=model, temperature=temperature, seed=seed)
    except TypeError:
        return ChatOllama(model=model, temperature=temperature)


def system_message(content: str) -> Any:
    """Import the inference dependency only when an LLM call is requested."""

    from langchain_core.messages import SystemMessage

    return SystemMessage(content=content)


def strip_reasoning_tags(content: str) -> str:
    return content.split("</think>")[-1].strip()


def parse_json_response(content: str) -> Dict[str, Any]:
    cleaned = strip_reasoning_tags(content)
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("The model response did not contain a JSON object.")
    return json.loads(cleaned[start:end])


def normalize_prediction(output: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp item scores and define the total as their sum."""

    individual_scores = output.get("individual_scores", {})
    rationale = output.get("rationale")
    normalized_scores: Dict[str, Dict[str, Any]] = {}
    total = 0

    for item_key in PHQ_ITEMS:
        item_output = individual_scores.get(item_key, {})
        raw_score = item_output.get("score", 0) if isinstance(item_output, dict) else 0
        try:
            score = int(round(float(raw_score)))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(3, score))
        total += score
        normalized_scores[item_key] = {
            "score": score,
            "evidence": item_output.get("evidence", "") if isinstance(item_output, dict) else "",
        }

    return {
        "individual_scores": normalized_scores,
        "total": total,
        "rationale": "" if rationale is None else str(rationale),
    }


def phq_scoring_schema() -> str:
    item_schema = ",\n".join(
        f'            "{item_key}": {{"score": 0, "evidence": "supporting input evidence"}}'
        for item_key in PHQ_ITEMS
    )
    return f"""{{
        "individual_scores": {{
{item_schema}
        }},
        "total": 0,
        "rationale": "brief input-grounded rationale"
    }}"""


def format_input(transcript_text: str, au_summary: str) -> str:
    evidence = f"Participant transcript:\n{transcript_text}"
    if au_summary:
        evidence += f"\n\nPrecomputed facial-action summaries:\n{au_summary}"
    return evidence


def invoke_json(prompt: str, llm: ChatOllama) -> Dict[str, Any]:
    response = llm.invoke([system_message(prompt)])
    return normalize_prediction(parse_json_response(response.content))


def run_direct_case(
    transcript_text: str,
    au_summary: str,
    llm: ChatOllama,
) -> Dict[str, Any]:
    prompt = f"""Estimate PHQ-8 symptom severity from the participant-derived input.

{format_input(transcript_text, au_summary)}

Score each PHQ-8 item from 0 to 3. Assign zero when the input does not support
an item. Return only valid JSON using this structure:
{phq_scoring_schema()}"""
    prediction = invoke_json(prompt, llm)
    return {
        "prediction": prediction,
        "initial_prediction": prediction,
        "audit": {},
        "knowledge_references": [],
        "observation": format_input(transcript_text, au_summary),
    }


def run_cot_case(
    transcript_text: str,
    au_summary: str,
    llm: ChatOllama,
) -> Dict[str, Any]:
    prompt = f"""Estimate PHQ-8 symptom severity from the participant-derived input.

{format_input(transcript_text, au_summary)}

Before assigning the total, assess each item separately and record the specific
input evidence supporting every non-zero score. Do not infer an unmentioned
symptom. Return only valid JSON using this structure:
{phq_scoring_schema()}"""
    prediction = invoke_json(prompt, llm)
    return {
        "prediction": prediction,
        "initial_prediction": prediction,
        "audit": {},
        "knowledge_references": [],
        "observation": format_input(transcript_text, au_summary),
    }


def run_perception_agent(transcript_text: str, au_summary: str, llm: ChatOllama) -> str:
    prompt = f"""Summarize only PHQ-8-relevant observations explicitly supported
by the participant-derived input. Separate transcript evidence from facial-action
summaries and avoid diagnostic conclusions.

{format_input(transcript_text, au_summary)}"""
    response = llm.invoke([system_message(prompt)])
    return strip_reasoning_tags(response.content)


def get_static_knowledge() -> str:
    return (
        "PHQ-8 item frequency scoring: 0 = not at all, 1 = several days, "
        "2 = more than half the days, and 3 = nearly every day. The eight "
        "items concern anhedonia, depressed mood, sleep, energy, appetite, "
        "self-evaluation, concentration, and psychomotor change."
    )


def run_reasoning_agent(observation: str, knowledge_context: str, llm: ChatOllama) -> Dict[str, Any]:
    prompt = f"""Estimate PHQ-8 symptom severity for research evaluation.

Observation summary:
{observation}

PHQ-8 criterion context:
{knowledge_context}

Score each item from 0 to 3 and cite input-grounded evidence for every non-zero
score. Do not treat the output as a diagnosis and do not infer unmentioned
symptoms. Return only valid JSON using this structure:
{phq_scoring_schema()}"""
    return invoke_json(prompt, llm)


def run_audit_agent(
    prediction: Dict[str, Any],
    observation: str,
    knowledge_context: str,
    llm: ChatOllama,
) -> Dict[str, Any]:
    prompt = f"""Audit whether this PHQ-8 estimate is supported by the supplied
observation and criterion context. This is a model-based consistency check, not
a clinician review.

Prediction:
{json.dumps(prediction)}

Observation:
{observation}

Criterion context:
{knowledge_context}

Return only valid JSON:
{{
  "decision": "PASS",
  "corrected_prediction": null,
  "reason": "brief evidence check",
  "unsupported_inference": false
}}

Use REJECT only for unsupported symptoms, fabricated evidence, or unjustified
severity. If rejected, return a corrected prediction with the original schema."""
    response = llm.invoke([system_message(prompt)])
    audit_output = parse_json_response(response.content)

    decision = str(audit_output.get("decision", "PASS")).upper()
    corrected = audit_output.get("corrected_prediction")
    if decision == "REJECT" and isinstance(corrected, dict):
        audit_output["corrected_prediction"] = normalize_prediction(corrected)
    else:
        audit_output["corrected_prediction"] = None

    audit_output["decision"] = decision
    audit_output["unsupported_inference"] = bool(
        audit_output.get("unsupported_inference", decision == "REJECT")
    )
    audit_output["reason"] = str(audit_output.get("reason", ""))
    return audit_output


def run_multi_agent_case(
    transcript_text: str,
    au_summary: str,
    config_name: str,
    models: ExperimentModels,
    seed: int,
) -> Dict[str, Any]:
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown configuration: {config_name}")

    config = ABLATION_CONFIGS[config_name]
    perception_llm = create_llm(models.backbone, models.temperature, seed)
    reasoning_llm = create_llm(models.backbone, models.temperature, seed)
    audit_llm = create_llm(models.backbone, models.temperature, seed)

    if config["use_perception"]:
        observation = run_perception_agent(transcript_text, au_summary, perception_llm)
    else:
        observation = format_input(transcript_text, au_summary)

    knowledge_references: List[dict] = []
    if config["use_knowledge"] and config["use_retrieval"]:
        retrieved_context, knowledge_references = KnowledgeRetriever(top_k=3).retrieve(observation)
        knowledge_context = f"{get_static_knowledge()}\nRetrieved item summaries:\n{retrieved_context}"
    elif config["use_knowledge"]:
        knowledge_context = get_static_knowledge()
    else:
        knowledge_context = "No separate PHQ-8 knowledge module is active."

    initial_prediction = run_reasoning_agent(observation, knowledge_context, reasoning_llm)
    final_prediction = initial_prediction
    audit_output: Dict[str, Any] = {}

    if config["use_audit"]:
        audit_output = run_audit_agent(initial_prediction, observation, knowledge_context, audit_llm)
        if audit_output.get("corrected_prediction"):
            final_prediction = audit_output["corrected_prediction"]

    return {
        "prediction": final_prediction,
        "initial_prediction": initial_prediction,
        "audit": audit_output,
        "knowledge_references": knowledge_references,
        "observation": observation,
    }


def resolve_columns(df: pd.DataFrame) -> tuple[str, str]:
    id_candidates = ["Participant_ID", "participant_id", "session_id", "Session_ID"]
    score_candidates = ["PHQ8_Score", "phq8_score", "PHQ_Score", "score"]
    id_column = next((column for column in id_candidates if column in df.columns), None)
    score_column = next((column for column in score_candidates if column in df.columns), None)
    if id_column is None or score_column is None:
        raise ValueError(
            "The split CSV must contain a participant/session ID and a PHQ-8 score. "
            f"Available columns: {list(df.columns)}"
        )
    return id_column, score_column


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


def run_experiment(args: argparse.Namespace) -> None:
    from tqdm import tqdm

    os.makedirs(args.output_dir, exist_ok=True)
    split_df = pd.read_csv(args.split_csv)
    id_column, score_column = resolve_columns(split_df)
    seeds = [int(seed) for seed in args.seeds.split(",")]
    models = ExperimentModels(backbone=args.backbone, temperature=args.temperature)

    jsonl_path = os.path.join(args.output_dir, "case_results.jsonl")
    csv_path = os.path.join(args.output_dir, "case_results.csv")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    flat_records: List[Dict[str, Any]] = []
    progress = tqdm(total=len(seeds) * len(split_df), desc=f"Running {args.mode}", unit="case")

    for seed in seeds:
        for _, row in split_df.iterrows():
            participant_id = int(row[id_column])
            true_score = int(row[score_column])
            try:
                transcript_text, au_summary = load_participant_data(
                    args.data_root,
                    participant_id,
                    include_au=args.input_modality == "transcript_au_summary",
                )
                if args.mode == "direct":
                    llm = create_llm(args.backbone, args.temperature, seed)
                    result = run_direct_case(transcript_text, au_summary, llm)
                    configuration = f"{args.backbone}_direct"
                elif args.mode == "cot":
                    llm = create_llm(args.backbone, args.temperature, seed)
                    result = run_cot_case(transcript_text, au_summary, llm)
                    configuration = f"{args.backbone}_cot"
                else:
                    result = run_multi_agent_case(
                        transcript_text,
                        au_summary,
                        args.config,
                        models,
                        seed,
                    )
                    configuration = f"{args.backbone}_{args.config}"

                prediction = result["prediction"]
                initial_prediction = result["initial_prediction"]
                audit = result.get("audit", {})
                knowledge_references = result.get("knowledge_references", [])
                record: Dict[str, Any] = {
                    "split": args.split_name,
                    "seed": seed,
                    "mode": args.mode,
                    "configuration": configuration,
                    "ablation": args.config if args.mode == "multi_agent" else "",
                    "backbone": args.backbone,
                    "temperature": args.temperature,
                    "input_modality": args.input_modality,
                    "participant_id": participant_id,
                    "true_score": true_score,
                    "predicted_score": prediction["total"],
                    "initial_predicted_score": initial_prediction["total"],
                    "audit_decision": audit.get("decision", ""),
                    "audit_corrected": bool(audit.get("corrected_prediction")),
                    "audit_unsupported_inference": audit.get("unsupported_inference"),
                    "audit_reason": audit.get("reason", ""),
                    "knowledge_reference_ids": [item["id"] for item in knowledge_references],
                    "knowledge_retrieval_method": (
                        knowledge_references[0].get("retrieval_method", "")
                        if knowledge_references
                        else ""
                    ),
                    "rationale": prediction.get("rationale", ""),
                    "prediction_json": prediction,
                    "error": "",
                }
            except Exception as error:
                record = {
                    "split": args.split_name,
                    "seed": seed,
                    "mode": args.mode,
                    "configuration": (
                        f"{args.backbone}_{args.config}"
                        if args.mode == "multi_agent"
                        else f"{args.backbone}_{args.mode}"
                    ),
                    "ablation": args.config if args.mode == "multi_agent" else "",
                    "backbone": args.backbone,
                    "temperature": args.temperature,
                    "input_modality": args.input_modality,
                    "participant_id": participant_id,
                    "true_score": true_score,
                    "predicted_score": None,
                    "initial_predicted_score": None,
                    "audit_decision": "ERROR",
                    "audit_corrected": None,
                    "audit_unsupported_inference": None,
                    "audit_reason": "",
                    "knowledge_reference_ids": [],
                    "knowledge_retrieval_method": "",
                    "rationale": "",
                    "prediction_json": {},
                    "error": str(error),
                }

            append_jsonl(jsonl_path, record)
            flat_record = dict(record)
            flat_record["knowledge_reference_ids"] = ";".join(record["knowledge_reference_ids"])
            flat_record["prediction_json"] = json.dumps(record["prediction_json"], ensure_ascii=True)
            flat_records.append(flat_record)
            progress.update(1)

    progress.close()
    pd.DataFrame(flat_records).to_csv(csv_path, index=False)
    print(f"Saved case-level results to {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run matched PHQ-8 direct, CoT, multi-agent, and ablation conditions."
    )
    parser.add_argument("--data-root", required=True, help="Authorized DAIC-WOZ directory.")
    parser.add_argument("--split-csv", required=True, help="Authorized local split CSV.")
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--mode", choices=["direct", "cot", "multi_agent"], default="multi_agent")
    parser.add_argument("--config", choices=list(ABLATION_CONFIGS), default="full_pipeline")
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--backbone", default="qwen2.5:7b")
    parser.add_argument(
        "--input-modality",
        choices=["text", "transcript_au_summary"],
        default="transcript_au_summary",
    )
    return parser


def main() -> None:
    run_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
