import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
import copy


RawMessage = Union[Dict[str, Any], List[Dict[str, Any]]]


def _read_message_file(path: str) -> List[RawMessage]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Message file not found at {path}")

    if file_path.suffix.lower() == ".jsonl":
        with file_path.open("r", encoding="utf-8") as fin:
            return [json.loads(line) for line in fin if line.strip()]

    with file_path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def _normalize_conversation(raw_entry: RawMessage) -> Dict[str, Any]:
    """
    Convert a raw entry into a standardized conversation dict with keys:
      - conversation: list of {"role": str, "content": str}
      - meta: optional metadata dictionary
      - reference_answer: optional string for downstream evaluation
      - id: optional identifier
    """
    entry_meta: Dict[str, Any] = {}
    reference_answer = None
    entry_id = None
    dataset_name = None
    source = None

    if isinstance(raw_entry, dict):
        entry_id = raw_entry.get("id")
        dataset_name = raw_entry.get("dataset_name")
        source = raw_entry.get("source")
        reference_answer = raw_entry.get("reference_answer")

        # Merge meta field if present
        if isinstance(raw_entry.get("meta"), dict):
            entry_meta.update(raw_entry["meta"])

        # Include well-known metadata keys if present at top level
        for key in ["task", "scene", "method", "language", "model_type"]:
            if key in raw_entry:
                entry_meta.setdefault(key, raw_entry[key])

        # Preserve any other top-level fields that aren't part of the
        # conversation specification.
        reserved = {
            "conversation",
            "messages",
            "meta",
            "reference_answer",
            "id",
            "dataset_name",
            "source",
        }
        for key, value in raw_entry.items():
            if key in reserved or key in {"role", "content"}:
                continue
            if key not in entry_meta:
                entry_meta[key] = value

        if "conversation" in raw_entry:
            conversation = copy.deepcopy(raw_entry["conversation"])
        elif "messages" in raw_entry:
            conversation = copy.deepcopy(raw_entry["messages"])
        else:
            # Assume the dict itself is a single message with role/content.
            conversation = [copy.deepcopy({k: raw_entry[k] for k in raw_entry if k in {"role", "content"}})]
    elif isinstance(raw_entry, Sequence) and not isinstance(raw_entry, (str, bytes)):
        conversation = copy.deepcopy(raw_entry)
    else:
        raise ValueError(f"Unsupported message entry type: {type(raw_entry)}")

    # Sanity check conversation structure
    for turn in conversation:
        if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
            raise ValueError(
                "Each conversation turn must be a dict with 'role' and 'content' keys."
            )

    return {
        "conversation": conversation,
        "meta": entry_meta,
        "reference_answer": reference_answer,
        "id": entry_id,
        "dataset_name": dataset_name,
        "source": source,
    }


def load_message_bank(
    *,
    default_messages: Sequence[RawMessage],
    override_path: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Load the default message list or override it from a JSON/JSONL file and
    normalize every entry into a standardized structure.
    """
    if override_path:
        raw_entries = _read_message_file(override_path)
    else:
        raw_entries = default_messages

    return [_normalize_conversation(entry) for entry in raw_entries]
