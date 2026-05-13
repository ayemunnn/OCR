import json
from typing import Any


def clean_json_response(raw_output: str) -> str:
    output = (raw_output or "").strip()

    if output.startswith("```"):
        output = output.strip("`")
        if output.lower().startswith("json"):
            output = output[4:].strip()

    start = output.find("{")
    end = output.rfind("}")
    if start != -1 and end != -1 and end > start:
        return output[start : end + 1]

    return output


def parse_json_response(raw_output: str) -> dict[str, Any]:
    json_candidate = clean_json_response(raw_output)
    parsed = json.loads(json_candidate)

    if not isinstance(parsed, dict):
        raise ValueError("Expected the model response to contain a JSON object.")

    return parsed
