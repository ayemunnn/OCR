from typing import Any

from huggingface_hub import InferenceClient

from app.core.config import Settings, get_settings
from app.services.json_service import parse_json_response


def build_document_extraction_prompt() -> str:
    return (
        "You read OCR text from arbitrary documents: invoices, receipts, reports, "
        "emails, forms, academic papers, legal contracts, handwritten notes, etc.\n\n"
        "Your job is to quickly skim the text and extract the most important information "
        "in a generic structure that works for any document type.\n\n"
        "Return ONLY valid JSON (no markdown, no backticks, no extra commentary).\n"
        "Use the following schema:\n\n"
        "{\n"
        '  "summary": "Short 3-6 sentence overview of the document.",\n'
        '  "key_points": ["Bullet point 1", "Bullet point 2", ...],\n'
        "  \"entities\": {\n"
        '    "people": ["names..."],\n'
        '    "organizations": ["names..."],\n'
        '    "locations": ["locations..."],\n'
        '    "dates": ["dates mentioned..."],\n'
        '    "amounts": ["monetary or numeric amounts with context if possible"]\n'
        "  },\n"
        "  \"metadata\": {\n"
        '    "document_type": "Your best guess (e.g., invoice, email, report, letter, form, contract, unknown)",\n'
        '    "language": "Language of the document if you can infer it (e.g., en, fr, es)",\n'
        '    "confidence_notes": "Optional short note about how confident you are and any limitations."\n'
        "  }\n"
        "}\n\n"
        "If you cannot fill a field, use null or an empty list/empty string as appropriate."
    )


def analyze_document_text(
    ocr_text: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()

    if not settings.hf_api_key:
        return {
            "structured_output": None,
            "raw_output": None,
            "message": "LLM processing skipped because API key is not configured.",
        }

    messages = [
        {
            "role": "system",
            "content": build_document_extraction_prompt(),
        },
        {
            "role": "user",
            "content": f"Here is the OCR text from the document:\n\n{ocr_text}",
        },
    ]

    client = InferenceClient(
        model=settings.hf_model_name,
        token=settings.hf_api_key,
    )
    completion = client.chat_completion(
        model=settings.hf_model_name,
        messages=messages,
        temperature=0.15,
        max_tokens=1024,
    )

    choice = completion.choices[0].message
    if isinstance(choice, dict):
        raw_output = (choice.get("content") or "").strip()
    else:
        raw_output = (getattr(choice, "content", "") or "").strip()

    return {
        "structured_output": parse_json_response(raw_output),
        "raw_output": raw_output,
        "message": "LLM processing completed.",
    }
