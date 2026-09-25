"""Extract a vendor quote from one file into structured JSON using Claude.

Usage: python extract_one.py <path-to-file>
Supported: .xlsx, .docx, .pdf, .txt (sent as text) and .png, .jpg, .jpeg (sent as images)
"""

import base64
import json
import sys
from pathlib import Path

import anthropic
import docx
import openpyxl
import pdfplumber
from dotenv import load_dotenv

MODEL = "claude-sonnet-5"

SCHEMA = """{
  "vendor_name": string,
  "quote_ref": string or null,
  "currency": string,              // the currency the vendor quoted in, e.g. "INR" or "USD"
  "gst_treatment": "inclusive" | "exclusive" | "unknown",
  "line_items": [
    {
      "description": string,        // exactly as the vendor wrote it
      "quantity": number or null,
      "unit": string or null,
      "unit_price": number or null,
      "confidence": number,         // 0 to 1, how sure you are about this line
      "source_snippet": string,     // the exact text you took this from
      "notes": string or null       // flag anything ambiguous; do NOT invent missing data
    }
  ],
  "overall_notes": string           // freight terms, validity, discounts, anything global
}"""

SYSTEM_PROMPT = f"""You extract vendor quotations from documents into structured JSON.

Return ONLY a single valid JSON object matching exactly this schema, with no other text, no markdown, and no code fences:

{SCHEMA}

Rules:
- If a value is missing or unclear, set it to null and lower that line's confidence. Never fabricate a price, quantity, or line item.
- Copy "description" and "source_snippet" verbatim from the document.
- Use "notes" to flag anything ambiguous (e.g. unclear units, prices that may or may not include tax, conflicting figures).
- Put quote-wide terms (freight, validity, payment terms, discounts, warranty) in "overall_notes".
- Output JSON only; comments from the schema above must not appear in your output."""


def read_xlsx(path: Path) -> str:
    wb = openpyxl.load_workbook(path, data_only=True)
    parts = []
    for ws in wb.worksheets:
        parts.append(f"=== Sheet: {ws.title} ===")
        for row in ws.iter_rows(values_only=True):
            if any(cell is not None for cell in row):
                parts.append("\t".join("" if c is None else str(c) for c in row))
    return "\n".join(parts)


def read_docx(path: Path) -> str:
    d = docx.Document(path)
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for i, table in enumerate(d.tables, 1):
        parts.append(f"=== Table {i} ===")
        for row in table.rows:
            parts.append("\t".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)


def read_pdf(path: Path) -> str:
    parts = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            parts.append(f"=== Page {i} ===")
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


READERS = {".xlsx": read_xlsx, ".docx": read_docx, ".pdf": read_pdf, ".txt": read_txt}

IMAGE_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # API limit per base64 image


def text_content(path: Path) -> list:
    text = READERS[path.suffix.lower()](path)
    if not text.strip():
        raise ValueError(f"No text could be extracted from {path}")
    return [
        {"type": "text", "text": f"File name: {path.name}\n\n<document>\n{text}\n</document>"}
    ]


def image_content(path: Path) -> list:
    data = path.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image is {len(data) / 1e6:.1f} MB; the API limit is 5 MB. Resize it and retry.")
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": IMAGE_MEDIA_TYPES[path.suffix.lower()],
                "data": base64.standard_b64encode(data).decode("ascii"),
            },
        },
        {
            "type": "text",
            "text": f"File name: {path.name}\n\nThe vendor quote is in the image above. "
            "Read the values directly off the image.\n\n"
            "This may be a photo taken at an angle, so rows and columns can appear visually "
            "offset. Before pairing an item with a price, check whether the item column and the "
            "price column are shifted by a row, and align them by reading each row across. If "
            "you can't confidently align a value to a row, lower its confidence and note it.",
        },
    ]


def build_content(path: Path) -> list:
    """Route a file to the text or image path based on its extension."""
    ext = path.suffix.lower()
    if ext in READERS:
        return text_content(path)
    if ext in IMAGE_MEDIA_TYPES:
        return image_content(path)
    supported = ", ".join([*READERS, *IMAGE_MEDIA_TYPES])
    raise ValueError(f"Unsupported file type '{path.suffix}'. Supported: {supported}")


def extract_quote(client: anthropic.Anthropic, content: list, model: str = MODEL) -> dict:
    return ask_for_json(client, SYSTEM_PROMPT, content, model=model)


# Models whose safety classifiers can decline a request; the API re-runs a declined
# request on Anthropic's recommended fallback model instead of returning the refusal.
SERVER_FALLBACK_MODELS = {"claude-opus-5"}


def ask_for_json(client: anthropic.Anthropic, system: str | list, content: list | str,
                 model: str = MODEL) -> dict:
    """Send one request and parse the reply as JSON."""
    params = dict(model=model, max_tokens=64000, system=system,
                  messages=[{"role": "user", "content": content}])
    if model in SERVER_FALLBACK_MODELS:
        stream_ctx = client.beta.messages.stream(**params, betas=["server-side-fallback-2026-07-01"],
                                                 fallbacks="default")
    else:
        stream_ctx = client.messages.stream(**params)
    with stream_ctx as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to answer this request.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Response was truncated (hit max_tokens).")

    raw = "".join(b.text for b in response.content if b.type == "text").strip()
    # Tolerate an accidental ```json fence despite the instructions.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model did not return valid JSON: {e}\n--- raw output ---\n{raw}")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python extract_one.py <path-to-file>")

    path = Path(sys.argv[1])
    if not path.is_file():
        sys.exit(f"File not found: {path}")

    try:
        content = build_content(path)
    except ValueError as e:
        sys.exit(str(e))

    load_dotenv(Path(__file__).parent / ".env")
    client = anthropic.Anthropic()
    try:
        result = extract_quote(client, content)
    except anthropic.AuthenticationError:
        sys.exit("Authentication failed - check ANTHROPIC_API_KEY in .env")
    except anthropic.RateLimitError:
        sys.exit("Rate limited by the API - wait a moment and retry.")
    except anthropic.APIStatusError as e:
        sys.exit(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        sys.exit("Could not connect to the Anthropic API.")
    except RuntimeError as e:
        sys.exit(str(e))

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
