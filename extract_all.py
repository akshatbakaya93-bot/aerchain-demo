"""Run the vendor-quote extraction over every file in dataset/ and save to extracted.json.

Usage: python extract_all.py
"""

import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from extract_one import build_content, extract_quote

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "dataset"
OUTPUT_PATH = ROOT / "extracted.json"
MAX_WORKERS = 4  # files processed in parallel


def should_skip(path: Path) -> bool:
    name = path.name
    return (
        name.startswith("00_Buyer_RFQ_Master")
        or "_REFERENCE" in name
        or name.startswith(".")
        or not path.is_file()
    )


def process_file(client: anthropic.Anthropic, path: Path) -> dict:
    """Extract one file. Never raises: failures are returned as an error record."""
    try:
        result = extract_quote(client, build_content(path))
        return {"source_file": path.name, "status": "ok", **result}
    except Exception as e:
        if isinstance(e, anthropic.APIStatusError):
            msg = f"API error {e.status_code}: {e.message}"
        else:
            msg = f"{type(e).__name__}: {e}"
        return {
            "source_file": path.name,
            "status": "error",
            "error": msg,
            "traceback": traceback.format_exc(),
        }


def summarize(record: dict) -> str:
    name = record["source_file"]
    if record["status"] != "ok":
        first_line = record["error"].splitlines()[0]
        return f"  FAILED  {name:<45} {first_line}"
    items = record.get("line_items") or []
    confs = [li["confidence"] for li in items if isinstance(li.get("confidence"), (int, float))]
    avg = f"{sum(confs) / len(confs):.2f}" if confs else "  - "
    vendor = record.get("vendor_name") or "?"
    return f"  ok      {name:<45} {vendor[:30]:<30} {len(items):>4} {avg:>9}"


def main() -> None:
    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic()

    files = sorted(p for p in DATASET_DIR.iterdir() if not should_skip(p))
    print(f"Extracting {len(files)} files from {DATASET_DIR.name}/ ...\n")

    def run(path: Path) -> dict:
        record = process_file(client, path)
        status = "done" if record["status"] == "ok" else "FAILED"
        print(f"  [{status}] {path.name}", flush=True)
        return record

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(pool.map(run, files))  # keeps the sorted file order

    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\n  {'status':<7} {'file':<45} {'vendor':<30} {'items':>4} {'avg conf':>9}")
    for record in results:
        print(summarize(record))

    failed = [r for r in results if r["status"] != "ok"]
    print(f"\n{len(results) - len(failed)} succeeded, {len(failed)} failed.")
    print(f"Saved {len(results)} records to {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
