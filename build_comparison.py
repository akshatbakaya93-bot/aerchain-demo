"""Build a line-by-line vendor comparison against the buyer's RFQ.

Usage: python build_comparison.py
Reads extracted.json and dataset/00_Buyer_RFQ_Master.xlsx, writes comparison.json.

Steps:
  1. Group extracted files by normalized vendor name.
  2. Map every vendor line to a canonical RFQ Line ID with Claude (one call per file).
  3. Merge each vendor's files, applying supersession (correction > revised > later-dated).
  4. Normalize prices to GST-exclusive INR and flag unit mismatches and freight terms.
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

import anthropic
import openpyxl
from dotenv import load_dotenv

from extract_one import MODEL, ask_for_json

ROOT = Path(__file__).parent
EXTRACTED_PATH = ROOT / "extracted.json"
RFQ_PATH = ROOT / "dataset" / "00_Buyer_RFQ_Master.xlsx"
OUTPUT_PATH = ROOT / "comparison.json"

USD_TO_INR = 83.5
GST_RATE = 0.18
MIN_MAPPING_CONFIDENCE = 0.6  # mappings below this are treated as unmatched
MAX_WORKERS = 4  # files mapped in parallel

NOT_QUOTED = "not quoted"


# ---------------------------------------------------------------- buyer RFQ


def load_rfq_lines(path: Path) -> dict:
    """Return {line_id: {category, description, required_qty, unit}} from the RFQ master."""
    ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    header = None
    lines = {}
    for row in ws.iter_rows(values_only=True):
        cells = ["" if c is None else str(c).strip() for c in row]
        if header is None:
            if "Line ID" in cells:
                header = {name: i for i, name in enumerate(cells)}
            continue
        line_id = cells[header["Line ID"]]
        if not line_id:
            continue
        lines[line_id] = {
            "category": row[header["Category"]],
            "description": row[header["Item Description"]],
            "required_qty": row[header["Required Qty"]],
            "unit": row[header["Unit"]],
        }
    if header is None:
        raise ValueError(f"No 'Line ID' header row found in {path.name}")
    return lines


# ---------------------------------------------------------------- vendors


VENDOR_ABBREVIATIONS = {"b'lore": "bangalore", "blore": "bangalore", "blr": "bangalore"}
VENDOR_SUFFIXES = {"pvt", "private", "ltd", "limited", "llp", "inc", "co", "corp", "pte"}


def vendor_key(record: dict) -> str:
    """Case-insensitive vendor key ignoring legal suffixes and parentheticals."""
    name = record.get("vendor_name") or ""
    if not name.strip():
        # Fall back to the vendor token in the filename, e.g. "07_Nexus_Quote_USD.xlsx".
        parts = record["source_file"].split("_")
        name = parts[1] if len(parts) > 1 else record["source_file"]
    s = re.sub(r"\(.*?\)", " ", name.lower())  # drop "(Singapore)" etc.
    tokens = [VENDOR_ABBREVIATIONS.get(t, t).replace("'", "") for t in re.findall(r"[a-z0-9']+", s)]
    return " ".join(t for t in tokens if t and t not in VENDOR_SUFFIXES)


# ---------------------------------------------------------------- supersession

DATE_PATTERN = re.compile(r"\bdate[d]?\s*[:\-]?\s*(\d{1,2}[-/ ][A-Za-z]{3}[-/ ]\d{4})", re.I)


def file_date(record: dict) -> date | None:
    """Best-effort quote date. The extraction schema has no date field, so this only
    finds dates written as 'Date: 12-Mar-2026' in overall_notes or quote_ref."""
    text = f"{record.get('overall_notes') or ''} {record.get('quote_ref') or ''}"
    m = DATE_PATTERN.search(text)
    if not m:
        return None
    try:
        return datetime.strptime(re.sub(r"[/ ]", "-", m.group(1)), "%d-%b-%Y").date()
    except ValueError:
        return None


def file_meta(record: dict, order: int) -> dict:
    name = record["source_file"].lower()
    notes = (record.get("overall_notes") or "").lower()
    ref = (record.get("quote_ref") or "").lower()
    return {
        "source_file": record["source_file"],
        "order": order,
        # Match specific phrases: a rate card saying "corrections may exist overleaf" is not one.
        "is_correction": "correction" in name or bool(re.search(r"correction sheet|handwritten correction", notes)),
        "is_revised": "revised" in name or "supersedes" in notes or bool(re.search(r"-r\d+$", ref)),
        "date": file_date(record),
    }


def precedence(meta: dict) -> tuple:
    """Ascending: later entries override earlier ones on overlapping lines."""
    return (meta["is_correction"], meta["is_revised"], meta["date"] or date.min, meta["order"])


def override_reason(winner: dict, loser: dict) -> str | None:
    if winner["is_correction"] and not loser["is_correction"]:
        return "handwritten correction overrides earlier document"
    if winner["is_revised"] and not loser["is_revised"]:
        return "revised quote supersedes earlier version"
    if winner["date"] and loser["date"] and winner["date"] > loser["date"]:
        return f"later-dated document ({winner['date']} vs {loser['date']})"
    return None  # no rule applies


# ---------------------------------------------------------------- mapping

MAPPING_SYSTEM = """You match vendor quote lines to a buyer's canonical RFQ line items.

Return ONLY a JSON object, with no other text, no markdown and no code fences:
{"mappings": [{"index": integer, "line_id": string or null, "confidence": number, "applies_to_required_qty": true | false | null, "reason": string}]}

Return exactly one entry per vendor line, using that line's index.

Rules:
- Match on product identity. Vendors may write a full name, a partial name (e.g. "Seagate 2TB"), or only a manufacturer model/part code (e.g. "MZ-V7S1T0BW" is the Samsung 970 EVO Plus 1TB). Use your knowledge of manufacturer part numbers to decode codes.
- It must be the same product: a different capacity, generation, wattage, screen size or model is NOT a match.
- If a vendor line is a bundle covering more than one canonical line, or matches no canonical line, set line_id to null.
- If you are not confident, set line_id to null and give a low confidence. Never force a match.
- "confidence" (0 to 1) is how sure you are that the vendor line is the same product as the canonical line.
- "applies_to_required_qty": if the vendor line is one of several quantity-based price tiers, true if this tier applies to the buyer's required quantity, false if not. null when there is no quantity condition.
- "reason": one short sentence, e.g. how you decoded a part number or why you left it unmatched."""


def map_file_lines(client: anthropic.Anthropic, record: dict, rfq_lines: dict) -> list:
    """Return one mapping dict per line item in the record, in the same order."""
    items = record["line_items"]
    canonical = "\n".join(
        f"{lid}\t{l['category']}\t{l['description']}\t{l['required_qty']} {l['unit']}"
        for lid, l in rfq_lines.items()
    )
    vendor_lines = "\n".join(
        json.dumps(
            {
                "index": i,
                "description": li.get("description"),
                "unit": li.get("unit"),
                "notes": li.get("notes"),
                "source_snippet": li.get("source_snippet"),
            },
            ensure_ascii=False,
        )
        for i, li in enumerate(items)
    )
    content = [
        {
            "type": "text",
            "text": f"<canonical_lines>\nLine ID\tCategory\tDescription\tRequired qty\n{canonical}\n"
            f"</canonical_lines>\n\n<vendor_lines vendor=\"{record.get('vendor_name')}\">\n"
            f"{vendor_lines}\n</vendor_lines>",
        }
    ]
    result = ask_for_json(client, MAPPING_SYSTEM, content)

    by_index = {m["index"]: m for m in result.get("mappings", []) if isinstance(m.get("index"), int)}
    mappings = []
    for i in range(len(items)):
        m = by_index.get(i)
        if m is None:
            mappings.append({"line_id": None, "confidence": 0.0, "applies_to_required_qty": None,
                             "reason": "model returned no mapping for this line"})
            continue
        line_id = m.get("line_id")
        conf = float(m.get("confidence") or 0)
        reason = m.get("reason") or ""
        if line_id is not None and line_id not in rfq_lines:
            reason = f"model returned unknown Line ID {line_id!r}. {reason}"
            line_id = None
        elif line_id is not None and conf < MIN_MAPPING_CONFIDENCE:
            reason = f"best guess {line_id} is below the {MIN_MAPPING_CONFIDENCE} threshold. {reason}"
            line_id = None
        mappings.append({"line_id": line_id, "confidence": conf,
                         "applies_to_required_qty": m.get("applies_to_required_qty"), "reason": reason})
    return mappings


# ---------------------------------------------------------------- normalization

SINGLE_UNIT_WORDS = {"unit", "units", "each", "ea", "pc", "pcs", "piece", "pieces", "no", "nos",
                     "stick", "sticks", "tube", "tubes", "combo", "card"}
MULTI_UNIT_WORDS = {"kit", "pair", "box", "pack", "set", "lot", "bundle"}


def unit_class(unit: str | None) -> str | None:
    """Collapse unit wording so 'pc', 'each' and 'single stick' compare equal."""
    if not unit:
        return None
    words = re.findall(r"[a-z]+", unit.lower())
    for w in words:
        if w in MULTI_UNIT_WORDS:
            return w
    if any(w in SINGLE_UNIT_WORDS for w in words):
        return "each"
    return unit.lower().strip()


def normalize_price(price, currency: str | None, gst: str | None) -> tuple[dict, list]:
    flags = []
    currency = (currency or "").upper()
    out = {"original_price": price, "original_currency": currency or None,
           "gst_treatment": gst, "price_inr_as_quoted": None, "price_inr_ex_gst": None}
    if price is None:
        return out, flags

    if currency == "INR":
        inr = float(price)
    elif currency == "USD":
        inr = float(price) * USD_TO_INR
        flags.append(f"Converted from USD {price} at {USD_TO_INR} INR/USD.")
    else:
        flags.append(f"Unknown currency {currency or '(none)'}; not converted.")
        return out, flags

    if gst == "inclusive":
        ex = inr / (1 + GST_RATE)
        flags.append(f"Quoted GST-inclusive; {GST_RATE:.0%} GST stripped for comparison.")
    elif gst == "exclusive":
        ex = inr
    else:
        ex = inr
        flags.append("GST treatment unknown; price used as quoted, so it may or may not include GST.")

    out["price_inr_as_quoted"] = round(inr, 2)
    out["price_inr_ex_gst"] = round(ex, 2)
    return out, flags


FREIGHT_PATTERN = re.compile(r"freight|fob|ex-godown|ex godown|delivered|shipping|duty", re.I)


def freight_sentences(notes: str | None) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", notes or "") if FREIGHT_PATTERN.search(s)]


STOCK_HINT = re.compile(r"stock|only\s+\d+\s*(?:pcs?|pieces|units)|\d+\s*(?:pcs?|pieces|units)\s+only", re.I)
STOCK_QTY = re.compile(r"only\s+(\d+)|(\d+)\s*(?:pcs?|pieces|units|nos)?\s*(?:only|in stock)", re.I)


def stock_limit(item: dict) -> tuple[bool, int | None]:
    """Detect a vendor stock limit such as 'only 6 in stock' or '(2 pcs only)'."""
    text = f"{item.get('notes') or ''} {item.get('source_snippet') or ''}"
    if not STOCK_HINT.search(text):
        return False, None
    m = STOCK_QTY.search(text)
    if m:
        return True, int(m.group(1) or m.group(2))
    return True, item.get("quantity") if isinstance(item.get("quantity"), int) else None


def inherit(file_value: str | None, vendor_values: set, unknown: set, label: str) -> tuple:
    """Fill an unknown per-file value from the vendor's other files if they agree."""
    if file_value and file_value not in unknown:
        return file_value, None
    known = {v for v in vendor_values if v and v not in unknown}
    if len(known) == 1:
        value = next(iter(known))
        return value, f"{label} not stated in this file; inherited '{value}' from the vendor's other files."
    return file_value, None


# ---------------------------------------------------------------- merge


def pick_within_file(entries: list) -> tuple:
    """When one file has several lines for the same Line ID (e.g. price tiers),
    prefer the one that applies to the buyer's required quantity."""
    rank = {True: 0, None: 1, False: 2}
    ranked = sorted(entries, key=lambda e: rank.get(e["mapping"]["applies_to_required_qty"], 1))
    return ranked[0], ranked[1:]


def build_vendor(key: str, records: list, mappings: dict, rfq_lines: dict) -> tuple:
    """Return (display_name, vendor_summary, {line_id: entry}, [unmatched lines])."""
    metas = {r["source_file"]: file_meta(r, i) for i, r in enumerate(records)}
    ordered = sorted(records, key=lambda r: precedence(metas[r["source_file"]]))

    display_name = max(records, key=lambda r: len(r.get("line_items") or []))["vendor_name"] or key
    vendor_gst = {r.get("gst_treatment") for r in records}
    vendor_currency = {r.get("currency") for r in records}
    freight = [s for r in records for s in freight_sentences(r.get("overall_notes"))]
    freight_flag = f"Freight/delivery terms: {' '.join(freight)}" if freight else None

    # Collect candidate lines per Line ID, per file.
    candidates: dict[str, dict[str, list]] = {}
    unmatched = []
    for r in ordered:
        for item, mapping in zip(r.get("line_items") or [], mappings.get(r["source_file"], [])):
            entry = {"record": r, "item": item, "mapping": mapping}
            if mapping["line_id"] is None:
                unmatched.append({
                    "vendor": display_name,
                    "source_file": r["source_file"],
                    "vendor_description": item.get("description"),
                    "unit_price": item.get("unit_price"),
                    "currency": r.get("currency"),
                    "extraction_confidence": item.get("confidence"),
                    "mapping_confidence": mapping["confidence"],
                    "reason": mapping["reason"],
                    "source_snippet": item.get("source_snippet"),
                })
                continue
            candidates.setdefault(mapping["line_id"], {}).setdefault(r["source_file"], []).append(entry)

    supersession_log = []
    lines = {}
    for line_id, per_file in candidates.items():
        # Files in ascending precedence; each file contributes its best entry.
        chosen = []
        alternatives = []
        for r in ordered:
            if r["source_file"] in per_file:
                primary, alts = pick_within_file(per_file[r["source_file"]])
                chosen.append(primary)
                alternatives += alts

        priced = [e for e in chosen if e["item"].get("unit_price") is not None]
        winner = priced[-1] if priced else chosen[-1]
        w_meta = metas[winner["record"]["source_file"]]
        flags = []

        superseded = []
        for e in chosen:
            if e is winner:
                continue
            if (e["item"].get("unit_price") == winner["item"].get("unit_price")
                    and e["record"].get("currency") == winner["record"].get("currency")):
                continue  # same price in both files, so nothing was actually overridden
            e_meta = metas[e["record"]["source_file"]]
            if precedence(e_meta) < precedence(w_meta):
                reason = override_reason(w_meta, e_meta)
                if reason is None:
                    reason = ("no supersession rule applies; used the later file in dataset order "
                              f"({w_meta['source_file']})")
                    flags.append(f"CONFLICT: {e_meta['source_file']} and {w_meta['source_file']} both "
                                 "quote this line and neither clearly supersedes the other. Verify with vendor.")
            else:
                reason = f"this file mentions the line without a price; kept the price from {w_meta['source_file']}"
            superseded.append({
                "source_file": e_meta["source_file"],
                "unit_price": e["item"].get("unit_price"),
                "currency": e["record"].get("currency"),
                "reason": reason,
            })
            supersession_log.append(f"{line_id}: {e_meta['source_file']} ({e['item'].get('unit_price')}) "
                                    f"-> {w_meta['source_file']} ({winner['item'].get('unit_price')}): {reason}")

        record, item, mapping = winner["record"], winner["item"], winner["mapping"]
        gst, gst_note = inherit(record.get("gst_treatment"), vendor_gst, {"unknown", None}, "GST treatment")
        currency, cur_note = inherit(record.get("currency"), vendor_currency, {"unknown", None}, "Currency")
        prices, price_flags = normalize_price(item.get("unit_price"), currency, gst)
        flags += [n for n in (gst_note, cur_note) if n] + price_flags

        if item.get("unit_price") is None:
            flags.append("Line is mentioned but no price is stated.")
        buyer_unit = rfq_lines[line_id]["unit"]
        if unit_class(item.get("unit")) and unit_class(item.get("unit")) != unit_class(buyer_unit):
            flags.append(f"UNIT MISMATCH: vendor quotes per '{item.get('unit')}', buyer needs "
                         f"'{buyer_unit}'. Not converted.")
        required_qty = rfq_lines[line_id]["required_qty"]
        limited, available = stock_limit(item)
        if limited and (available is None or not isinstance(required_qty, (int, float))):
            flags.append(f"LIMITED STOCK: vendor notes limited availability; buyer needs {required_qty}.")
        elif limited and available < required_qty:
            flags.append(f"LIMITED STOCK: vendor has only {available} available; buyer needs {required_qty}.")
        elif mapping["applies_to_required_qty"] is False:
            flags.append("Chosen price tier may not apply to the buyer's required quantity.")
        if item.get("notes"):
            flags.append(f"Vendor line note: {item['notes']}")
        if freight_flag:
            flags.append(freight_flag)

        extraction_conf = item.get("confidence")
        confs = [c for c in (extraction_conf, mapping["confidence"]) if isinstance(c, (int, float))]
        lines[line_id] = {
            "status": "quoted" if item.get("unit_price") is not None else "quoted_without_price",
            "unit_price_inr_ex_gst": prices["price_inr_ex_gst"],
            "unit_price_inr_as_quoted": prices["price_inr_as_quoted"],
            "original_price": prices["original_price"],
            "original_currency": prices["original_currency"],
            "gst_treatment": prices["gst_treatment"],
            "confidence": min(confs) if confs else None,
            "extraction_confidence": extraction_conf,
            "mapping_confidence": mapping["confidence"],
            "mapping_reason": mapping["reason"],
            "vendor_description": item.get("description"),
            "vendor_unit": item.get("unit"),
            "source_file": record["source_file"],
            "source_snippet": item.get("source_snippet"),
            "flags": flags,
            "superseded": superseded,
            "alternatives": [
                {"vendor_description": a["item"].get("description"), "unit_price": a["item"].get("unit_price"),
                 "source_file": a["record"]["source_file"], "source_snippet": a["item"].get("source_snippet"),
                 "applies_to_required_qty": a["mapping"]["applies_to_required_qty"]}
                for a in alternatives
            ],
        }

    summary = {
        "vendor_key": key,
        "names_seen": sorted({r.get("vendor_name") for r in records if r.get("vendor_name")}),
        "files_in_precedence_order": [
            {"source_file": metas[r["source_file"]]["source_file"],
             "is_correction": metas[r["source_file"]]["is_correction"],
             "is_revised": metas[r["source_file"]]["is_revised"],
             "date": str(metas[r["source_file"]]["date"]) if metas[r["source_file"]]["date"] else None,
             "line_items": len(r.get("line_items") or []),
             "overall_notes": r.get("overall_notes")}
            for r in ordered
        ],
        "freight_terms": freight,
        "supersession_log": supersession_log,
    }
    return display_name, summary, lines, unmatched


# ---------------------------------------------------------------- main


def main() -> None:
    if not EXTRACTED_PATH.exists():
        sys.exit(f"{EXTRACTED_PATH.name} not found - run extract_all.py first.")
    records = json.loads(EXTRACTED_PATH.read_text())
    rfq_lines = load_rfq_lines(RFQ_PATH)

    errors = [{"source_file": r["source_file"], "stage": "extraction", "error": r.get("error")}
              for r in records if r.get("status") != "ok"]
    records = [r for r in records if r.get("status") == "ok"]

    groups: dict[str, list] = {}
    for r in records:
        groups.setdefault(vendor_key(r), []).append(r)

    print(f"Loaded {len(rfq_lines)} RFQ lines and {len(records)} extracted files "
          f"from {len(groups)} vendors:")
    for key, recs in groups.items():
        print(f"  {key:<28} {', '.join(r['source_file'] for r in recs)}")

    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic()
    to_map = [r for r in records if r.get("line_items")]
    print(f"\nMapping line items to RFQ Line IDs with {MODEL} ({len(to_map)} files) ...")

    def map_one(r: dict):
        try:
            result = map_file_lines(client, r, rfq_lines)
            print(f"  [done] {r['source_file']}", flush=True)
            return r["source_file"], result, None
        except Exception as e:
            msg = (f"API error {e.status_code}: {e.message}" if isinstance(e, anthropic.APIStatusError)
                   else f"{type(e).__name__}: {e}")
            print(f"  [FAILED] {r['source_file']}: {msg.splitlines()[0]}", flush=True)
            return r["source_file"], None, msg

    mappings = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for source_file, result, error in pool.map(map_one, to_map):
            if error:
                errors.append({"source_file": source_file, "stage": "mapping", "error": error})
            else:
                mappings[source_file] = result

    # A file whose mapping failed contributes no lines; drop it so it can't look "not quoted" silently.
    failed = {e["source_file"] for e in errors}
    vendors, vendor_lines, unmatched = {}, {}, []
    for key, recs in groups.items():
        recs = [r for r in recs if r["source_file"] not in failed]
        if not recs:
            continue
        name, summary, lines, unm = build_vendor(key, recs, mappings, rfq_lines)
        vendors[name] = summary
        vendor_lines[name] = lines
        unmatched += unm

    comparison = {}
    for line_id, rfq in rfq_lines.items():
        comparison[line_id] = {
            "category": rfq["category"],
            "description": rfq["description"],
            "required_qty": rfq["required_qty"],
            "unit": rfq["unit"],
            "vendors": {name: vendor_lines[name].get(line_id, NOT_QUOTED) for name in vendors},
        }

    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "assumptions": {
            "usd_to_inr": USD_TO_INR,
            "gst_rate": GST_RATE,
            "min_mapping_confidence": MIN_MAPPING_CONFIDENCE,
            "precedence": "handwritten correction > revised > later-dated > dataset file order",
        },
        "lines": comparison,
        "vendors": vendors,
        "unmatched_vendor_lines": unmatched,
        "errors": errors,
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))

    print(f"\n  {'vendor':<32} {'quoted':>7} {'no price':>9} {'unmatched':>10} {'overrides':>10} {'conflicts':>10}")
    for name, lines in vendor_lines.items():
        quoted = sum(1 for l in lines.values() if l["status"] == "quoted")
        no_price = sum(1 for l in lines.values() if l["status"] == "quoted_without_price")
        unm = sum(1 for u in unmatched if u["vendor"] == name)
        overrides = len(vendors[name]["supersession_log"])
        conflicts = sum(1 for l in lines.values() for f in l["flags"] if f.startswith("CONFLICT"))
        print(f"  {name[:32]:<32} {quoted:>4}/{len(rfq_lines):<2} {no_price:>9} {unm:>10} {overrides:>10} {conflicts:>10}")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  {e['source_file']} ({e['stage']}): {(e['error'] or '').splitlines()[0]}")
    print(f"\nSaved {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
