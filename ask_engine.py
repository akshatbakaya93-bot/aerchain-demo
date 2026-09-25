"""Ask-panel engine: the model plans, Python computes, the model explains.

1. plan:    the model reads the question and picks analyses (or asks a clarifying question).
2. compute: Python runs those analyses on comparison.json - all arithmetic happens here.
3. explain: the model writes the answer from the computed results, with caveats.
"""

import json
import re

import anthropic

from extract_one import ask_for_json

LOW_CONFIDENCE = 0.6
HIGH_CONFIDENCE = 0.8

# code: (icon, label, severity, short label for column headers)
CAVEATS = {
    "conflict": ("⚠️", "Conflict between vendor documents", "danger", "Conflict"),
    "no_price": ("⛔", "Quoted without a price", "warn", "No price"),
    "unit_mismatch": ("📏", "Unit mismatch", "danger", "Unit"),
    "limited_stock": ("📦", "Limited stock", "warn", "Stock"),
    "tier": ("🏷️", "Price tier may not apply to required qty", "warn", "Tier"),
    "low_confidence": ("🔻", "Low confidence", "danger", "Low conf"),
    "fob": ("🚢", "FOB: freight, duty and import GST extra", "danger", "FOB"),
    "freight_extra": ("🚚", "Freight charged extra", "warn", "Freight extra"),
    "freight_included": ("🚚", "Freight included", "ok", "Freight incl."),
    "freight_unknown": ("🚚", "Freight terms not stated", "warn", "Freight ?"),
    "gst_unknown": ("🧾", "GST treatment unclear", "danger", "GST unclear"),
    "gst_inferred": ("🧾", "GST treatment inferred from other documents", "info", "GST inferred"),
    "gst_stripped": ("🧾", "Quoted GST-inclusive; 18% removed", "info", "GST incl.→removed"),
    "usd": ("💱", "Converted from USD", "info", "USD"),
    "overridden": ("↺", "Replaced an earlier price", "info", "Revised"),
    "alt_price": ("🏷️", "Other price tiers in the same document", "info", "Tiers"),
}
# Shown as icons inside a price cell; the rest are vendor-wide and shown in the column header.
LINE_CAVEATS = ("conflict", "unit_mismatch", "limited_stock", "tier")
VENDOR_CAVEATS = ("fob", "freight_extra", "freight_included", "freight_unknown", "gst_unknown", "gst_stripped", "usd")
EXCLUDABLE = ("fob", "gst_unknown", "conflict", "low_confidence", "limited_stock", "unit_mismatch", "freight_extra")

CLARIFY_OPTIONS = [
    "Lowest landed cost",
    "Best quality (questionnaire pass)",
    "Best warranty",
    "A balance of cost, quality and warranty",
]


# ---------------------------------------------------------------- caveats


def caveat(code: str, detail: str = "") -> dict:
    icon, label, severity, short = CAVEATS[code]
    return {"code": code, "icon": icon, "label": label, "severity": severity, "short": short, "detail": detail}


def is_priced(entry) -> bool:
    return isinstance(entry, dict) and entry.get("status") == "quoted" and entry.get("unit_price_inr_ex_gst") is not None


def freight_caveat(vendor: dict) -> dict:
    terms = " ".join(vendor.get("freight_terms") or [])
    t = terms.lower()
    if "fob" in t:
        return caveat("fob", terms)
    if "extra" in t or "ex-godown" in t:
        return caveat("freight_extra", terms)
    if "included" in t:
        return caveat("freight_included", terms)
    return caveat("freight_unknown", "No freight terms found in this vendor's documents.")


def vendor_caveats(vendor: dict, entries: list) -> list:
    """Vendor-wide caveats for a column header."""
    entries = [e for e in entries if isinstance(e, dict)]
    out = [freight_caveat(vendor)]
    gst = {e.get("gst_treatment") for e in entries}
    if "unknown" in gst or None in gst:
        out.append(caveat("gst_unknown", "Some prices may or may not include GST; used as quoted."))
    if "inclusive" in gst:
        out.append(caveat("gst_stripped", "Quoted GST-inclusive; 18% removed for comparison."))
    if "USD" in {e.get("original_currency") for e in entries}:
        out.append(caveat("usd", "Quoted in USD and converted at the assumed rate."))
    return out


def entry_caveats(entry: dict, vendor: dict) -> list:
    """Every caveat that applies to one vendor's price for one line."""
    out = []
    for f in entry.get("flags") or []:
        if f.startswith("CONFLICT"):
            out.append(caveat("conflict", f))
        elif f.startswith("UNIT MISMATCH"):
            out.append(caveat("unit_mismatch", f))
        elif f.startswith("LIMITED STOCK"):
            out.append(caveat("limited_stock", f))
        elif "tier may not apply" in f:
            out.append(caveat("tier", f))
        elif "GST treatment not stated" in f:
            out.append(caveat("gst_inferred", f))
        elif f.startswith("Converted from USD"):
            out.append(caveat("usd", f))
    if not is_priced(entry):
        out.append(caveat("no_price", entry.get("source_snippet") or ""))
    conf = entry.get("confidence")
    if isinstance(conf, (int, float)) and conf < LOW_CONFIDENCE:
        out.append(caveat("low_confidence", f"Confidence {conf:.2f} is below {LOW_CONFIDENCE}."))
    gst = entry.get("gst_treatment")
    if gst in ("unknown", None):
        out.append(caveat("gst_unknown", "Price used as quoted; it may or may not include GST."))
    elif gst == "inclusive":
        out.append(caveat("gst_stripped", "Quoted GST-inclusive; 18% removed for comparison."))
    out.append(freight_caveat(vendor))
    if entry.get("superseded"):
        out.append(caveat("overridden", "; ".join(
            f"{s['source_file']}: {s['unit_price']} ({s['reason']})" for s in entry["superseded"])))
    if entry.get("alternatives"):
        out.append(caveat("alt_price", "; ".join(
            f"{a['unit_price']} ({a['source_snippet']})" for a in entry["alternatives"])))
    return out


# ---------------------------------------------------------------- analyses (all arithmetic lives here)


def _eligible(data: dict, line_id: str, vendor: str, exclude: set):
    """Return (entry, caveats) if this vendor has a usable price for the line, else None."""
    entry = data["lines"][line_id]["vendors"].get(vendor)
    if not is_priced(entry):
        return None
    cavs = entry_caveats(entry, data["vendors"][vendor])
    if {c["code"] for c in cavs} & exclude:
        return None
    return entry, cavs


def _warning_labels(cavs: list) -> str:
    return "; ".join(sorted({c["label"] for c in cavs if c["severity"] in ("danger", "warn")}))


def cheapest_per_line(data: dict, vendors: list, lines: list, exclude: set) -> dict:
    rows, award, tied, no_quote = [], {}, [], []
    for lid in lines:
        line = data["lines"][lid]
        qty = line["required_qty"]
        cands = []
        for v in vendors:
            hit = _eligible(data, lid, v, exclude)
            if hit:
                cands.append((hit[0]["unit_price_inr_ex_gst"], v, hit[1]))
        cands.sort(key=lambda c: (c[0], c[1]))
        base = {"Line": lid, "Item": line["description"], "Qty": qty}
        if not cands:
            no_quote.append(lid)
            rows.append({**base, "Cheapest vendor": "No eligible quote", "Unit price (₹ ex-GST)": None,
                         "Line total (₹)": None, "Next best vendor": None, "Next best price (₹)": None,
                         "Gap to next (₹/unit)": None, "Quotes compared": 0, "Caveats": ""})
            continue
        best = cands[0][0]
        winners = [c for c in cands if c[0] == best]
        runner = next((c for c in cands if c[0] != best), None)
        line_total = round(best * qty, 2)
        rows.append({
            **base,
            "Cheapest vendor": " / ".join(w[1] for w in winners),
            "Unit price (₹ ex-GST)": best,
            "Line total (₹)": line_total,
            "Next best vendor": runner[1] if runner else None,
            "Next best price (₹)": runner[0] if runner else None,
            "Gap to next (₹/unit)": round(runner[0] - best, 2) if runner else None,
            "Quotes compared": len(cands),
            "Caveats": _warning_labels([c for w in winners for c in w[2]]),
        })
        if len(winners) > 1:
            tied.append((lid, line_total))
        else:
            a = award.setdefault(winners[0][1], {"lines": [], "value": 0.0})
            a["lines"].append(lid)
            a["value"] += line_total

    award_rows = [{"Vendor": v, "Lines awarded": len(a["lines"]), "Line IDs": ", ".join(a["lines"]),
                   "Award value (₹)": round(a["value"], 2)}
                  for v, a in sorted(award.items(), key=lambda kv: -kv[1]["value"])]
    if tied:
        award_rows.append({"Vendor": "Tied: buyer to choose", "Lines awarded": len(tied),
                           "Line IDs": ", ".join(t[0] for t in tied),
                           "Award value (₹)": round(sum(t[1] for t in tied), 2)})
    return {
        "rows": rows,
        "award_by_vendor": award_rows,
        "summary": {
            "grand_total_inr_ex_gst": round(sum(r["Line total (₹)"] or 0 for r in rows), 2),
            "lines_covered": len(lines) - len(no_quote),
            "lines_requested": len(lines),
            "lines_without_eligible_quote": no_quote,
            "tied_lines": [t[0] for t in tied],
            "vendors_awarded": len(award),
        },
    }


def vendor_totals(data: dict, vendors: list, lines: list, exclude: set) -> dict:
    eligible = {v: {lid for lid in lines if _eligible(data, lid, v, exclude)} for v in vendors}
    common = [lid for lid in lines if all(lid in eligible[v] for v in vendors)]
    rows = []
    for v in vendors:
        entries = {lid: data["lines"][lid]["vendors"].get(v) for lid in lines}
        priced = [lid for lid in lines if lid in eligible[v]]
        confs = [entries[lid]["confidence"] for lid in priced
                 if isinstance(entries[lid].get("confidence"), (int, float))]
        warn_lines = sum(
            1 for lid, e in entries.items() if isinstance(e, dict)
            and any(c["code"] in (*LINE_CAVEATS, "no_price", "low_confidence")
                    for c in entry_caveats(e, data["vendors"][v]))
        )
        total = lambda ids: round(sum(entries[i]["unit_price_inr_ex_gst"] * data["lines"][i]["required_qty"]
                                      for i in ids), 2)
        rows.append({
            "Vendor": v,
            "Lines priced": len(priced),
            "Quoted without price": sum(1 for e in entries.values() if isinstance(e, dict) and not is_priced(e)),
            "Not quoted": sum(1 for e in entries.values() if not isinstance(e, dict)),
            "Excluded by filters": sum(1 for lid, e in entries.items() if is_priced(e) and lid not in eligible[v]),
            "Total of priced lines (₹)": total(priced),
            "Total on common lines (₹)": total(common) if common else None,
            "Avg confidence": round(sum(confs) / len(confs), 2) if confs else None,
            "Lines with warnings": warn_lines,
            "Vendor caveats": "; ".join(c["label"] for c in vendor_caveats(data["vendors"][v], list(entries.values()))
                                        if c["severity"] in ("danger", "warn")),
        })
    if common:
        for rank, r in enumerate(sorted(rows, key=lambda r: r["Total on common lines (₹)"]), 1):
            r["Rank on common lines"] = rank
    rows.sort(key=lambda r: (r.get("Rank on common lines") or 99, -r["Lines priced"]))
    return {"rows": rows, "summary": {"common_lines": common, "common_line_count": len(common),
                                      "lines_requested": len(lines)}}


def price_table(data: dict, vendors: list, lines: list, exclude: set) -> dict:
    rows = []
    for lid in lines:
        line = data["lines"][lid]
        row = {"Line": lid, "Item": line["description"], "Qty": line["required_qty"]}
        prices = []
        for v in vendors:
            e = line["vendors"].get(v)
            if not isinstance(e, dict):
                row[v] = "not quoted"
            elif not is_priced(e):
                row[v] = "no price"
            elif not _eligible(data, lid, v, exclude):
                row[v] = "excluded"
            else:
                row[v] = e["unit_price_inr_ex_gst"]
                prices.append(e["unit_price_inr_ex_gst"])
        lo, hi = (min(prices), max(prices)) if prices else (None, None)
        row.update({"Min (₹)": lo, "Max (₹)": hi,
                    "Spread (₹)": round(hi - lo, 2) if prices else None,
                    "Spread %": round((hi - lo) / lo * 100, 1) if prices and lo else None})
        rows.append(row)
    return {"rows": rows, "summary": {}}


def vendor_notes(data: dict, vendors: list, lines: list, exclude: set) -> dict:
    rows = [{"Vendor": v, "Document": f["source_file"], "Notes": f["overall_notes"]}
            for v in vendors for f in data["vendors"][v]["files_in_precedence_order"]]
    return {"rows": rows, "summary": {}}


ANALYSES = {
    "cheapest_per_line": cheapest_per_line,
    "vendor_totals": vendor_totals,
    "price_table": price_table,
    "vendor_notes": vendor_notes,
}


def resolve_vendors(spec, data: dict, passers: list | None) -> tuple:
    all_vendors = list(data["vendors"])
    if spec == "questionnaire_passers":
        return list(passers or []), []
    if not spec:
        return all_vendors, []
    resolved, unknown = [], []
    for name in spec:
        n = str(name).lower().strip()
        match = next((v for v in all_vendors if v.lower() == n), None) or \
            next((v for v in all_vendors if n in v.lower() or v.lower() in n), None)
        (resolved if match else unknown).append(match or name)
    return list(dict.fromkeys(resolved)), unknown


def resolve_lines(spec, data: dict) -> tuple:
    all_lines = list(data["lines"])
    if not spec:
        return all_lines, []
    resolved, unknown = [], []
    for item in spec:
        s = str(item).strip()
        if s.upper() in data["lines"]:
            resolved.append(s.upper())
            continue
        cat = [lid for lid, l in data["lines"].items() if str(l["category"]).lower() == s.lower()]
        (resolved.extend(cat) if cat else unknown.append(s))
    return [l for l in all_lines if l in set(resolved)], unknown


def run_analysis(spec: dict, data: dict, passers: list | None) -> dict:
    kind = spec.get("type")
    if kind not in ANALYSES:
        return {"title": spec.get("title") or str(kind), "type": kind, "error": f"Unknown analysis '{kind}'."}
    vendors, bad_vendors = resolve_vendors(spec.get("vendors"), data, passers)
    lines, bad_lines = resolve_lines(spec.get("lines"), data)
    exclude = {c for c in (spec.get("exclude") or []) if c in EXCLUDABLE}
    result = {"title": spec.get("title") or kind.replace("_", " ").capitalize(), "type": kind,
              "vendors": vendors, "exclude": sorted(exclude)}
    if bad_vendors or bad_lines:
        result["ignored"] = {"vendors": bad_vendors, "lines": bad_lines}
    if not vendors:
        result["error"] = "No vendors matched (e.g. no vendor passed the questionnaire)."
        return result
    result.update(ANALYSES[kind](data, vendors, lines, exclude))
    return result


# ---------------------------------------------------------------- model steps

PLAN_INSTRUCTIONS = f"""You are the planning step of a procurement Q&A tool. Decide how to answer the buyer's question from the comparison data above. You do NOT write the answer and you do NOT do any arithmetic: Python runs the analyses you choose, then a separate step explains the results.

Return ONLY a JSON object with no other text and no markdown fences:
{{"action": "compute" | "clarify",
 "clarifying_question": string or null,
 "interpretation": string,
 "questionnaire_assessment": [{{"vendor": string, "passed": true | false | null, "met": [string], "failed_or_unanswered": [string], "evidence": string}}] or null,
 "analyses": [{{"type": string, "title": string, "vendors": [string] | "questionnaire_passers" | null, "lines": [string] | null, "exclude": [string]}}]}}

Analyses Python can run exactly:
- cheapest_per_line: per line, the lowest GST-exclusive INR unit price among the chosen vendors, line total (price x required qty), next-best vendor and gap, plus an award summary per vendor and the grand total. Use for "cheapest per line", "split the award", "who should supply what".
- vendor_totals: per vendor, lines priced / without price / not quoted, total value of priced lines, total on the lines every chosen vendor priced (like-for-like) with a rank, average confidence and warning counts. Use for "cheapest vendor overall", coverage, counts and rankings.
- price_table: every chosen vendor's price for the chosen lines, with min, max and spread. Use for "compare prices for X".
- vendor_notes: each vendor's document notes (questionnaire answers, warranty, payment, delivery, freight). Use for warranty, terms, certification or delivery questions.

Parameters: "vendors" is a list of exact vendor names from the data, "questionnaire_passers", or null for all. "lines" is a list of Line IDs or category names, or null for all. "exclude" lists caveat codes whose quotes are left out of the calculation: {", ".join(EXCLUDABLE)}. Only exclude when the buyer asks, or when a like-for-like basis is needed; otherwise include everything and let the explanation carry the caveats. You may request the same analysis twice (e.g. with and without FOB quotes) to show the effect of a caveat.

Clarify instead of guessing:
- If the question asks for the "best", "recommended", "right" or "top" vendor(s) without saying what matters, set action to "clarify", write a short clarifying_question asking whether they care most about lowest landed cost, best quality (questionnaire pass), best warranty, or a balance, and return no analyses.
- Don't clarify when the question names its criterion, is factual (cheapest, counts, who passed), or when the buyer has already clarified.

Questionnaire:
- Fill questionnaire_assessment whenever the question involves the questionnaire, quality or qualification, or you use "questionnaire_passers". Judge each vendor only from the notes in the data (vendors -> files_in_precedence_order -> overall_notes).
- If the buyer's RFx questionnaire is provided, judge against it: a requirement is met only if the notes show it; list unanswered ones under failed_or_unanswered, and set passed to null when key requirements are unanswered. Only passed=true counts as a passer.
- If no RFx is provided, use these criteria and say so in the interpretation: ISO 9001 certification, authorised reseller or distributor status, and warranty or after-sales support in Bangalore.

"interpretation": one sentence stating how you read the question and any criteria or exclusions you applied."""

EXPLAIN_INSTRUCTIONS = """You are the explanation step of a procurement Q&A tool. Python has already computed the results provided in the user message from the comparison data above. Write the answer to the buyer's question from those results.

Rules:
- Do no arithmetic. Every number you state (prices, totals, counts, gaps, ranks, percentages) must appear in the computed results. Don't add, subtract, average, round to new units or convert anything yourself. If a figure the buyer would want wasn't computed, say it isn't available rather than estimating it.
- Use only the computed results, the questionnaire assessment and the comparison data. No outside knowledge of prices, vendors or products.
- Name the Line IDs and vendors your answer relies on.
- Surface every caveat that affects the answer: the Caveats and Vendor caveats columns, FOB/import pricing (freight, duty and import GST not included), freight extra vs included, unclear or inferred GST, conflicts, missing prices, lines not quoted or without an eligible quote, unit mismatches, limited stock, price tiers and low confidence. Never present a vendor with FOB or unclear-GST pricing as cheapest without saying its landed cost may be higher.
- If a questionnaire assessment is included, say it is a judgment from vendor notes and name the criteria used.
- Be concise: lead with the direct answer, then the caveats that matter. The computed tables are shown to the buyer beside your answer, so summarise rather than repeating every row.

Return ONLY a JSON object with no other text and no markdown fences:
{"answer": string (markdown), "caveats": [string], "lines_used": [string], "vendors_used": [string]}"""


def _system(data: dict, instructions: str) -> list:
    # The data block comes first and is cached, so the plan and explain calls share it.
    return [
        {"type": "text",
         "text": f"<comparison_json>\n{json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n</comparison_json>",
         "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": instructions},
    ]


def _rfx_block(rfx: dict | None) -> str:
    if not rfx:
        return "No RFx questionnaire was provided by the buyer.\n\n"
    body = json.dumps({k: rfx.get(k) for k in ("questionnaire", "terms")}, ensure_ascii=False)
    return f"<buyer_rfx>\n{body}\n</buyer_rfx>\nThe questionnaire and terms the buyer sent to vendors.\n\n"


def _numbers_in(obj) -> list:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in _numbers_in(v)]
    if isinstance(obj, list):
        return [n for v in obj for n in _numbers_in(v)]
    return []


def unverified_numbers(answer: str, results: list, data: dict) -> list:
    """Figures (>= 100) in the answer that don't match any computed or source value."""
    # Prices quoted straight from the data are fine too; only derived figures must come from results.
    source_prices = [e.get(k) for l in data["lines"].values() for e in l["vendors"].values() if isinstance(e, dict)
                     for k in ("unit_price_inr_ex_gst", "unit_price_inr_as_quoted", "original_price")]
    known = _numbers_in(results) + _numbers_in(data["assumptions"]) + _numbers_in(source_prices) + \
        [float(l["required_qty"]) for l in data["lines"].values() if isinstance(l["required_qty"], (int, float))]
    missing = []
    for raw in re.findall(r"(?<![A-Za-z\d.])\d[\d,]*(?:\.\d+)?", answer):
        value = float(raw.replace(",", ""))
        if value < 100:
            continue
        if not any(abs(value - k) <= 1.0 for k in known):  # the answer must quote computed figures as-is
            missing.append(raw)
    return list(dict.fromkeys(missing))


def ask(client: anthropic.Anthropic, data: dict, question: str, rfx: dict | None, model: str,
        allow_clarify: bool = True) -> dict:
    """Answer a question. Returns {"type": "clarify", ...} or {"type": "answer", ...}."""
    plan = ask_for_json(client, _system(data, PLAN_INSTRUCTIONS),
                        f"{_rfx_block(rfx)}Question: {question}", model=model)

    if plan.get("action") == "clarify" and allow_clarify:
        return {"type": "clarify",
                "clarifying_question": plan.get("clarifying_question") or "What matters most for this decision?"}

    assessment = plan.get("questionnaire_assessment")
    passers = [a["vendor"] for a in assessment or [] if a.get("passed") is True]
    passers, _ = resolve_vendors(passers, data, None)
    results = [run_analysis(spec, data, passers) for spec in plan.get("analyses") or []]

    payload = {"question": question, "interpretation": plan.get("interpretation"),
               "questionnaire_assessment": assessment, "computed_results": results}
    explanation = ask_for_json(
        client, _system(data, EXPLAIN_INSTRUCTIONS),
        f"{_rfx_block(rfx)}<computed>\n{json.dumps(payload, ensure_ascii=False, default=str)}\n</computed>",
        model=model,
    )
    answer = explanation.get("answer") or ""
    return {
        "type": "answer",
        "interpretation": plan.get("interpretation"),
        "answer": answer,
        "caveats": explanation.get("caveats") or [],
        "lines_used": explanation.get("lines_used") or [],
        "vendors_used": explanation.get("vendors_used") or [],
        "assessment": assessment,
        "results": results,
        "unverified_numbers": unverified_numbers(answer, results, data),
    }
