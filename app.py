"""RFx and vendor quote comparison app.

Run: streamlit run app.py
Reads comparison.json (built by build_comparison.py). ANTHROPIC_API_KEY comes from .env.
"""

import json
import os
import re
import tempfile
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import ask_engine
from ask_engine import (CAVEATS, CLARIFY_OPTIONS, HIGH_CONFIDENCE, LINE_CAVEATS, LOW_CONFIDENCE, entry_caveats,
                        is_priced, vendor_caveats)
from extract_one import IMAGE_MEDIA_TYPES, READERS, ask_for_json, build_content, extract_quote

SONNET_MODEL = "claude-sonnet-5"  # RFx drafting and document extraction
OPUS_MODEL = "claude-opus-5"  # Ask panel (planning and explaining answers)

ROOT = Path(__file__).parent
COMPARISON_PATH = ROOT / "comparison.json"
load_dotenv(ROOT / ".env")

STAGES = ["1 · Create RFx", "2 · Collect responses", "3 · Compare & Ask"]
SEVERITY_COLOR = {"danger": "red", "warn": "orange", "info": "blue", "ok": "green"}

st.set_page_config(page_title="RFx & Quote Comparison", layout="wide")


# ---------------------------------------------------------------- helpers


@st.cache_data
def load_comparison(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


def md_escape(text) -> str:
    return re.sub(r"([\\`*_{}\[\]<>#|~$])", r"\\\1", str(text))


def conf_dot(conf) -> str:
    if not isinstance(conf, (int, float)):
        return "⚪"
    return "🟢" if conf >= HIGH_CONFIDENCE else "🟡" if conf >= LOW_CONFIDENCE else "🔴"


def money(x) -> str:
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return "" if x is None else str(x)
    return f"₹{x:,.0f}" if float(x).is_integer() else f"₹{x:,.2f}"


def show_table(rows: list) -> None:
    if not rows:
        st.caption("No rows.")
        return
    df = pd.DataFrame(rows)
    money_cols = [c for c in df.columns if "₹" in c]
    st.dataframe(df.style.format(money, subset=money_cols), hide_index=True, use_container_width=True)


def require_api_key() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    st.error("ANTHROPIC_API_KEY not found. Add it to .env and restart the app.")
    return False


def run_with_errors(fn, *args, **kwargs):
    """Call the API-backed function, showing errors in the UI. Returns None on failure."""
    try:
        return fn(*args, **kwargs)
    except anthropic.AuthenticationError:
        st.error("Authentication failed. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.APIStatusError as e:
        st.error(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        st.error("Could not connect to the Anthropic API.")
    except RuntimeError as e:
        st.error(str(e))
    return None


def goto(stage: int) -> None:
    # Applied at the top of the next run, before the stage selector is drawn.
    st.session_state["goto_stage"] = STAGES[stage]


# ---------------------------------------------------------------- stage 1: create RFx

RFX_INSTRUCTIONS = """You turn a buyer's plain-English purchasing need into a structured RFx (request for quotation) to send to vendors.

Return ONLY a JSON object with no other text and no markdown fences:
{"title": string,
 "line_items": [{"description": string, "quantity": number or null, "unit": string, "notes": string or null}],
 "questionnaire": [{"question": string, "category": string, "source": "stated" | "inferred"}],
 "terms": {"delivery_location": string or null, "payment_terms": string or null, "quality_terms": [string], "other_terms": [string]}}

Rules:
- One line item per distinct product. Describe it using only what the buyer said; don't pick a brand or model they didn't name.
- If the buyer gives no quantity for an item, set quantity to null and say so in notes. Never invent quantities.
- Questionnaire: qualifying questions vendors must answer. Include every requirement the buyer stated (source "stated"), and add the standard qualifying questions a procurement team would ask for this purchase, such as certifications, authorised-reseller status, lead time, warranty, after-sales support and acceptance of the payment terms (source "inferred"). Phrase each as a question to the vendor.
- Terms: record only the payment, delivery and quality terms the buyer stated. Don't add terms they didn't state; turn suggestions into questionnaire items instead."""

RFX_EXAMPLE = ("I need 20 i5 CPUs, 15 Ryzen 5, 30 Samsung 1TB SSDs, plus monitors and keyboards, delivered to "
               "Bangalore, 30-day payment terms, vendors must be ISO 9001 certified")


def lines_to_list(text: str) -> list:
    return [t.strip() for t in text.splitlines() if t.strip()]


def create_rfx_stage() -> None:
    st.header("Create RFx")
    st.caption("Describe what you need in plain English. A structured RFx is drafted for you to review and edit.")

    need = st.text_area("What do you need?", value=st.session_state.get("rfx_need", RFX_EXAMPLE), height=110)
    if st.button("Generate RFx", type="primary", disabled=not need.strip()) and require_api_key():
        with st.spinner("Drafting the RFx..."):
            rfx = run_with_errors(ask_for_json, anthropic.Anthropic(), RFX_INSTRUCTIONS, need.strip(),
                                  model=SONNET_MODEL)
        if rfx is not None:
            st.session_state["rfx_need"] = need
            st.session_state["rfx_draft"] = rfx
            st.session_state["rfx_version"] = st.session_state.get("rfx_version", 0) + 1
            st.session_state.pop("rfx_sent", None)

    rfx = st.session_state.get("rfx_draft")
    if not rfx:
        return

    v = st.session_state["rfx_version"]  # new widget keys per draft, so edits reset on regenerate
    st.subheader("Review and edit")
    title = st.text_input("Title", value=rfx.get("title") or "", key=f"title_{v}")

    st.markdown("**Line items**")
    items = pd.DataFrame(rfx.get("line_items") or [], columns=["description", "quantity", "unit", "notes"])
    items = st.data_editor(
        items, key=f"items_{v}", num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={
            "description": st.column_config.TextColumn("Description", required=True, width="large"),
            "quantity": st.column_config.NumberColumn("Quantity", min_value=0, step=1),
            "unit": st.column_config.TextColumn("Unit"),
            "notes": st.column_config.TextColumn("Notes", width="medium"),
        },
    )
    missing_qty = items["description"].notna() & items["quantity"].isna()
    if missing_qty.any():
        st.warning("No quantity yet for: " + ", ".join(items.loc[missing_qty, "description"].astype(str))
                   + ". Fill these in before sending.")

    st.markdown("**Vendor questionnaire**")
    questions = pd.DataFrame(rfx.get("questionnaire") or [], columns=["question", "category", "source"])
    questions = st.data_editor(
        questions, key=f"questions_{v}", num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={
            "question": st.column_config.TextColumn("Question", required=True, width="large"),
            "category": st.column_config.TextColumn("Category"),
            "source": st.column_config.SelectboxColumn("Source", options=["stated", "inferred"],
                                                       help="stated = from your request; inferred = suggested"),
        },
    )

    st.markdown("**Terms**")
    terms = rfx.get("terms") or {}
    c1, c2 = st.columns(2)
    delivery = c1.text_input("Delivery location", value=terms.get("delivery_location") or "", key=f"deliv_{v}")
    payment = c2.text_input("Payment terms", value=terms.get("payment_terms") or "", key=f"pay_{v}")
    quality = c1.text_area("Quality terms (one per line)", value="\n".join(terms.get("quality_terms") or []),
                           key=f"qual_{v}")
    other = c2.text_area("Other terms (one per line)", value="\n".join(terms.get("other_terms") or []),
                         key=f"other_{v}")

    clean = lambda df: [
        {k: (None if pd.isna(val) else val) for k, val in row.items()}
        for row in df.dropna(how="all").to_dict("records")
    ]
    edited = {
        "title": title,
        "line_items": clean(items),
        "questionnaire": clean(questions),
        "terms": {"delivery_location": delivery or None, "payment_terms": payment or None,
                  "quality_terms": lines_to_list(quality), "other_terms": lines_to_list(other)},
    }
    # Streamlit drops widget state when the buyer switches stage; keep the edits to restore on return.
    st.session_state["rfx_edited"] = edited

    if st.button("Send to vendors →", type="primary"):
        st.session_state["rfx_sent"] = edited
        goto(1)
        st.rerun()


# ---------------------------------------------------------------- stage 2: collect responses

UPLOAD_TYPES = [ext.lstrip(".") for ext in [*READERS, *IMAGE_MEDIA_TYPES]]


def extract_upload(name: str, data: bytes) -> dict:
    """Run the real extraction pipeline on an uploaded file, keeping its original filename."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(name).name
        path.write_bytes(data)
        return extract_quote(anthropic.Anthropic(), build_content(path), model=SONNET_MODEL)


def show_extraction(result: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vendor", result.get("vendor_name") or "?")
    c2.metric("Quote ref", result.get("quote_ref") or "none")
    c3.metric("Currency", result.get("currency") or "?")
    c4.metric("GST", result.get("gst_treatment") or "?")
    items = result.get("line_items") or []
    if items:
        st.dataframe(
            pd.DataFrame(items), hide_index=True, use_container_width=True,
            column_config={"confidence": st.column_config.ProgressColumn(
                "confidence", min_value=0, max_value=1, format="%.2f")},
        )
        low = [i for i in items if isinstance(i.get("confidence"), (int, float)) and i["confidence"] < LOW_CONFIDENCE]
        if low:
            st.warning(f"{len(low)} line(s) below {LOW_CONFIDENCE} confidence. Check them against the source.")
    else:
        st.info("No priced line items in this file (e.g. a questionnaire or terms document).")
    st.markdown(f"**Overall notes:** {md_escape(result.get('overall_notes') or '-')}")
    with st.expander("Raw JSON"):
        st.json(result)


def collect_stage(data: dict) -> None:
    st.header("Collect responses")
    rfx = st.session_state.get("rfx_sent")
    if rfx:
        st.success(f"RFx “{rfx['title']}” sent: {len(rfx['line_items'])} line items, "
                   f"{len(rfx['questionnaire'])} questionnaire questions.")
    else:
        st.info("No RFx sent from Stage 1 in this session. The responses below answer the existing "
                "RFQ in dataset/00_Buyer_RFQ_Master.xlsx.")

    vendors = data["vendors"]
    n_files = sum(len(v["files_in_precedence_order"]) for v in vendors.values())
    st.subheader(f"Responses received: {len(vendors)} vendors, {n_files} documents")
    st.dataframe(
        pd.DataFrame([
            {"Vendor": name, "Document": f["source_file"], "Priced lines": f["line_items"],
             "Revised": f["is_revised"], "Correction": f["is_correction"]}
            for name, v in vendors.items() for f in v["files_in_precedence_order"]
        ]),
        hide_index=True, use_container_width=True,
    )
    st.caption(f"Already extracted and compared (generated {data['generated_at']}).")

    st.subheader("Extract a vendor file live")
    st.caption(f"Upload any vendor file ({', '.join(UPLOAD_TYPES)}). It goes through the same extraction "
               "as the pipeline. The result is shown here only and isn't added to the comparison.")
    upload = st.file_uploader("Vendor file", type=UPLOAD_TYPES)
    if upload is not None:
        cache = st.session_state.setdefault("live_extractions", {})
        key = f"{upload.name}:{upload.size}"
        if st.button(f"Run extraction on {upload.name}", type="primary") and require_api_key():
            with st.spinner(f"Extracting {upload.name}..."):
                try:
                    cache[key] = extract_upload(upload.name, upload.getvalue())
                except ValueError as e:  # unreadable or unsupported file
                    st.error(str(e))
                except (anthropic.APIError, RuntimeError) as e:
                    st.error(f"Extraction failed: {e}")
        if key in cache:
            show_extraction(cache[key])

    st.divider()
    st.button("Continue to Compare & Ask →", type="primary", on_click=goto, args=(2,))


# ---------------------------------------------------------------- stage 3: comparison grid


def cell_popover(entry: dict, vendor_name: str, vendor: dict, line_id: str, line: dict) -> None:
    st.markdown(f"**{md_escape(vendor_name)}** · {line_id} {md_escape(line['description'])}")
    if is_priced(entry):
        st.markdown(f"#### {money(entry['unit_price_inr_ex_gst'])} per {md_escape(line['unit'])}, GST-exclusive")
        st.caption(f"As quoted: {entry.get('original_price')} {entry.get('original_currency') or ''}, "
                   f"GST {entry.get('gst_treatment')}")
    else:
        st.markdown(":orange[⛔ **Quoted without a price**]")

    conf = entry.get("confidence")
    conf_txt = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
    st.markdown(f"{conf_dot(conf)} Confidence **{conf_txt}**")
    st.caption(f"Extraction {entry.get('extraction_confidence')} · mapping {entry.get('mapping_confidence')}: "
               f"{entry.get('mapping_reason') or ''}")

    st.markdown(f"**Source:** `{entry.get('source_file')}`")
    snippet = (entry.get("source_snippet") or "").strip() or "(none)"
    st.markdown("\n".join(f"> {md_escape(l)}" for l in snippet.splitlines()))
    st.caption(f"Vendor wrote: {entry.get('vendor_description')}")

    cavs = entry_caveats(entry, vendor)
    if cavs:
        st.markdown("**Flags**")
        for c in sorted(cavs, key=lambda c: ["danger", "warn", "info", "ok"].index(c["severity"])):
            color = SEVERITY_COLOR[c["severity"]]
            detail = f"  \n{md_escape(c['detail'])}" if c["detail"] else ""
            st.markdown(f":{color}[{c['icon']} **{c['label']}**]{detail}")


def price_cell(col, entry, vendor_name: str, vendor: dict, line_id: str, line: dict, is_lowest: bool) -> None:
    if not isinstance(entry, dict):
        col.markdown(":gray[∅ not quoted]")
        return
    cavs = entry_caveats(entry, vendor)
    icons = "".join(c["icon"] for c in cavs if c["code"] in LINE_CAVEATS)
    if is_priced(entry):
        label = f"{'⬇ ' if is_lowest else ''}{money(entry['unit_price_inr_ex_gst'])} {conf_dot(entry.get('confidence'))}"
    else:
        label = "⛔ No price"
    if icons:
        label += f" {icons}"
    with col.popover(label, use_container_width=True):
        cell_popover(entry, vendor_name, vendor, line_id, line)


def line_has_warning(line: dict, vendors: list, data: dict) -> bool:
    for v in vendors:
        e = line["vendors"].get(v)
        if not isinstance(e, dict):
            continue
        if any(c["code"] in (*LINE_CAVEATS, "no_price", "low_confidence")
               for c in entry_caveats(e, data["vendors"][v])):
            return True
    return False


def flag_counts(data: dict, vendors: list) -> dict:
    counts = {"Conflicts": 0, "No price": 0, "Not quoted": 0, "Unit mismatches": 0, "Limited stock": 0,
              "Low confidence": 0}
    code_to_metric = {"conflict": "Conflicts", "no_price": "No price", "unit_mismatch": "Unit mismatches",
                      "limited_stock": "Limited stock", "low_confidence": "Low confidence"}
    for line in data["lines"].values():
        for v in vendors:
            e = line["vendors"].get(v)
            if not isinstance(e, dict):
                counts["Not quoted"] += 1
                continue
            for code in {c["code"] for c in entry_caveats(e, data["vendors"][v])}:
                if code in code_to_metric:
                    counts[code_to_metric[code]] += 1
    return counts


def legend() -> None:
    with st.expander("How to read this table"):
        st.markdown(
            "- Prices are **INR per unit, GST-exclusive**. Click a price to see its source, confidence and flags.\n"
            f"- Confidence: 🟢 ≥ {HIGH_CONFIDENCE} · 🟡 ≥ {LOW_CONFIDENCE} · 🔴 below {LOW_CONFIDENCE}\n"
            "- **⬇** marks the lowest listed price on the line. It does not account for freight, FOB import "
            "costs or unclear GST (see the column headers).\n"
            "- In a cell: " + " · ".join(f"{CAVEATS[c][0]} {CAVEATS[c][3]}" for c in LINE_CAVEATS)
            + " · ⛔ quoted without price · ∅ not quoted\n"
            "- Under a vendor name (applies to all their prices): "
            + " · ".join(f"{CAVEATS[c][0]} {CAVEATS[c][3]}" for c in ("fob", "freight_extra", "gst_unknown", "usd"))
        )


def comparison_view(data: dict) -> None:
    all_vendors = list(data["vendors"])
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    vendors = c1.multiselect("Vendors", all_vendors, default=all_vendors)
    only_warnings = c2.toggle("Only lines with warnings", value=False,
                              help="Conflicts, unit mismatches, limited stock, tiers, low confidence or no price.")
    if not vendors:
        st.info("Select at least one vendor.")
        return

    counts = flag_counts(data, vendors)
    counts["Unmatched lines"] = sum(1 for u in data["unmatched_vendor_lines"] if u["vendor"] in vendors)
    for col, (label, n) in zip(st.columns(len(counts)), counts.items()):
        col.metric(label, n)
    legend()

    widths = [0.5, 2.0, 0.7] + [1.3] * len(vendors)
    head = st.columns(widths)
    head[0].markdown("**Line**")
    head[1].markdown("**Item**")
    head[2].markdown("**Qty**")
    for col, v in zip(head[3:], vendors):
        entries = [l["vendors"].get(v) for l in data["lines"].values()]
        tags = " ".join(f":{SEVERITY_COLOR[c['severity']]}[{c['icon']} {c['short']}]"
                        for c in vendor_caveats(data["vendors"][v], entries))
        col.markdown(f"**{md_escape(v)}**  \n{tags}")
    st.divider()

    for line_id, line in data["lines"].items():
        if only_warnings and not line_has_warning(line, vendors, data):
            continue
        prices = [line["vendors"][v]["unit_price_inr_ex_gst"] for v in vendors if is_priced(line["vendors"].get(v))]
        lowest = min(prices) if prices else None
        row = st.columns(widths, vertical_alignment="center")
        row[0].markdown(f"**{line_id}**")
        row[1].markdown(f"{md_escape(line['description'])}  \n:gray[{md_escape(line['category'])}]")
        row[2].markdown(f"{line['required_qty']} {md_escape(line['unit'])}")
        for col, v in zip(row[3:], vendors):
            e = line["vendors"].get(v, "not quoted")
            price_cell(col, e, v, data["vendors"][v], line_id, line,
                       is_lowest=is_priced(e) and e["unit_price_inr_ex_gst"] == lowest)

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Unmatched vendor lines")
        st.caption("Vendor lines that couldn't be confidently matched to a buyer line. They're not in the table.")
        unmatched = [u for u in data["unmatched_vendor_lines"] if u["vendor"] in vendors]
        if unmatched:
            st.dataframe(pd.DataFrame(unmatched), hide_index=True, use_container_width=True)
        else:
            st.write("None.")
    with right:
        st.subheader("Assumptions")
        a = data["assumptions"]
        st.markdown(
            f"- **Currency:** USD converted to INR at **{a['usd_to_inr']}**\n"
            f"- **Price basis:** INR per unit, **GST-exclusive**. {a['gst_rate']:.0%} removed from "
            "GST-inclusive quotes; prices with unclear GST are used as quoted\n"
            f"- **Matching:** vendor lines below {a['min_mapping_confidence']} confidence are left unmatched\n"
            f"- **Revisions:** {a['precedence']}\n"
            "- **Not adjusted:** freight, FOB import costs and unit mismatches are flagged, not converted\n"
            f"- Generated {data['generated_at']}"
        )
        if data.get("errors"):
            st.error("Pipeline errors:\n\n" + "\n".join(
                f"- {e['source_file']} ({e['stage']}): {e['error']}" for e in data["errors"]))


# ---------------------------------------------------------------- stage 3: ask

SUGGESTIONS = [
    "Cheapest per line",
    "Who passed the questionnaire?",
    "Split award among questionnaire-passers",
    "Best value considering caveats",
]


def set_question(q: str) -> None:
    st.session_state["question"] = q


def queue_question() -> None:
    q = (st.session_state.get("question") or "").strip()
    if q:
        st.session_state["ask_queue"] = (q, True)


def choose_clarification(option: str) -> None:
    pending = st.session_state.pop("pending_clarification")
    question = f"{pending['question']}\n\nThe buyer clarified what matters most: {option}."
    st.session_state["ask_queue"] = (question, False)
    st.session_state["ask_display_question"] = f"{pending['question']} ({option})"


def render_answer(question: str, r: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**Q: {md_escape(question)}**")
        if r.get("interpretation"):
            st.caption(f"Interpreted as: {r['interpretation']}")

        text_col, table_col = st.columns([2, 3])
        with text_col:
            st.markdown(r.get("answer") or "_No answer returned._")
            for c in r.get("caveats") or []:
                st.warning(c, icon="⚠️")
            if r.get("unverified_numbers"):
                st.caption("Figures in the text not found in the computed results: "
                           + ", ".join(r["unverified_numbers"]) + ". Check them against the tables.")
            used = []
            if r.get("lines_used"):
                used.append("Lines: " + ", ".join(r["lines_used"]))
            if r.get("vendors_used"):
                used.append("Vendors: " + ", ".join(r["vendors_used"]))
            if used:
                st.caption(" · ".join(used))
        with table_col:
            for res in r.get("results") or []:
                st.markdown(f"**{md_escape(res['title'])}**")
                notes = []
                if res.get("exclude"):
                    notes.append("excluding " + ", ".join(CAVEATS[c][3] for c in res["exclude"]))
                if res.get("vendors"):
                    notes.append(f"{len(res['vendors'])} vendor(s)")
                if notes:
                    st.caption("Computed: " + "; ".join(notes))
                if res.get("error"):
                    st.warning(res["error"])
                    continue
                show_table(res.get("rows") or [])
                if res.get("award_by_vendor"):
                    st.markdown("Award by vendor")
                    show_table(res["award_by_vendor"])
                summary = res.get("summary") or {}
                if "grand_total_inr_ex_gst" in summary:
                    st.metric("Total (GST-exclusive)", money(summary["grand_total_inr_ex_gst"]),
                              help=f"{summary['lines_covered']} of {summary['lines_requested']} lines covered")
            if r.get("assessment"):
                st.markdown("**Questionnaire assessment**")
                st.caption("A judgment from the vendors' notes, not a calculation.")
                show_table([{"Vendor": a.get("vendor"),
                             "Passed": {True: "✅ yes", False: "❌ no"}.get(a.get("passed"), "❔ unclear"),
                             "Met": "; ".join(a.get("met") or []),
                             "Failed or unanswered": "; ".join(a.get("failed_or_unanswered") or []),
                             "Evidence": a.get("evidence")} for a in r["assessment"]])


def ask_view(data: dict) -> None:
    if not require_api_key():
        return
    st.caption("Ask about the comparison in your own words. Numbers are calculated from the comparison data; "
               "the answer explains them and calls out caveats."
               + (" Questionnaire questions use the RFx you sent in Stage 1." if st.session_state.get("rfx_sent")
                  else ""))

    st.markdown("**Try asking…**")
    cols = st.columns(len(SUGGESTIONS))
    for col, q in zip(cols, SUGGESTIONS):
        col.button(q, on_click=set_question, args=(q,), use_container_width=True, key=f"chip_{q}")

    question = st.text_area("Your question", key="question", height=80,
                            placeholder="e.g. Which vendor is cheapest for SSDs, excluding FOB quotes?")
    st.button("Ask", type="primary", disabled=not question.strip(), on_click=queue_question)

    pending = st.session_state.get("pending_clarification")
    if pending:
        with st.container(border=True):
            st.markdown(f"**Q: {md_escape(pending['question'])}**")
            st.info(pending["clarifying_question"], icon="❓")
            opt_cols = st.columns(len(CLARIFY_OPTIONS))
            for col, option in zip(opt_cols, CLARIFY_OPTIONS):
                col.button(option, on_click=choose_clarification, args=(option,), use_container_width=True,
                           key=f"clarify_{option}")

    queued = st.session_state.pop("ask_queue", None)
    if queued:
        q, allow_clarify = queued
        shown = st.session_state.pop("ask_display_question", q)
        st.session_state.pop("pending_clarification", None)
        with st.spinner("Working out the answer..."):
            result = run_with_errors(ask_engine.ask, anthropic.Anthropic(), data, q,
                                     st.session_state.get("rfx_sent"), OPUS_MODEL, allow_clarify=allow_clarify)
        if result is not None:
            if result["type"] == "clarify":
                st.session_state["pending_clarification"] = {"question": q, **result}
            else:
                st.session_state.setdefault("history", []).insert(0, (shown, result))
            st.rerun()

    for q, r in st.session_state.get("history", []):
        render_answer(q, r)


# ---------------------------------------------------------------- main


def compare_stage(data: dict) -> None:
    st.header("Compare & Ask")
    st.caption("Every vendor's price for each buyer line, normalised to GST-exclusive INR, with flags. "
               "Use the Ask tab for questions across the comparison.")
    tab_compare, tab_ask = st.tabs(["Comparison", "Ask"])
    with tab_compare:
        comparison_view(data)
    with tab_ask:
        ask_view(data)


def main() -> None:
    st.title("RFx & Quote Comparison")
    if not COMPARISON_PATH.exists():
        st.error("comparison.json not found. Run `python build_comparison.py` first.")
        st.stop()
    data = load_comparison(str(COMPARISON_PATH), COMPARISON_PATH.stat().st_mtime)

    if "goto_stage" in st.session_state:
        st.session_state["stage"] = st.session_state.pop("goto_stage")
    stage = st.radio("Stage", STAGES, key="stage", horizontal=True, label_visibility="collapsed")

    if stage == STAGES[0] and st.session_state.get("last_stage", STAGES[0]) != STAGES[0] \
            and "rfx_edited" in st.session_state:
        # Returning to Stage 1: rebuild the editors from the saved edits under fresh widget keys.
        st.session_state["rfx_draft"] = st.session_state["rfx_edited"]
        st.session_state["rfx_version"] = st.session_state.get("rfx_version", 0) + 1
    st.session_state["last_stage"] = stage
    st.divider()

    if stage == STAGES[0]:
        create_rfx_stage()
    elif stage == STAGES[1]:
        collect_stage(data)
    else:
        compare_stage(data)


main()
