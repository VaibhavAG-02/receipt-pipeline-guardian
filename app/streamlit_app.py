"""Receipt Pipeline Guardian — operator console.

Deploys free on Streamlit Community Cloud. If the warehouse is missing on first
load it builds everything in-process (about a minute), so nothing needs to be
committed and the app self-heals on Streamlit's ephemeral filesystem.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import altair as alt
import duckdb
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import theme  # noqa: E402

from rpg.config import ARTIFACTS, WAREHOUSE  # noqa: E402

st.set_page_config(
    page_title="Receipt Pipeline Guardian",
    page_icon="▮",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(theme.css(), unsafe_allow_html=True)
alt.themes.register("rpg", theme.chart_theme)
alt.themes.enable("rpg")

H = lambda s: st.markdown(s, unsafe_allow_html=True)  # noqa: E731


def _run(cmd: list[str], cwd: Path, env_extra: dict[str, str] | None = None) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **(env_extra or {})}
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        # Show the real cause in the UI. capture_output otherwise hides it and
        # the user sees only an opaque CalledProcessError.
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-25:]
        st.error(f"`{' '.join(cmd)}` failed (exit {proc.returncode})")
        st.code("\n".join(tail) or "no output captured")
        raise RuntimeError(f"step failed: {' '.join(cmd)}")


def build_warehouse(n_receipts: int, seed: int) -> None:
    with st.status("Building pipeline", expanded=True) as status:
        st.write("Generating receipts, applying the contract gate, training, explaining")
        _run([sys.executable, "-m", "rpg.pipeline", "--receipts", str(n_receipts),
              "--seed", str(seed)], cwd=ROOT)
        st.write("Building bronze, silver and gold, then running data checks")
        # `python -m dbt.cli.main` rather than a bare `dbt`: the console script
        # may not be on PATH in a hosted environment, but the module always is
        # if dbt installed at all. Same interpreter, no PATH assumption.
        _run([sys.executable, "-m", "dbt.cli.main", "build", "--profiles-dir", "."],
             cwd=ROOT / "dbt", env_extra={"RPG_WAREHOUSE": str(WAREHOUSE)})
        status.update(label="Pipeline built", state="complete", expanded=False)


@st.cache_data(show_spinner=False)
def q(sql: str) -> pd.DataFrame:
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    try:
        return con.execute(sql).fetch_df()
    finally:
        con.close()


with st.sidebar:
    H('<div class="sec-eyebrow">Controls</div>')
    n = st.slider("Receipts", 5_000, 60_000, 20_000, step=5_000)
    seed = st.number_input("Seed", value=42, step=1)
    if st.button("Rebuild pipeline", use_container_width=True):
        st.cache_data.clear()
        build_warehouse(n, int(seed))
        st.rerun()
    H(theme.note("Changing the seed regenerates the receipts, retrains the model "
                 "and rebuilds every mart. Nothing here is precomputed."))

if not WAREHOUSE.exists():
    st.info("No warehouse yet. Building it now — about a minute.")
    build_warehouse(20_000, 42)
    st.rerun()

summary = json.loads((ARTIFACTS / "run_summary.json").read_text()) \
    if (ARTIFACTS / "run_summary.json").exists() else {}
gate = summary.get("gate", {})
metrics = summary.get("metrics", {})
drift = summary.get("drift", {})
fair_ok = summary.get("fairness_passed", True)

gate_ok = gate.get("passed", False)
H(theme.masthead(
    "Receipt Pipeline<br>Guardian",
    "Bad receipt data is caught at the gate, not discovered three weeks later in "
    "a dashboard nobody trusts. Contract checks quarantine what is provably "
    "broken; a scored model queues what merely looks wrong.",
    [
        ("RECEIPTS IN", f"{gate.get('n_rows', 0):,}"),
        ("QUARANTINED", f"{gate.get('n_quarantined', 0):,}"),
        ("FLAG RATE", f"{metrics.get('recall_at_threshold', 0):.1%}"),
        ("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}"),
    ],
    ("GATE", "PASS" if gate_ok else "FAIL"),
))

tabs = st.tabs(["Gate", "Model", "Drift & fairness", "Queue", "Warehouse", "Score a receipt"])

# ------------------------------------------------------------------ GATE --
with tabs[0]:
    H(theme.stats([
        ("Rows in batch", f"{gate.get('n_rows', 0):,}", "", "after ingest"),
        ("Quarantined", f"{gate.get('n_quarantined', 0):,}", "watch", "held, not dropped"),
        ("Quarantine rate", f"{gate.get('quarantine_rate', 0):.2%}", "", "ceiling 15%"),
        ("Null rate", f"{gate.get('null_rate', 0):.3%}", "", "ceiling 2%"),
        ("Verdict", "PASS" if gate_ok else "FAIL", "ok" if gate_ok else "alert",
         "batch may land"),
    ]))
    if gate.get("failures"):
        st.error(" · ".join(gate["failures"]))

    H(theme.section("Bronze", "Why rows were held back"))
    qr = q("select quarantine_reason as reason, count(*) as n "
           "from main_bronze.br_quarantine group by 1 order by n desc")
    if not qr.empty:
        st.altair_chart(
            alt.Chart(qr).mark_bar(height=18, color=theme.WATCH, cornerRadius=1).encode(
                x=alt.X("n:Q", title="RECEIPTS"),
                y=alt.Y("reason:N", sort="-x", title=None),
            ).properties(height=190),
            use_container_width=True)
    H(theme.note(
        "Every rejected row keeps its reason and stays queryable. Dropping records "
        "silently is how a warehouse ends up with numbers nobody can reconcile."))

    H(theme.section("Gold", "Daily vitals"))
    daily = q("select * from main_gold.mart_quality_metrics order by submitted_date")
    if not daily.empty:
        long = daily.melt(
            id_vars="submitted_date",
            value_vars=["quarantine_rate", "reconciliation_failure_rate", "flag_rate"],
            var_name="signal", value_name="rate")
        st.altair_chart(
            alt.Chart(long).mark_line(strokeWidth=1.6).encode(
                x=alt.X("submitted_date:T", title=None),
                y=alt.Y("rate:Q", title="RATE", axis=alt.Axis(format="%")),
                color=alt.Color("signal:N", title=None),
            ).properties(height=260),
            use_container_width=True)
    H(theme.note(
        "Three independent signals: schema health, arithmetic health, and the "
        "model's opinion. They move independently, which is what makes a real "
        "incident legible instead of a single ambiguous line."))

# ----------------------------------------------------------------- MODEL --
with tabs[1]:
    H(theme.stats([
        ("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}", "ok", "positives ~6%"),
        ("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}", "", "flattering here"),
        ("Precision @ 5%", f"{metrics.get('precision_at_5pct', 0):.1%}", "ok",
         "reviewer capacity"),
        ("Recall @ threshold", f"{metrics.get('recall_at_threshold', 0):.1%}", "", "operating point"),
        ("Hold-out rows", f"{metrics.get('n_test', 0):,}", "", "chronological split"),
    ]))
    H(theme.note(
        "PR-AUC leads. With a 6% positive rate a model that predicts 'clean' every "
        "time scores 94% accuracy and is worthless, and ROC-AUC is similarly "
        "generous. Precision at top-k is the number that maps to the real "
        "constraint: a reviewer can only work through so many receipts a day. "
        "The split is chronological because the behavioural features look back "
        "24 hours, so a random split would leak the future."))

    left, right = st.columns([1.05, 1])
    with left:
        H(theme.section("Evaluation", "Recall by defect type"))
        bt = metrics.get("recall_by_anomaly_type", {})
        if bt:
            rt = pd.DataFrame([{"defect": k, "n": v["n"], "recall": v["recall_at_threshold"]}
                               for k, v in bt.items()]).sort_values("recall")
            rt["weak"] = rt["recall"] < 0.7
            st.altair_chart(
                alt.Chart(rt).mark_bar(height=16, cornerRadius=1).encode(
                    x=alt.X("recall:Q", title="RECALL", axis=alt.Axis(format="%"),
                            scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("defect:N", sort="x", title=None),
                    color=alt.Color("weak:N", title=None, legend=None,
                                    scale=alt.Scale(domain=[True, False],
                                                    range=[theme.ALERT, theme.OK])),
                ).properties(height=200), use_container_width=True)
            st.dataframe(rt, hide_index=True, use_container_width=True)
        H(theme.note(
            "Broken out on purpose. A model that nails arithmetic errors while "
            "missing every price outlier should not be able to hide behind a good "
            "average — and that is exactly what happens here."))
    with right:
        H(theme.section("Attribution", "Global signal (mean |SHAP|)"))
        fa = q("select * from main_gold.mart_feature_attribution limit 12")
        if not fa.empty:
            st.altair_chart(
                alt.Chart(fa).mark_bar(height=13, color=theme.INK_MUTE, cornerRadius=1).encode(
                    x=alt.X("mean_abs_shap:Q", title="MEAN |SHAP|"),
                    y=alt.Y("feature:N", sort="-x", title=None),
                ).properties(height=290), use_container_width=True)
        H(theme.note(
            "Mean |SHAP| rather than gain-based importance: the two routinely "
            "disagree, and this is the one consistent with the per-receipt "
            "explanations in the queue."))

# ------------------------------------------------------- DRIFT & FAIRNESS --
with tabs[2]:
    retrain = drift.get("retrain_recommended", False)
    H(theme.stats([
        ("Prediction PSI", f"{drift.get('prediction_psi', 0):.3f}",
         "watch" if retrain else "ok", drift.get("prediction_status", "")),
        ("Drifted features", f"{len(drift.get('drifted_features', []))}", "",
         "PSI ≥ 0.25"),
        ("Retrain", "RECOMMENDED" if retrain else "NOT NEEDED",
         "watch" if retrain else "ok", "gate output"),
        ("Fairness", "PASS" if fair_ok else "BLOCKED", "ok" if fair_ok else "alert",
         "max slice gap 0.20"),
    ]))
    for v in summary.get("fairness_violations", []):
        st.error(v)

    H(theme.section("Monitoring", "Population stability by feature"))
    psi_df = pd.DataFrame(sorted(drift.get("feature_psi", {}).items(),
                                 key=lambda kv: -kv[1]), columns=["feature", "psi"]).head(14)
    if not psi_df.empty:
        # Altair 6 does not allow nested alt.condition, so the band is computed
        # here and mapped through an explicit scale. Clearer to read anyway.
        psi_df["band"] = pd.cut(psi_df["psi"], bins=[-1, 0.10, 0.25, 1e9],
                                labels=["stable", "moderate", "significant"])
        st.altair_chart(
            alt.Chart(psi_df).mark_bar(height=13, cornerRadius=1).encode(
                x=alt.X("psi:Q", title="PSI"),
                y=alt.Y("feature:N", sort="-x", title=None),
                color=alt.Color("band:N", title=None, scale=alt.Scale(
                    domain=["stable", "moderate", "significant"],
                    range=[theme.INK_MUTE, theme.WATCH, theme.ALERT])),
            ).properties(height=300), use_container_width=True)
    H(theme.note(
        "PSI against the training reference, binned on the reference so the "
        "question is where new data landed relative to what we trained on. Bands "
        "are the conventional <code>0.10</code> / <code>0.25</code>. Prediction "
        "drift is tracked separately because scores can move while inputs look "
        "stable — that is the model extrapolating, and it is the earlier warning. "
        "Retrain reads as recommended here because the reference and current "
        "windows are different time periods by construction."))

    H(theme.section("Release gate", "Recall by slice"))
    sl = q("select * from main_gold.mart_model_slices")
    st.dataframe(sl, hide_index=True, use_container_width=True)
    H(theme.note(
        "A slice that lags overall recall by more than 0.20 blocks promotion. "
        "Slices below the reliability threshold are shown but never block — an "
        "11-row slice at zero recall is noise, and a gate that fires on noise gets "
        "switched off within a week. Receipts carry no demographic attributes, so "
        "these slices are operational; a protected attribute plugs in unchanged."))

# ----------------------------------------------------------------- QUEUE --
with tabs[3]:
    H(theme.section("Gold", "Review queue"))
    c1, c2 = st.columns([1, 2])
    with c1:
        top_n = st.slider("Rows", 10, 300, 60, step=10)
    rq = q(f"select * from main_gold.mart_review_queue limit {int(top_n)}")
    hints = sorted(rq["rule_based_hint"].unique().tolist()) if not rq.empty else []
    with c2:
        chosen = st.multiselect("Rule-based hint", hints, default=hints)
    view = rq[rq["rule_based_hint"].isin(chosen)] if chosen else rq
    st.dataframe(
        view[["receipt_id", "submitted_at", "store_id", "region", "anomaly_score",
              "model_reasoning", "reported_total", "computed_total",
              "total_residual_pct", "max_qty", "image_quality", "rule_based_hint"]],
        hide_index=True, use_container_width=True, height=460)
    H(theme.note(
        "<code>model_reasoning</code> is the signed top-3 SHAP contribution for "
        "that specific receipt — not global importance, which would print the same "
        "answer on every row. <code>rule_based_hint</code> is deterministic. They "
        "are kept apart because the rule stays explainable when the model is stale."))

# ------------------------------------------------------------- WAREHOUSE --
with tabs[4]:
    H(theme.section("Silver → Gold", "Type 2 store dimension"))
    changed = q("""select store_natural_key as store_id, version_number as v,
                          valid_from, valid_to, is_current, region, store_format,
                          change_reason
                   from main_gold.dim_store
                   where store_natural_key in (
                     select store_natural_key from main_gold.dim_store
                     group by 1 having count(*) > 1)
                   order by store_natural_key, version_number limit 40""")
    st.dataframe(changed, hide_index=True, use_container_width=True, height=380)
    H(theme.note(
        "Stores with more than one version. Facts join on <code>submitted_at >= "
        "valid_from and < valid_to</code>, so a receipt from March resolves to the "
        "store as it was in March. Joining on <code>store_id</code> alone would "
        "restate last quarter's numbers every time a store is remodelled. Five "
        "tests keep this honest — no overlaps, no gaps, one current version per "
        "store, every fact inside its window, and a fan-out guard on row count."))

    H(theme.section("Ops", "Store health by region and format"))
    sh = q("""select region, store_format, sum(n_receipts) as n_receipts,
                     sum(n_flagged)*1.0/nullif(sum(n_receipts),0) as flag_rate,
                     avg(avg_image_quality) as avg_image_quality
              from main_gold.mart_store_health
              group by 1,2 order by n_receipts desc""")
    if not sh.empty:
        st.altair_chart(
            alt.Chart(sh).mark_rect(cornerRadius=1).encode(
                x=alt.X("store_format:N", title=None),
                y=alt.Y("region:N", title=None),
                color=alt.Color("flag_rate:Q", title="FLAG RATE",
                                scale=alt.Scale(scheme="yellowgreenblue", reverse=True)),
            ).properties(height=210), use_container_width=True)
    st.dataframe(sh, hide_index=True, use_container_width=True)
    H(theme.note("Sliced by the attributes in force at the time, via the Type 2 join."))

# --------------------------------------------------------------- SCORE ----
with tabs[5]:
    import io as _io

    from rpg.explain import explain
    from rpg.features import build_features
    from rpg.ingest import (
        IngestError,
        build_receipt,
        ocr_receipt,
        read_upload,
        render_receipt_image,
    )
    from rpg.quality import split_quarantine
    from rpg.train import load_model
    from rpg.train import score as score_features

    def run_through_pipeline(df: pd.DataFrame):
        """Same gate, same features, same model as a batch run. No side path."""
        as_of = pd.Timestamp(df["submitted_at"].max())
        clean, held = split_quarantine(df, as_of=as_of)
        if clean.empty:
            return clean, held, pd.DataFrame()
        feats = build_features(clean)
        scored = score_features(feats)
        ex = explain(load_model(), feats)
        out = (clean[["receipt_id", "store_id", "submitted_at", "total"]]
               .merge(scored, on="receipt_id")
               .merge(ex[["receipt_id", "top_factors"]], on="receipt_id")
               .sort_values("anomaly_score", ascending=False))
        return clean, held, out

    def show_result(clean, held, out):
        if not held.empty:
            H(theme.section("Contract gate", "Held at the door"))
            st.dataframe(held[["receipt_id", "store_id", "quarantine_reason"]],
                         hide_index=True, use_container_width=True)
            H(theme.note("These never reach the model. They are provably broken, "
                         "and a deterministic rule is the right tool."))
        if out.empty:
            return
        H(theme.section("Model", "Scored"))
        top = out.iloc[0]
        tone = "alert" if top.anomaly_score >= 0.75 else (
            "watch" if top.anomaly_score >= 0.35 else "ok")
        H(theme.stats([
            ("Receipts scored", f"{len(out):,}", "", "through the gate"),
            ("Highest score", f"{top.anomaly_score:.4f}", tone, str(top.receipt_id)),
            ("Flagged", f"{int(out.is_flagged.sum())}", "", "at the threshold"),
        ]))
        st.dataframe(
            out[["receipt_id", "store_id", "total", "anomaly_score", "is_flagged",
                 "top_factors"]],
            hide_index=True, use_container_width=True)
        H(theme.note("<code>top_factors</code> is the signed top-3 SHAP contribution "
                     "for that specific receipt — why <em>this</em> one scored what "
                     "it did, not what matters on average."))

    H(theme.section("Try it", "Put a receipt through the pipeline"))
    H(theme.note("Whatever you submit here takes the same route as generated data: "
                 "contract gate, then features, then the trained model. Nothing is "
                 "special-cased, so what you see is what a batch run would do."))

    mode = st.radio("Input", ["Build one by hand", "Upload a file", "Upload a photo"],
                    horizontal=True, label_visibility="collapsed")

    # ---- 1. form -------------------------------------------------------
    if mode == "Build one by hand":
        st.caption("Edit the basket or the total, then watch the score move.")
        default_items = pd.DataFrame([
            {"name": "Whole Milk 1gal", "qty": 2, "unit_price": 4.29},
            {"name": "Pasta 16oz", "qty": 3, "unit_price": 1.99},
            {"name": "Cheddar Block 8oz", "qty": 1, "unit_price": 4.75},
        ])
        edited = st.data_editor(default_items, num_rows="dynamic",
                                use_container_width=True, key="basket")
        computed = float((edited["qty"] * edited["unit_price"]).sum())
        c1, c2, c3 = st.columns(3)
        with c1:
            total = st.number_input("Printed total", value=round(computed * 1.0825, 2),
                                    step=0.01, format="%.2f")
        with c2:
            iq = st.slider("Photo quality", 0.05, 1.0, 0.92, 0.01)
        with c3:
            store = st.selectbox("Store", [f"ST{str(i).zfill(3)}" for i in range(1, 11)])
        st.caption(f"Line items add up to {computed:,.2f}; "
                   f"with tax that is {computed * 1.0825:,.2f}.")

        if st.button("Score this receipt", type="primary"):
            try:
                items = [{"sku": f"SKU{i}", "name": str(r["name"]),
                          "qty": int(r["qty"]), "unit_price": float(r["unit_price"])}
                         for i, r in edited.iterrows()]
                df = build_receipt(items, total=total, store_id=store,
                                   image_quality=iq,
                                   submitted_at=pd.Timestamp.now("UTC"))
                show_result(*run_through_pipeline(df))
            except IngestError as e:
                st.error(str(e))

    # ---- 2. file -------------------------------------------------------
    elif mode == "Upload a file":
        st.caption("CSV, JSON or JSONL — one row per receipt, or one row per line item.")
        sample = [{
            "receipt_id": "R-0001", "store_id": "ST001", "user_id": "U000001",
            "submitted_at": "2026-06-20T10:00:00Z", "total": 15.75,
            "items": [{"sku": "SKU1001", "name": "Whole Milk", "qty": 2,
                       "unit_price": 4.29}],
        }]
        st.download_button("Download a sample file",
                           json.dumps(sample, indent=2).encode(),
                           "sample_receipts.json", "application/json")
        up = st.file_uploader("Receipt file", type=["csv", "json", "jsonl"],
                              label_visibility="collapsed")
        if up is not None:
            try:
                df = read_upload(up.getvalue(), up.name)
                st.success(f"Parsed {len(df):,} receipt(s).")
                show_result(*run_through_pipeline(df))
            except IngestError as e:
                st.error(str(e))

    # ---- 3. photo ------------------------------------------------------
    else:
        st.caption("A photo of a receipt, read with Tesseract.")
        H(theme.note("This is the weakest link and is labelled as such. Tesseract on "
                     "a creased thermal receipt under kitchen light is a coin flip. "
                     "That is a real property of the problem, not something a "
                     "confidence number fixes."))
        if st.button("Generate a sample receipt image"):
            rec = {"store_id": "ST007", "items": [
                {"sku": "S1", "name": "Whole Milk", "qty": 2, "unit_price": 4.29},
                {"sku": "S2", "name": "Pasta 16oz", "qty": 3, "unit_price": 1.99},
                {"sku": "S3", "name": "Cheddar Block", "qty": 1, "unit_price": 4.75}],
                "subtotal": 20.50, "tax": 1.69, "total": 22.19}
            buf = _io.BytesIO()
            render_receipt_image(rec).save(buf, "PNG")
            st.session_state["sample_png"] = buf.getvalue()
        if "sample_png" in st.session_state:
            st.download_button("Download the sample image",
                               st.session_state["sample_png"],
                               "sample_receipt.png", "image/png")
            st.image(st.session_state["sample_png"], width=300)

        photo = st.file_uploader("Receipt photo", type=["png", "jpg", "jpeg"],
                                 label_visibility="collapsed")
        if photo is not None:
            try:
                df, parsed = ocr_receipt(photo.getvalue())
                a, b = st.columns([1, 1.4])
                with a:
                    st.image(photo.getvalue(), caption="Uploaded", width=280)
                with b:
                    st.markdown("**What was read**")
                    st.dataframe(pd.DataFrame(parsed["items"]), hide_index=True,
                                 use_container_width=True)
                    st.caption(
                        f"Printed total {'read from the receipt' if parsed['total_was_read'] else 'not found, so computed from the items'}"
                        f" · tax {'read' if parsed['tax'] is not None else 'not read, left null'}")
                    if parsed["unparsed"]:
                        with st.expander(f"{len(parsed['unparsed'])} line(s) not parsed"):
                            st.code("\n".join(parsed["unparsed"]))
                show_result(*run_through_pipeline(df))
                H(theme.note("A misread tax label leaves tax null rather than wrong. "
                             "That field is nullable by design and the marts recompute "
                             "it from the line items — which is exactly the failure "
                             "this pipeline exists to absorb."))
            except IngestError as e:
                st.error(str(e))

H(theme.section("Caveats", "What this does not claim"))
H(theme.note(
    "The data is synthetic, so absolute metrics describe the generator more than "
    "production receipts — the relative difficulty ranking is the transferable "
    "part. <code>tax_is_null</code> dominates attribution because the OCR-dropout "
    "defect is defined by nulling that field, which is a giveaway the model would "
    "not get on real data. Drift is measured train-versus-holdout, not across real "
    "time. And this is not production ready: no idempotent or incremental loads, "
    "no orchestration, no secrets management, no PII controls. PRODUCTION.md sizes "
    "each gap."))
