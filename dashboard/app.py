"""Read-only local experiment dashboard with a downloadable blind review sheet."""

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluators.judge_agreement import agreement, review_sample  # noqa: E402
from framework.metrics import category_metrics, failure_counts, model_metrics  # noqa: E402
from framework.schema import load_config  # noqa: E402
from framework.storage import read_results  # noqa: E402
from regression.compare_runs import compare  # noqa: E402
from regression.release_gate import check_gates  # noqa: E402

st.set_page_config(page_title="Local LLM Lab", page_icon="◈", layout="wide")
st.markdown(
    """<style>
.block-container {padding-top: 4rem; max-width: 1400px;}
[data-testid="stMetric"] {border:1px solid #d9e2e8; border-radius:10px; padding:16px;}
.eyebrow {font-size:12px;letter-spacing:2px;color:#58767d;font-weight:700;}
</style>""",
    unsafe_allow_html=True,
)
st.markdown('<div class="eyebrow">LOCAL LLM EVALUATION & REGRESSION</div>', unsafe_allow_html=True)
st.title("Local LLM Lab")
st.caption(
    "Compare quality. Inspect failures. Catch regressions. Every response stays on your machine."
)

config = load_config(ROOT / "config.yaml")
results_dir = ROOT / config.results_dir
run_paths = sorted(results_dir.glob("*/results.csv"), reverse=True)
for alias in ("latest_results.csv", "baseline_results.csv"):
    if (results_dir / alias).exists():
        run_paths.append(results_dir / alias)

with st.sidebar:
    st.header("Experiment workspace")
    st.caption("Ollama · SQLite · local files")
    page = st.radio(
        "Workspace",
        [
            "Model comparison",
            "Failure analysis",
            "Regression analysis",
            "Test explorer",
            "Judge calibration",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    uploaded = st.file_uploader("Or open a result CSV", type="csv")
    choice = (
        st.selectbox(
            "Evaluation run", run_paths, format_func=lambda p: str(p.relative_to(results_dir))
        )
        if run_paths
        else None
    )
    if st.button("Refresh runs"):
        st.rerun()

source = uploaded if uploaded is not None else choice
if source is None:
    st.info("No evaluation runs yet. Run the benchmark to begin.")
    st.code(
        "python evaluation_runner.py --baseline\nstreamlit run dashboard/app.py", language="bash"
    )
    st.write("56 balanced benchmark cases · 24 adversarial cases · 2 starter models")
    st.stop()
try:
    frame = read_results(source)
except Exception as exc:
    st.error(f"Could not load run: {exc}")
    st.stop()

st.caption(
    f"Run {frame.run_id.iloc[0]} · {len(frame)} responses · suite: {', '.join(frame.suite.unique())}"
)
if not frame.run_complete.all():
    st.warning(
        "This run is incomplete. Release gates will fail and matched regression comparison is unavailable."
    )
if (frame.status != "ok").any():
    st.warning("This run contains execution errors. They count as failed tests and block release.")


def display_response(row):
    st.markdown(f"**{row['test_id']}** · {row['model']} · {row['category']}")
    st.write("PASS" if row["passed"] else f"FAIL · {row['failure_type']}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Prompt**")
        st.code(row.get("prompt", ""), language=None, wrap_lines=True)
        st.markdown("**Reference answer**")
        st.code(row.get("expected_answer", ""), language=None, wrap_lines=True)
    with right:
        st.markdown("**Model response**")
        st.code(row.get("response", ""), language=None, wrap_lines=True)
        st.caption(
            f"Score {row['score']:.2f} · {row['latency']:.2f}s · {row.get('evaluation_method', '')}"
        )
        if row.get("failure_reason"):
            st.error(row["failure_reason"])
        if row.get("done_reason") == "length":
            st.warning("Generation reached its token limit.")
        if row.get("judge_json"):
            st.markdown("**Local judge verdict**")
            st.json(json.loads(row["judge_json"]))
        if row.get("semantic_score", "") != "":
            st.caption(f"Semantic similarity: {row.get('semantic_score')} (auxiliary signal)")


if page == "Model comparison":
    summary = model_metrics(frame)
    for col, (_, row) in zip(st.columns(len(summary)), summary.iterrows()):
        with col:
            st.subheader(row.model)
            st.metric("Overall pass rate", f"{row.overall_pass_rate:.1%}")
            st.caption(
                f"{int(row.tests)} cases · {row.average_latency:.2f}s mean · {row.p95_latency:.2f}s p95"
            )
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Quality by category")
        categories = category_metrics(frame)
        categories["category"] = categories.category.str.replace("_", " ").str.title()
        chart = (
            alt.Chart(categories)
            .mark_bar(cornerRadiusEnd=3)
            .encode(
                x=alt.X(
                    "pass_rate:Q",
                    title="Pass rate",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%"),
                ),
                y=alt.Y("category:N", title=None, axis=alt.Axis(labelLimit=180)),
                yOffset="model:N",
                color=alt.Color(
                    "model:N",
                    title="Model",
                    scale=alt.Scale(range=["#147d80", "#c28b42", "#7b6ba7"]),
                ),
                tooltip=[
                    "model:N",
                    "category:N",
                    alt.Tooltip("pass_rate:Q", format=".1%"),
                    "tests:Q",
                ],
            )
            .properties(height=360)
        )
        st.altair_chart(chart, width="stretch")
    with right:
        st.subheader("Response time")
        st.bar_chart(summary.set_index("model")[["average_latency", "p95_latency"]], stack=False)
    st.subheader("Metric details")
    st.dataframe(summary, hide_index=True, width="stretch")
    st.caption(
        "Rates are proportions from 0 to 1. Hallucination rate is failure rate on successfully evaluated closed-world hallucination probes; it is a proxy. Latency includes model loading; judge time is excluded. Small case counts do not establish general model rankings."
    )
    with st.expander("Compare normal and adversarial suites"):
        alternate = (
            st.selectbox(
                "Other suite run",
                run_paths,
                key="suite_run",
                format_func=lambda p: str(p.relative_to(results_dir)),
            )
            if run_paths
            else None
        )
        if alternate:
            other = read_results(alternate)
            if set(frame.suite) == set(other.suite):
                st.info("Choose a run from a different suite to compare benchmark difficulty.")
            else:
                combined = pd.concat(
                    [
                        model_metrics(frame).assign(suite=frame.suite.iloc[0]),
                        model_metrics(other).assign(suite=other.suite.iloc[0]),
                    ]
                )
                st.dataframe(combined, hide_index=True)
                st.caption(
                    "Different test sets: descriptive comparison only. These are not matched regression deltas."
                )

elif page == "Failure analysis":
    failures = frame[~frame.passed]
    st.subheader("Failure patterns")
    st.metric("Failed cases", len(failures))
    if failures.empty:
        st.success("All evaluated cases passed.")
    else:
        counts = failure_counts(frame)
        st.bar_chart(
            counts.pivot(index="failure_type", columns="model", values="count").fillna(0),
            horizontal=True,
        )
        st.caption(
            "Failure types describe the failed check. Review the response to distinguish a formatting mismatch from an incorrect claim or inappropriate refusal."
        )
        kind = st.selectbox("Inspect a failure type", sorted(failures.failure_type.unique()))
        filtered = failures[failures.failure_type == kind]
        selected = st.selectbox(
            "Failed response",
            range(len(filtered)),
            format_func=lambda i: f"{filtered.iloc[i].model} · {filtered.iloc[i].test_id}",
        )
        display_response(filtered.iloc[selected])

elif page == "Regression analysis":
    st.subheader("Release readiness")
    baseline_source = st.file_uploader("Upload baseline CSV", type="csv", key="baseline_upload")
    if baseline_source is None and run_paths:
        default = next(
            (i for i, path in enumerate(run_paths) if path.name == "baseline_results.csv"), 0
        )
        baseline_source = st.selectbox(
            "Baseline run",
            run_paths,
            index=default,
            format_func=lambda p: str(p.relative_to(results_dir)),
            key="baseline_choice",
        )
    if baseline_source is not None:
        try:
            baseline = read_results(baseline_source)
            mapping = st.checkbox("Compare a model replacement")
            old_model = new_model = None
            if mapping:
                old_model = st.selectbox("Baseline model", sorted(baseline.model.unique()))
                new_model = st.selectbox("Current model", sorted(frame.model.unique()))
            report = compare(baseline, frame, old_model, new_model)
            if report["baseline_run"] == report["current_run"]:
                st.info(
                    "Baseline and current are the same run; select a later run to measure a change."
                )
            gate = check_gates(frame, config.release_gates, baseline, old_model, new_model)
            if gate["passed"]:
                st.success("RELEASE GATE · PASS")
            else:
                st.error("RELEASE GATE · FAILED")
                for reason in gate["reasons"]:
                    st.write("• " + reason)
            for model in report["models"]:
                st.markdown(f"### {model['baseline_model']} → {model['current_model']}")
                metrics = pd.DataFrame(model["metrics"])
                metrics["direction"] = metrics.apply(
                    lambda row: (
                        "unchanged"
                        if row.delta == 0
                        else "unavailable"
                        if pd.isna(row.delta)
                        else "improved"
                        if (row.delta > 0) == row.higher_is_better
                        else "regressed"
                    ),
                    axis=1,
                )
                st.dataframe(metrics, hide_index=True, width="stretch")
                st.caption(
                    "Rate deltas use proportions: −0.06 is a drop of 6 percentage points. Latency deltas use seconds."
                )
                left, right = st.columns(2)
                left.write({"New failures": model["new_failures"]})
                right.write({"Fixed failures": model["fixed_failures"]})
            st.download_button(
                "Download regression report",
                json.dumps(report, indent=2),
                "regression.json",
                "application/json",
            )
        except Exception as exc:
            st.error(f"Comparison unavailable: {exc}")

elif page == "Test explorer":
    st.subheader("Every prompt, every response")
    a, b, c, d = st.columns(4)
    models = a.multiselect(
        "Model", sorted(frame.model.unique()), default=sorted(frame.model.unique())
    )
    categories = b.multiselect(
        "Category", sorted(frame.category.unique()), default=sorted(frame.category.unique())
    )
    outcome = c.selectbox("Outcome", ["All", "PASS", "FAIL"])
    failure_type = d.selectbox(
        "Failure type", ["All", *sorted(x for x in frame.failure_type.unique() if x)]
    )
    filtered = frame[frame.model.isin(models) & frame.category.isin(categories)]
    if outcome != "All":
        filtered = filtered[filtered.passed == (outcome == "PASS")]
    if failure_type != "All":
        filtered = filtered[filtered.failure_type == failure_type]
    query = st.text_input("Search test ID or prompt")
    if query:
        filtered = filtered[
            filtered.test_id.str.contains(query, case=False, regex=False)
            | filtered.prompt.str.contains(query, case=False, regex=False)
        ]
    st.dataframe(
        filtered[["test_id", "model", "category", "passed", "score", "latency", "failure_type"]],
        hide_index=True,
        width="stretch",
    )
    if filtered.empty:
        st.info("No responses match these filters.")
    else:
        index = st.selectbox(
            "Open response",
            range(len(filtered)),
            format_func=lambda i: f"{filtered.iloc[i].model} · {filtered.iloc[i].test_id}",
        )
        display_response(filtered.iloc[index])
        st.download_button(
            "Download filtered results",
            filtered.to_csv(index=False),
            "filtered_results.csv",
            "text/csv",
        )

elif page == "Judge calibration":
    st.subheader("Evaluate the evaluator")
    st.write(
        "Review 30–50 responses yourself before trusting the judge. The review sheet hides both automated verdicts to reduce anchoring."
    )
    try:
        sample = review_sample(frame, 40)
        st.download_button(
            "Download 40-response blind review sheet",
            sample.to_csv(index=False),
            "human_labels.csv",
            "text/csv",
        )
    except ValueError as exc:
        st.info(str(exc))
    st.caption(
        "Fill human_passed with PASS or FAIL and record your reasoning in human_reason. Leave uncertain items blank. Use the same run when uploading labels."
    )
    labels_file = st.file_uploader("Upload completed human labels", type="csv", key="labels")
    if labels_file is not None:
        try:
            report, disagreements = agreement(
                frame, pd.read_csv(labels_file, keep_default_na=False)
            )
            a, b, c = st.columns(3)
            a.metric("Human / judge agreement", f"{report['agreement']:.1%}")
            b.metric("Reviewed pairs", report["compared"])
            c.metric(
                "Cohen's κ",
                f"{report['cohen_kappa']:.2f}"
                if report["cohen_kappa"] is not None
                else "Undefined",
            )
            st.json(report)
            st.subheader("Disagreements to inspect")
            st.dataframe(disagreements, hide_index=True, width="stretch")
            st.download_button(
                "Download disagreements",
                disagreements.to_csv(index=False),
                "disagreements.csv",
                "text/csv",
            )
        except Exception as exc:
            st.error(str(exc))
