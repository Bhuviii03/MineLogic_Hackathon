import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from agents import (
    df, set_df, parse_pdf_to_rows, query_agent, TrendWatcherAgent, ReportAgent,
    kpis_for, ACC_TYPES, STATES
)

st.set_page_config(page_title="Mine Safety — Agentic AI", layout="wide")
st.title("🛡️ Agentic AI for Indian Coal Mines")

tab_up, tab1, tab2, tab3, tab4 = st.tabs(["⬆ Upload & Extract","📊 Insights","🧑‍✈️ Ask Officer","🚨 Alerts","📄 Reports"])

# ---------- Upload & Extract ----------
with tab_up:
    st.subheader("Upload DGMS PDF(s) → Auto-extract to dataset")
    files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
    if st.button("Extract & Append") and files:
        rows = []
        for f in files:
            rows.append(parse_pdf_to_rows(f.read()))
        add = pd.concat(rows, ignore_index=True)
        st.write("Extracted preview:", add.head())
        new_df = pd.concat([df, add], ignore_index=True)
        set_df(new_df)  # updates global + saves to data/incidents.xlsx
        st.success(f"Added {len(add)} records. Dataset now has {len(new_df)} rows. Refresh Insights tab.")
    st.caption("Note: simple heuristic extractor (date/state/type/severity). You can edit data/incidents.xlsx later.")

# ---------- Insights ----------
with tab1:
    st.subheader("Dataset KPIs")
    k = kpis_for(df)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Incidents", k["total"])
    c2.metric("Fatality Rate %", k["fatality_rate%"])
    c3.metric("Serious Injury Rate %", k["serious_rate%"])
    c4.metric("Severity Index", k["severity_index"])

    if not df.empty and "Accident Type" in df.columns:
        fig, ax = plt.subplots()
        df["Accident Type"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Accident Type Distribution"); ax.set_ylabel("Count"); ax.set_xlabel("Type")
        st.pyplot(fig)

    if "Year" in df.columns:
        by_year = df.groupby("Year").size()
        st.line_chart(by_year)

        # Add a new tab for advanced visuals

    # ------------------------------
    # Helper: compute per-accident metrics
    # ------------------------------
    def by_accident_metrics(dfin):
        if dfin.empty or "Accident Type" not in dfin.columns or "Injury Severity" not in dfin.columns:
            return pd.DataFrame()
        t = dfin.groupby("Accident Type").agg(
            total=("Accident Type","count"),
            fatal=("Injury Severity", lambda x:(x=="Fatal").sum()),
            major=("Injury Severity", lambda x:(x=="Major").sum()),
            minor=("Injury Severity", lambda x:(x=="Minor").sum()),
        ).reset_index()
        t["fatal_rate%"]   = (t["fatal"]/t["total"]*100).round(2)
        t["serious_rate%"] = (t["major"]/t["total"]*100).round(2)
        t["high_risk%"]    = ((t["major"]+t["fatal"])/t["total"]*100).round(2)
        t["severity_index"]= ((1*t["major"] + 2*t["fatal"])/t["total"]).round(2)
        return t

    # =============================================================================
    # B) Risk Priority Matrix (by Accident Type)
    # =============================================================================
    st.markdown("### 🎯 Risk Priority Matrix — High-Risk % vs Severity Index (by Accident Type)")
    stats = by_accident_metrics(df)
    if not stats.empty:
        x = stats["high_risk%"]; y = stats["severity_index"]; labels = stats["Accident Type"]

        # thresholds = medians (you can change to percentiles)
        x_thr = float(x.median()); y_thr = float(y.median())

        fig2, ax2 = plt.subplots(figsize=(8,5))
        ax2.scatter(x, y, s=80)
        for xi, yi, lab in zip(x, y, labels):
            ax2.annotate(lab, (xi, yi), textcoords="offset points", xytext=(6,4))

        ax2.axvline(x_thr, ls="--", color="gray"); ax2.axhline(y_thr, ls="--", color="gray")
        ax2.set_xlabel("High-Risk Probability (%)")
        ax2.set_ylabel("Severity Index  (Major=1, Fatal=2)")
        ax2.set_title("Risk Priority Matrix — Focus Top-Right Quadrant")
        st.pyplot(fig2)

        st.caption(f"Median split used: High-Risk%={x_thr:.1f}, Severity Index={y_thr:.2f}. Items in the top-right quadrant deserve first attention.")
        st.dataframe(stats.sort_values(["severity_index","high_risk%"], ascending=False), use_container_width=True)
    else:
        st.info("Not enough data to compute accident-type metrics.")

    # =============================================================================
    # C) Severity & High-Risk by Age Group
    # =============================================================================
    st.markdown("### 👥 Severity & High-Risk by Age Group")
    if {"Age Group","Injury Severity"}.issubset(df.columns):
        ag = df.groupby("Age Group").agg(
            total=("Accident Type","count"),
            fatal=("Injury Severity", lambda x:(x=="Fatal").sum()),
            major=("Injury Severity", lambda x:(x=="Major").sum()),
            minor=("Injury Severity", lambda x:(x=="Minor").sum()),
        ).reset_index()
        if not ag.empty:
            ag["high_risk%"]    = ((ag["major"]+ag["fatal"])/ag["total"]*100).round(2)
            ag["severity_index"]= ((1*ag["major"] + 2*ag["fatal"])/ag["total"]).round(2)

            # Keep age groups in a sensible order if labels exist in your data
            desired_order = ["≤27","28–32","33–37","38–47","48–60","60+"]
            ag["Age Group"] = pd.Categorical(ag["Age Group"], categories=[g for g in desired_order if g in ag["Age Group"].tolist()], ordered=True)
            ag = ag.sort_values("Age Group")

            # dual-axis: bars (high-risk%) + line (severity index)
            fig3, ax3 = plt.subplots(figsize=(8,5))
            ax3.bar(ag["Age Group"].astype(str), ag["high_risk%"])
            ax3.set_ylabel("High-Risk Probability (%)")
            ax3.set_xlabel("Age Group")
            ax3.set_title("High-Risk% (bars) & Severity Index (line) by Age Group")

            ax4 = ax3.twinx()
            ax4.plot(ag["Age Group"].astype(str), ag["severity_index"], marker="o")
            ax4.set_ylabel("Severity Index")

            st.pyplot(fig3)
            st.dataframe(ag[["Age Group","total","fatal","major","minor","high_risk%","severity_index"]], use_container_width=True)
        else:
            st.info("No data after grouping by Age Group.")
    else:
        st.info("Columns needed: 'Age Group' and 'Injury Severity'. If you only have 'Worker Age', create 'Age Group' during preprocessing.")


# ---------- Ask Officer (Query Agent) ----------
with tab2:
    st.subheader("Ask the Digital Safety Officer")
    q = st.text_input("e.g., 'gas leak underground jharkhand 2021' or 'fatal fall of roof odisha'")
    if st.button("Answer"):
        res = query_agent(q, df)
        if "message" in res:
            st.warning(res["message"])
        else:
            st.write("**Filters understood:**", res["filters_used"])
            st.write("**KPIs:**", res["kpis"])
            st.dataframe(res["sample"], use_container_width=True)
            st.write("**Recommendations:**")
            for r in res["recommendations"]:
                st.success(r)

# ---------- Alerts (Trend Watcher) ----------
with tab3:
    st.subheader("Quarter-over-Quarter Trend Alerts")
    agg_mode = st.selectbox("Aggregate by", ["State • Accident Type","Accident Type (All India)","State (All causes)"])
    if agg_mode == "State • Accident Type":
        keys = ("State","Accident Type")
    elif agg_mode == "Accident Type (All India)":
        keys = ("Accident Type",)
    else:
        keys = ("State",)

    min_current = st.slider("Min current count", 1, 10, 1)
    min_increase = st.slider("Min % increase (spike)", 0, 100, 20)
    min_new = st.slider("Min new-count (prev=0 → now)", 1, 5, 1)
    min_dsev = st.slider("Min Δ Severity Index", 0.0, 2.0, 0.5, 0.1)

    if st.button("Run Alerts"):
        alerts, deltas, wins = TrendWatcherAgent(df, keys=keys,
                                                 min_current_count=min_current,
                                                 min_increase_pct=min_increase,
                                                 min_new_count=min_new,
                                                 min_delta_sev_idx=min_dsev)
        st.caption(f"Window: {wins[0][0]} → {wins[0][1]}  vs  {wins[1][0]} → {wins[1][1]}")
        if alerts:
            for a in alerts: st.error(a)
        else:
            st.info("No alerts under current thresholds.")
        if not deltas.empty:
            st.dataframe(deltas.head(30), use_container_width=True)

# ---------- Reports (Excel) ----------
with tab4:
    st.subheader("Generate Safety Audit (Excel)")
    scope_state = st.selectbox("Filter by State", ["(ALL)"] + STATES)
    years_sorted = sorted([int(y) for y in df["Year"].dropna().unique()])
    scope_year  = st.selectbox("Filter by Year", ["(ALL)"] + years_sorted)
    scope = {}
    if scope_state != "(ALL)": scope["State"] = scope_state
    if scope_year  != "(ALL)": scope["Year"]  = scope_year

    if st.button("Generate Report"):
        summary, xlsx_bytes = ReportAgent(scope, df)
        if xlsx_bytes is None:
            st.warning(summary)
        else:
            st.code(summary)
            fname = "safety_audit.xlsx" if not scope else f"safety_audit_{'_'.join([f'{k}-{v}' for k,v in scope.items()])}.xlsx".replace(" ","_")
            st.download_button("⬇️ Download Excel", data=xlsx_bytes, file_name=fname, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
