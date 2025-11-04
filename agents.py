import os, io, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import dateparser
import pdfplumber

# -------------------- Data loading / saving --------------------
DATA_PATHS = ["coal_mine_incidents.csv"]

def load_data():
    for p in DATA_PATHS:
        if os.path.exists(p):
            if p.endswith(".xlsx"):
                return pd.read_excel(p)
            else:
                return pd.read_csv(p)
    # fallback demo
    return pd.DataFrame({
        "Date of Incident":["2021-01-12","2021-03-05","2022-07-18"],
        "Year":[2021,2021,2022],
        "State":["Jharkhand","Odisha","Chhattisgarh"],
        "Type of Mine":["Underground","Opencast","Underground"],
        "Accident Type":["Gas Leak","Fall of Roof","Collision"],
        "Injury Severity":["Major","Minor","Fatal"],
        "Worker Age":[29,33,46],
        "Worker Experience (Years)":[1,6,12],
        "PPE Used":["No","Yes","No"]
    })

def save_data(df: pd.DataFrame):
    os.makedirs("data", exist_ok=True)
    df.to_excel("data/incidents.xlsx", index=False)

# -------------------- Preprocess --------------------
ACCIDENT_TYPES = ["Fall of Roof","Gas Leak","Collision","Explosion","Slip/Trip","Fire","Electrocution","Machinery"]
MINE_TYPES = ["Underground","Opencast"]
INDIAN_STATES = [
    "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chhattisgarh","Goa","Gujarat","Haryana",
    "Himachal Pradesh","Jharkhand","Karnataka","Kerala","Madhya Pradesh","Maharashtra","Manipur",
    "Meghalaya","Mizoram","Nagaland","Odisha","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana",
    "Tripura","Uttar Pradesh","Uttarakhand","West Bengal","Jammu and Kashmir","Ladakh"
]

def prep_df(df: pd.DataFrame):
    # basic types
    df["Date of Incident"] = pd.to_datetime(df.get("Date of Incident"), errors="coerce")
    if "Year" not in df.columns or df["Year"].isna().any():
        df["Year"] = df["Date of Incident"].dt.year
    # helpers
    df["Severity Score"] = df["Injury Severity"].map({"Minor":0,"Major":1,"Fatal":2})
    df["High Risk"] = df["Injury Severity"].isin(["Major","Fatal"]).astype(int)
    return df

df = prep_df(load_data())
ACC_TYPES = sorted(list(set(ACCIDENT_TYPES) | set(df.get("Accident Type", pd.Series([],dtype=str)).dropna().unique())))
STATES    = sorted(list(set(INDIAN_STATES) | set(df.get("State", pd.Series([],dtype=str)).dropna().unique())))

def set_df(new_df: pd.DataFrame):
    global df, ACC_TYPES, STATES
    df = prep_df(new_df.copy())
    ACC_TYPES = sorted(list(set(ACCIDENT_TYPES) | set(df["Accident Type"].dropna().unique())))
    STATES    = sorted(list(set(INDIAN_STATES) | set(df["State"].dropna().unique())))
    save_data(df)

# -------------------- KPIs --------------------
def kpis_for(d: pd.DataFrame):
    if d.empty:
        return {"total":0,"fatality_rate%":0,"serious_rate%":0,"high_risk%":0,"severity_index":0}
    total = len(d)
    fatal = (d["Injury Severity"]=="Fatal").sum()
    major = (d["Injury Severity"]=="Major").sum()
    return {
        "total": total,
        "fatality_rate%": round(fatal/total*100, 2),
        "serious_rate%":  round(major/total*100, 2),
        "high_risk%":     round((fatal+major)/total*100, 2),
        "severity_index": round(((1*major + 2*fatal)/total), 2)
    }

    # If your CSV has Worker Age and you haven't created Age Group yet:
if "Age Group" not in df.columns and "Worker Age" in df.columns:
    age_bins = [0,27,32,37,47,60,200]
    age_lbls = ["≤27","28–32","33–37","38–47","48–60","60+"]
    df["Age Group"] = pd.cut(pd.to_numeric(df["Worker Age"], errors="coerce"),
                             bins=age_bins, labels=age_lbls, right=True)


# -------------------- PDF extraction (simple, robust) --------------------
ACC_SYNONYMS = {
    "Fall of Roof":["fall of roof","roof fall","strata fall","fall of sides","fall-of-roof"],
    "Gas Leak":["gas leak","methane","firedamp","gas emission","gas outburst"],
    "Collision":["collision","vehicle hit","dozer hit","dumpers collide","haul truck crash","run over"],
    "Explosion":["explosion","blast accident","detonation","misfire explosion"],
    "Slip/Trip":["slip","trip","fall on ground","fell while walking"],
    "Fire":["fire","smoke","flame"],
    "Electrocution":["electrocution","electric shock","shock"],
    "Machinery":["machinery","equipment","conveyor","crusher","screen","shovel","dragline","belt"]
}
def find_first(text: str, candidates: list):
    tl = text.lower()
    for c in candidates:
        if c.lower() in tl: return c
    return None

def guess_accident_type(text: str):
    tl = text.lower()
    for canonical, syns in ACC_SYNONYMS.items():
        if any(s in tl for s in syns): return canonical
    return None

def guess_mine_type(text: str):
    if "underground" in text.lower() or "u/g" in text.lower(): return "Underground"
    if "opencast" in text.lower() or "open cast" in text.lower() or "o/c" in text.lower(): return "Opencast"
    return None

def guess_severity(text: str):
    tl = text.lower()
    if "fatal" in tl or "died" in tl or "death" in tl: return "Fatal"
    if "serious" in tl or "fracture" in tl or "hospitalized" in tl or "major" in tl: return "Major"
    if "minor" in tl or "first aid" in tl or "bruise" in tl: return "Minor"
    return "Major"  # conservative

def guess_date(text: str):
    # try multiple lines for the first parseable date
    for line in text.splitlines():
        dt = dateparser.parse(line, settings={"DATE_ORDER":"DMY"})
        if dt: return dt.date().isoformat()
    return None

def parse_pdf_to_rows(pdf_bytes: bytes) -> pd.DataFrame:
    """Return 1–N rows extracted from a DGMS PDF (simple heuristic)."""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            full_text = "\n".join([p.extract_text() or "" for p in pdf.pages])
    except Exception:
        full_text = ""

    # if PDF has multiple accidents listed, we still create a single summary row (MVP)
    date_str = guess_date(full_text)
    state    = find_first(full_text, STATES) or None
    mine_t   = guess_mine_type(full_text) or "Underground"
    acc_t    = guess_accident_type(full_text) or "Machinery"
    severity = guess_severity(full_text)

    row = {
        "Date of Incident": dateparser.parse(date_str).date().isoformat() if date_str else None,
        "State": state or "Unknown",
        "Type of Mine": mine_t,
        "Accident Type": acc_t,
        "Injury Severity": severity,
        "PPE Used": "Unknown",
    }
    out = pd.DataFrame([row])
    out["Date of Incident"] = pd.to_datetime(out["Date of Incident"], errors="coerce")
    out["Year"] = out["Date of Incident"].dt.year
    return out

# -------------------- Query Agent --------------------
RECO = {
    "Fall of Roof":[
        "Increase roof bolting & strata monitoring.",
        "Pre-shift roof/side inspection with sign-off.",
        "Ground-movement alarms where feasible."
    ],
    "Gas Leak":[
        "Continuous methane detection with alarms.",
        "Ventilation audits; verify air quantities.",
        "Confined space permit & gas testing."
    ],
    "Collision":[
        "One-way haul roads & speed limits.",
        "Proximity alarms; segregate pedestrians.",
        "Reverse parking with spotter."
    ],
    "Electrocution":[
        "Strict lockout/tagout; insulated tools.",
        "Earth leakage & monthly audits.",
        "Refresher training for electrical crews."
    ],
    "Machinery":[
        "Preventive maintenance with checklists.",
        "Guarding & interlocks verification.",
        "Operator competency re-assessment."
    ]
}
DEFAULT_RECO = [
    "Reinforce SOP compliance and supervision.",
    "Toolbox talks on recent incidents/near-misses.",
    "Targeted refresher training for exposed crews."
]

def parse_query(q: str, _df: pd.DataFrame):
    ql = str(q).lower(); f = {}
    yrs = re.findall(r"\b(20\d{2})\b", ql)
    if yrs: f["Year"] = int(yrs[0])
    for a in sorted(_df["Accident Type"].dropna().unique()):
        if a.lower() in ql: f["Accident Type"] = a; break
    for s in sorted(_df["State"].dropna().unique()):
        if s.lower() in ql: f["State"] = s; break
    if "underground" in ql or "ug" in ql: f["Type of Mine"] = "Underground"
    if "opencast" in ql or "oc" in ql:    f["Type of Mine"] = "Opencast"
    if "fatal" in ql: f["Injury Severity"] = "Fatal"
    elif "major" in ql: f["Injury Severity"] = "Major"
    if "without ppe" in ql or "no ppe" in ql: f["PPE Used"] = "No"
    elif "with ppe" in ql or "ppe used" in ql: f["PPE Used"] = "Yes"
    return f

def apply_filters(dfin, filt: dict):
    d = dfin.copy()
    for col, val in filt.items():
        if col not in d.columns:
            continue
        d = d[d[col] == val]
    return d

def query_agent(query_text: str, dfin: pd.DataFrame = df, sample_rows: int = 10):
    filt = parse_query(query_text, dfin)
    res  = apply_filters(dfin, filt)
    if res.empty:
        return {"filters_used":filt, "message":"No records found."}
    total = kpis_for(res)
    cause   = filt.get("Accident Type")
    recos   = RECO.get(cause, DEFAULT_RECO)
    cols    = [c for c in ["Date of Incident","State","Type of Mine","Accident Type","Injury Severity","PPE Used"] if c in res.columns]
    sample  = res.sort_values("Date of Incident", ascending=False).head(sample_rows)[cols]
    return {"filters_used":filt, "kpis":total, "sample":sample, "recommendations":recos, "filtered_df":res}

# -------------------- Trend Watcher Agent --------------------
def quarter_bounds(year:int, q:int):
    q_starts = {1:"01-01", 2:"04-01", 3:"07-01", 4:"10-01"}
    q_ends   = {1:"03-31", 2:"06-30", 3:"09-30", 4:"12-31"}
    return f"{year}-{q_starts[q]}", f"{year}-{q_ends[q]}"

def last_quarter_windows(dfin):
    d = dfin["Date of Incident"].dropna()
    if d.empty:
        # fallback to whole year if dates missing
        y = pd.Timestamp.today().year
        return (f"{y}-10-01", f"{y}-12-31"), (f"{y}-07-01", f"{y}-09-30")
    last = d.max()
    y, m = last.year, last.month
    q = (m-1)//3 + 1
    cur = quarter_bounds(y, q)
    prev = quarter_bounds(y if q>1 else y-1, q-1 if q>1 else 4)
    return cur, prev

def _agg(dfin, keys=("State","Accident Type")):
    t = dfin.groupby(list(keys)).agg(
        total=("Accident Type","count"),
        fatal=("Injury Severity", lambda x:(x=="Fatal").sum()),
        major=("Injury Severity", lambda x:(x=="Major").sum())
    ).reset_index()
    if t.empty:
        t["serious_rate_pct"]=[]; t["severity_index"]=[]; t["high_risk_pct"]=[]
        return t
    t["serious_rate_pct"] = (t["major"]/t["total"]*100).round(2)
    t["severity_index"]   = ((1*t["major"] + 2*t["fatal"]) / t["total"]).round(2)
    t["high_risk_pct"]    = ((t["major"] + t["fatal"]) / t["total"]*100).round(2)
    return t

def _compare(cur, prev, keys):
    m = pd.merge(cur, prev, on=list(keys), how="outer", suffixes=("_cur","_prev")).fillna(0)
    m["delta_total"] = m["total_cur"] - m["total_prev"]
    m["delta_pct"]   = np.where(m["total_prev"]>0, (m["delta_total"]/m["total_prev"]*100).round(2), np.nan)
    m["delta_sev_idx"]       = (m["severity_index_cur"] - m["severity_index_prev"]).round(2)
    m["delta_high_risk_pct"] = (m["high_risk_pct_cur"] - m["high_risk_pct_prev"]).round(2)
    return m.sort_values(["delta_total","delta_pct"], ascending=False)

def TrendWatcherAgent(dfin, keys=("State","Accident Type"),
                      min_current_count=2, min_increase_pct=30, min_new_count=1, min_delta_sev_idx=0.5):
    cur_w, prev_w = last_quarter_windows(dfin)
    cs, ce = cur_w; ps, pe = prev_w
    cur_df = dfin[(dfin["Date of Incident"]>=pd.to_datetime(cs)) & (dfin["Date of Incident"]<=pd.to_datetime(ce))]
    prev_df= dfin[(dfin["Date of Incident"]>=pd.to_datetime(ps)) & (dfin["Date of Incident"]<=pd.to_datetime(pe))]
    cur, prev = _agg(cur_df, keys), _agg(prev_df, keys)
    deltas = _compare(cur, prev, keys)
    alerts=[]
    for _, r in deltas.iterrows():
        curr, prevn = int(r["total_cur"]), int(r["total_prev"])
        dpct = r["delta_pct"]; sev_c = r["severity_index_cur"]; sev_p = r["severity_index_prev"]; dsev = r["delta_sev_idx"]; hr_c = r["high_risk_pct_cur"]; dhr = r["delta_high_risk_pct"]
        label = " • ".join([str(r.get(k,"(All)")) for k in keys])
        if prevn == 0 and curr >= min_new_count:
            alerts.append(f"NEW • {label}: {curr} (prev 0) | SevIdx={sev_c} HR%={hr_c} | {cs}→{ce}")
            continue
        if prevn > 0 and curr >= min_current_count and (not np.isnan(dpct)) and dpct >= min_increase_pct:
            alerts.append(f"SPIKE • {label}: ↑{int(round(dpct))}% (cur {curr} vs {prevn}) | SevIdx={sev_c}")
            continue
        if dsev >= min_delta_sev_idx and curr >= min_new_count:
            alerts.append(f"SEVERITY • {label}: SevIdx {sev_p}→{sev_c} (Δ={dsev}) | HR% Δ={dhr}")
    return alerts, deltas, (cur_w, prev_w)

# -------------------- Report Agent (Excel) --------------------
def _group_table(d, by):
    t = d.groupby(by).agg(
        total=("Accident Type","count"),
        fatal=("Injury Severity", lambda x:(x=="Fatal").sum()),
        major=("Injury Severity", lambda x:(x=="Major").sum()),
        minor=("Injury Severity", lambda x:(x=="Minor").sum())
    ).sort_values("total", ascending=False)
    t["fatal_rate%"]   = (t["fatal"]/t["total"]*100).round(2)
    t["serious_rate%"] = (t["major"]/t["total"]*100).round(2)
    t["severity_index"]= ((1*t["major"] + 2*t["fatal"]) / t["total"]).round(2)
    t["high_risk%"]    = ((t["major"] + t["fatal"]) / t["total"]*100).round(2)
    return t

def ReportAgent(scope: dict, base_df: pd.DataFrame):
    d = base_df.copy()
    for k,v in scope.items():
        if k in d.columns: d = d[d[k]==v]
    if d.empty: return "No data for scope.", None

    total = len(d); fatal=(d["Injury Severity"]=="Fatal").sum(); major=(d["Injury Severity"]=="Major").sum()
    fatal_rate=round(fatal/total*100,2); serious_rate=round(major/total*100,2)
    high_risk=round((fatal+major)/total*100,2); sev_idx=round(((1*major+2*fatal)/total),2)

    by_cause=_group_table(d,"Accident Type")
    by_minetype=_group_table(d,"Type of Mine") if "Type of Mine" in d.columns else pd.DataFrame()
    by_age=d.groupby("Age Group").size().to_frame("count").sort_values("count", ascending=False) if "Age Group" in d.columns else pd.DataFrame()
    by_exp=d.groupby("Experience Group").size().to_frame("count").sort_values("count", ascending=False) if "Experience Group" in d.columns else pd.DataFrame()
    by_ppe=_group_table(d,"PPE Used") if "PPE Used" in d.columns else pd.DataFrame()
    by_state=_group_table(d,"State") if "State" in d.columns else pd.DataFrame()

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as xw:
        pd.DataFrame({"Metric":["Total incidents","Fatality rate %","Serious injury rate %","High-risk %","Severity Index"],
                      "Value":[total,fatal_rate,serious_rate,high_risk,sev_idx]}).to_excel(xw,"KPIs",index=False)
        d.sort_values("Date of Incident").to_excel(xw,"Records",index=False)
        by_cause.to_excel(xw,"By_AccidentType")
        if not by_minetype.empty: by_minetype.to_excel(xw,"By_MineType")
        if not by_age.empty: by_age.to_excel(xw,"By_AgeGroup")
        if not by_exp.empty: by_exp.to_excel(xw,"By_Experience")
        if not by_ppe.empty: by_ppe.to_excel(xw,"By_PPE")
        if not by_state.empty: by_state.to_excel(xw,"By_State")
    bio.seek(0)

    top3 = by_cause.head(3)[["total","fatal_rate%","serious_rate%","severity_index","high_risk%"]]
    summary = (
        f"Safety Audit — Scope: {scope}\n"
        f"Total incidents: {total} | Fatality: {fatal_rate}% | Serious: {serious_rate}% | "
        f"High-risk: {high_risk}% | Severity Index: {sev_idx}\n"
        f"Top causes:\n{top3.to_string() if not top3.empty else 'N/A'}"
    )
    return summary, bio
