import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduRisk · Student Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Playfair+Display:wght@700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --mint:   #d4f5e2;
  --sage:   #52b788;
  --sky:    #dbeafe;
  --azure:  #3b82f6;
  --lemon:  #fef9c3;
  --amber:  #f59e0b;
  --snow:   #f8fafc;
  --slate:  #1e293b;
  --mist:   #e2e8f0;
  --danger: #ef4444;
  --warn:   #f97316;
  --ok:     #22c55e;
}

html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif !important;
  background: #f0f7f4 !important;
  color: var(--slate) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #e8f5ee 0%, #dbeafe 100%) !important;
  border-right: 2px solid #c7e8d8 !important;
}
section[data-testid="stSidebar"] .block-container { padding-top: 2rem; }

/* Top bar */
.hero-bar {
  background: linear-gradient(135deg, #1a6b45 0%, #1d4ed8 100%);
  border-radius: 18px;
  padding: 2.2rem 2.5rem;
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 1.5rem;
  box-shadow: 0 8px 32px rgba(29,78,216,.15);
}
.hero-title {
  font-family: 'Playfair Display', serif !important;
  font-size: 2.4rem; color: #fff; margin: 0;
  text-shadow: 0 2px 8px rgba(0,0,0,.2);
}
.hero-sub { color: #b2f0d0; font-size: .95rem; margin-top: .3rem; }
.hero-badge {
  background: rgba(255,255,255,.18);
  border: 1.5px solid rgba(255,255,255,.4);
  border-radius: 50px; padding: .5rem 1.2rem;
  color: #fff; font-size: .85rem; font-weight: 600;
  backdrop-filter: blur(6px);
}

/* KPI cards */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
  border-radius: 16px; padding: 1.3rem 1.5rem;
  display: flex; flex-direction: column; gap: .3rem;
  box-shadow: 0 2px 12px rgba(0,0,0,.07);
  border: 1.5px solid rgba(255,255,255,.8);
  position: relative; overflow: hidden;
}
.kpi-card::before {
  content: ''; position: absolute;
  top: -20px; right: -20px;
  width: 80px; height: 80px;
  border-radius: 50%; opacity: .15;
}
.kpi-green  { background: var(--mint); } .kpi-green::before  { background: var(--sage); }
.kpi-blue   { background: var(--sky);  } .kpi-blue::before   { background: var(--azure); }
.kpi-yellow { background: var(--lemon);} .kpi-yellow::before { background: var(--amber); }
.kpi-red    { background: #fee2e2;     } .kpi-red::before    { background: var(--danger); }

.kpi-label  { font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; opacity: .6; }
.kpi-value  { font-size: 2rem; font-weight: 700; line-height: 1; }
.kpi-delta  { font-size: .78rem; font-weight: 500; opacity: .7; }

/* Section headers */
.sec-head {
  display: flex; align-items: center; gap: .6rem;
  font-size: 1.15rem; font-weight: 700;
  color: var(--slate); margin: 1.5rem 0 .8rem;
  border-left: 4px solid var(--sage);
  padding-left: .8rem;
}

/* Risk badge */
.risk-badge {
  display: inline-flex; align-items: center; gap: .4rem;
  border-radius: 50px; padding: .35rem 1rem;
  font-weight: 700; font-size: .85rem;
}
.risk-high   { background: #fee2e2; color: #b91c1c; border: 1.5px solid #fca5a5; }
.risk-medium { background: #fff7ed; color: #c2410c; border: 1.5px solid #fdba74; }
.risk-low    { background: var(--mint); color: #166534; border: 1.5px solid #86efac; }

/* Table */
.styled-table { width: 100%; border-collapse: collapse; font-size: .88rem; }
.styled-table th {
  background: #e8f5ee; color: var(--slate); font-weight: 600;
  padding: .65rem 1rem; text-align: left; border-bottom: 2px solid #c7e8d8;
}
.styled-table td { padding: .55rem 1rem; border-bottom: 1px solid var(--mist); }
.styled-table tr:hover td { background: #f0fdf4; }

/* Prediction form card */
.pred-card {
  background: #fff; border-radius: 18px;
  padding: 2rem; margin-top: 1rem;
  box-shadow: 0 4px 20px rgba(0,0,0,.08);
  border: 1.5px solid var(--mist);
}

/* Insight pill */
.insight-pill {
  background: var(--lemon); border: 1.5px solid #fde68a;
  border-radius: 12px; padding: .9rem 1.2rem;
  margin: .4rem 0; font-size: .9rem; line-height: 1.5;
}
.insight-pill strong { color: #92400e; }

/* Metric row */
.metric-row {
  display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;
}
.metric-chip {
  background: #fff; border-radius: 12px;
  padding: .7rem 1.2rem; font-size: .88rem;
  border: 1.5px solid var(--mist);
  box-shadow: 0 2px 8px rgba(0,0,0,.05);
}
.metric-chip b { color: var(--azure); }

/* Footer */
.footer {
  text-align: center; padding: 2rem 0 1rem;
  color: #94a3b8; font-size: .8rem; border-top: 1px solid var(--mist);
  margin-top: 2rem;
}

/* Streamlit overrides */
.stButton > button {
  background: linear-gradient(135deg, var(--sage), var(--azure)) !important;
  color: #fff !important; border: none !important;
  border-radius: 10px !important; font-weight: 600 !important;
  padding: .55rem 1.8rem !important;
  transition: transform .15s, box-shadow .15s !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px rgba(59,130,246,.3) !important;
}
.stSelectbox > div > div, .stSlider { color: var(--slate) !important; }
div[data-testid="stMetric"] {
  background: #fff; border-radius: 12px; padding: .8rem 1rem;
  border: 1.5px solid var(--mist);
}
div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
#  DATA GENERATION
# ╔══════════════════════════════════════════════════════════╗
@st.cache_data
def generate_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = max(100, n)

    # Base student attributes
    study_time   = rng.integers(1, 6, n)          # 1–5 hrs / day
    attendance   = rng.integers(40, 101, n)        # %
    prev_grade   = rng.integers(40, 101, n)        # previous term marks
    family_sup   = rng.choice([0, 1], n, p=[.4,.6])
    internet     = rng.choice([0, 1], n, p=[.3,.7])
    extra_act    = rng.choice([0, 1], n, p=[.45,.55])
    health       = rng.integers(1, 6, n)           # 1=poor … 5=excellent
    absences     = rng.integers(0, 25, n)
    gender       = rng.choice(["Male","Female","Other"], n, p=[.48,.48,.04])
    grade_level  = rng.choice(["Grade 9","Grade 10","Grade 11","Grade 12"], n)
    school_type  = rng.choice(["Public","Private"], n, p=[.55,.45])

    # Exam score (realistically influenced by predictors)
    noise = rng.normal(0, 6, n)
    exam_score = (
        40
        + study_time * 5
        + (attendance - 70) * 0.3
        + (prev_grade - 60) * 0.25
        + family_sup * 4
        + internet * 3
        - absences * 0.8
        + health * 1.5
        + noise
    ).clip(30, 100)

    df = pd.DataFrame({
        "Student_ID":   [f"STU{str(i+1).zfill(4)}" for i in range(n)],
        "Gender":       gender,
        "Grade_Level":  grade_level,
        "School_Type":  school_type,
        "Study_Hours":  study_time,
        "Attendance_%": attendance.astype(float),
        "Prev_Grade":   prev_grade.astype(float),
        "Family_Support": family_sup,
        "Internet_Access": internet,
        "Extra_Curricular": extra_act,
        "Health_Score": health,
        "Absences":     absences,
        "Exam_Score":   exam_score.round(1),
    })

    # Derived columns
    df["At_Risk"] = (df["Exam_Score"] < 50).astype(int)
    df["Risk_Score"] = (
        (100 - df["Attendance_%"]) * 0.35
        + (df["Absences"] / df["Absences"].max()) * 25
        + ((100 - df["Exam_Score"]) / 100) * 30
        + ((5 - df["Health_Score"]) / 5) * 10
    ).clip(0, 100).round(1)

    df["Risk_Level"] = pd.cut(
        df["Risk_Score"],
        bins=[0, 30, 60, 100],
        labels=["Low", "Medium", "High"],
    )
    return df


# ╔══════════════════════════════════════════════════════════╗
#  MODEL TRAINING
# ╔══════════════════════════════════════════════════════════╗
@st.cache_resource
def train_model(df: pd.DataFrame):
    features = [
        "Study_Hours", "Attendance_%", "Prev_Grade",
        "Family_Support", "Internet_Access", "Extra_Curricular",
        "Health_Score", "Absences",
    ]
    X = df[features]
    y = df["At_Risk"]

    le_map = {}
    X_enc  = X.copy()

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_enc)

    X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=.2, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    report   = classification_report(y_te, y_pred, output_dict=True)
    cm       = confusion_matrix(y_te, y_pred)
    auc      = roc_auc_score(y_te, y_prob)
    cv_score = cross_val_score(model, X_sc, y, cv=5, scoring="f1").mean()

    imp = pd.DataFrame({
        "Feature":    features,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    return model, scaler, features, report, cm, auc, cv_score, imp


# ╔══════════════════════════════════════════════════════════╗
#  CHART HELPERS
# ╔══════════════════════════════════════════════════════════╗
MINT   = "#52b788"
AZURE  = "#3b82f6"
AMBER  = "#f59e0b"
DANGER = "#ef4444"
LIGHT_MINT  = "#d4f5e2"
LIGHT_BLUE  = "#dbeafe"
LIGHT_LEMON = "#fef9c3"

def set_chart_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#f8fafc")
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#e2e8f0")
    ax.tick_params(colors="#475569", labelsize=9)
    if title:  ax.set_title(title, fontsize=11, fontweight="bold", color="#1e293b", pad=12)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color="#64748b")
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color="#64748b")
    ax.figure.patch.set_facecolor("#f8fafc")


def chart_risk_donut(df):
    counts = df["Risk_Level"].value_counts()
    colors = [MINT, AMBER, DANGER]
    labels = ["Low", "Medium", "High"]
    vals   = [counts.get(l, 0) for l in labels]

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, _ = ax.pie(
        vals, labels=None, colors=colors,
        startangle=90, wedgeprops=dict(width=.55, edgecolor="#fff", linewidth=2),
    )
    total = sum(vals)
    at_risk = vals[2]
    ax.text(0, 0.1, f"{at_risk}", ha="center", va="center",
            fontsize=26, fontweight="bold", color="#1e293b")
    ax.text(0, -0.25, "At Risk", ha="center", va="center",
            fontsize=10, color="#64748b")
    legend = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=8, frameon=False)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_title("Risk Distribution", fontsize=11, fontweight="bold",
                 color="#1e293b", pad=10)
    plt.tight_layout()
    return fig


def chart_score_dist(df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    low  = df[df["At_Risk"] == 0]["Exam_Score"]
    high = df[df["At_Risk"] == 1]["Exam_Score"]
    ax.hist(low,  bins=20, color=MINT,   alpha=.75, label="Not At Risk", edgecolor="#fff")
    ax.hist(high, bins=20, color=DANGER, alpha=.75, label="At Risk",     edgecolor="#fff")
    ax.axvline(50, color="#f59e0b", linestyle="--", linewidth=1.8, label="Threshold (50)")
    set_chart_style(ax, "Exam Score Distribution", "Score", "Students")
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    return fig


def chart_attendance_vs_score(df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = df["At_Risk"].map({0: MINT, 1: DANGER})
    ax.scatter(df["Attendance_%"], df["Exam_Score"],
               c=colors, alpha=.5, s=18, edgecolors="none")
    ax.axhline(50, color=AMBER, linestyle="--", linewidth=1.5, alpha=.8)
    set_chart_style(ax, "Attendance vs Exam Score", "Attendance (%)", "Exam Score")
    p1 = mpatches.Patch(color=MINT,   label="Safe")
    p2 = mpatches.Patch(color=DANGER, label="At Risk")
    ax.legend(handles=[p1, p2], fontsize=8, frameon=False)
    plt.tight_layout()
    return fig


def chart_study_box(df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    groups = [df[df["Study_Hours"] == h]["Exam_Score"].values for h in range(1, 6)]
    bp = ax.boxplot(groups, patch_artist=True, notch=False,
                    medianprops=dict(color="#fff", linewidth=2),
                    whiskerprops=dict(color="#94a3b8"),
                    capprops=dict(color="#94a3b8"),
                    flierprops=dict(marker="o", color=AMBER, alpha=.4, markersize=3))
    palette = [LIGHT_BLUE, "#bfdbfe", LIGHT_MINT, MINT, "#22c55e"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
    ax.set_xticklabels([f"{h}h" for h in range(1, 6)])
    set_chart_style(ax, "Study Hours vs Exam Score", "Daily Study Hours", "Exam Score")
    plt.tight_layout()
    return fig


def chart_feature_importance(imp_df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = [AZURE if v > imp_df["Importance"].median() else LIGHT_BLUE
              for v in imp_df["Importance"]]
    bars = ax.barh(imp_df["Feature"], imp_df["Importance"],
                   color=colors, edgecolor="none", height=.6)
    for bar, val in zip(bars, imp_df["Importance"]):
        ax.text(val + .002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8, color="#475569")
    set_chart_style(ax, "Feature Importance", "Importance Score", "")
    plt.tight_layout()
    return fig


def chart_heatmap(df):
    cols = ["Study_Hours", "Attendance_%", "Prev_Grade",
            "Absences", "Health_Score", "Exam_Score"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    cmap = sns.diverging_palette(145, 220, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                linewidths=.5, linecolor="#e2e8f0",
                annot_kws={"size": 8}, ax=ax,
                cbar_kws={"shrink": .75})
    ax.set_title("Correlation Matrix", fontsize=11, fontweight="bold",
                 color="#1e293b", pad=12)
    fig.patch.set_facecolor("#f8fafc")
    plt.tight_layout()
    return fig


def chart_grade_risk(df):
    pivot = (df.groupby(["Grade_Level", "Risk_Level"])
               .size().unstack(fill_value=0)
               .reindex(columns=["Low","Medium","High"], fill_value=0))
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    pivot.plot(kind="bar", ax=ax, color=[MINT, AMBER, DANGER],
               edgecolor="none", width=.65)
    set_chart_style(ax, "Risk by Grade Level", "Grade", "Students")
    ax.legend(title="Risk", fontsize=8, frameon=False, title_fontsize=8)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    return fig


def chart_confusion(cm):
    fig, ax = plt.subplots(figsize=(3.5, 3))
    cmap = sns.light_palette(MINT, as_cmap=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=["Safe","At Risk"],
                yticklabels=["Safe","At Risk"],
                linewidths=1, linecolor="#fff",
                annot_kws={"size": 14, "fontweight": "bold"}, ax=ax,
                cbar=False)
    ax.set_xlabel("Predicted", fontsize=9, color="#64748b")
    ax.set_ylabel("Actual",    fontsize=9, color="#64748b")
    ax.set_title("Confusion Matrix", fontsize=11, fontweight="bold",
                 color="#1e293b", pad=10)
    fig.patch.set_facecolor("#f8fafc")
    plt.tight_layout()
    return fig


# ╔══════════════════════════════════════════════════════════╗
#  SIDEBAR
# ╔══════════════════════════════════════════════════════════╗
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding-bottom:1rem;'>
      <div style='font-size:2.2rem;'>🎓</div>
      <div style='font-size:1.1rem; font-weight:700; color:#1a6b45;'>EduRisk</div>
      <div style='font-size:.75rem; color:#64748b;'>Student Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Dashboard Settings**")
    n_students = st.slider("Number of Students", 100, 1000, 500, 50)
    seed       = st.number_input("Random Seed", 1, 999, 42)

    st.markdown("---")
    st.markdown("**🔍 Filter Data**")
    df_raw = generate_data(n_students, seed)

    grade_filter = st.multiselect(
        "Grade Level",
        options=df_raw["Grade_Level"].unique().tolist(),
        default=df_raw["Grade_Level"].unique().tolist(),
    )
    risk_filter = st.multiselect(
        "Risk Level",
        options=["Low","Medium","High"],
        default=["Low","Medium","High"],
    )
    school_filter = st.multiselect(
        "School Type",
        options=df_raw["School_Type"].unique().tolist(),
        default=df_raw["School_Type"].unique().tolist(),
    )

    st.markdown("---")
    page = st.radio(
        "📌 Navigate",
        ["📊 Overview", "📈 Deep Analysis", "🤖 ML Model", "🔮 Predict Student"],
        label_visibility="collapsed",
    )

# Apply filters
df = df_raw[
    df_raw["Grade_Level"].isin(grade_filter) &
    df_raw["Risk_Level"].isin(risk_filter) &
    df_raw["School_Type"].isin(school_filter)
].copy()

# Train model on full data
model, scaler, features, report, cm, auc, cv_score, imp_df = train_model(df_raw)


# ╔══════════════════════════════════════════════════════════╗
#  HERO HEADER
# ╔══════════════════════════════════════════════════════════╗
st.markdown(f"""
<div class="hero-bar">
  <div>
    <div class="hero-title">🎓 EduRisk Dashboard</div>
    <div class="hero-sub">Student Performance & Risk Prediction · {len(df)} students loaded</div>
  </div>
  <div style="display:flex; gap:.7rem; flex-wrap:wrap;">
    <div class="hero-badge">📅 Real-Time Analytics</div>
    <div class="hero-badge">🤖 ML Powered</div>
    <div class="hero-badge">🎯 AUC {auc:.2f}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "📊 Overview":
    # KPI cards
    total    = len(df)
    at_risk  = int(df["At_Risk"].sum())
    avg_att  = df["Attendance_%"].mean()
    avg_exam = df["Exam_Score"].mean()

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card kpi-blue">
        <div class="kpi-label">Total Students</div>
        <div class="kpi-value" style="color:#1d4ed8">{total}</div>
        <div class="kpi-delta">📚 Active records</div>
      </div>
      <div class="kpi-card kpi-red">
        <div class="kpi-label">At Risk</div>
        <div class="kpi-value" style="color:#b91c1c">{at_risk}</div>
        <div class="kpi-delta">⚠️ Need intervention ({at_risk/total*100:.0f}%)</div>
      </div>
      <div class="kpi-card kpi-green">
        <div class="kpi-label">Avg Attendance</div>
        <div class="kpi-value" style="color:#166534">{avg_att:.1f}%</div>
        <div class="kpi-delta">📅 Across all grades</div>
      </div>
      <div class="kpi-card kpi-yellow">
        <div class="kpi-label">Avg Exam Score</div>
        <div class="kpi-value" style="color:#92400e">{avg_exam:.1f}</div>
        <div class="kpi-delta">📝 Class average</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Charts row 1
    st.markdown('<div class="sec-head">📊 Risk & Score Overview</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.3, 1.3])

    with c1:
        st.pyplot(chart_risk_donut(df), use_container_width=True)
    with c2:
        st.pyplot(chart_score_dist(df), use_container_width=True)
    with c3:
        st.pyplot(chart_grade_risk(df), use_container_width=True)

    # Charts row 2
    st.markdown('<div class="sec-head">🔬 Performance Drivers</div>', unsafe_allow_html=True)
    c4, c5 = st.columns(2)
    with c4:
        st.pyplot(chart_attendance_vs_score(df), use_container_width=True)
    with c5:
        st.pyplot(chart_study_box(df), use_container_width=True)

    # Quick insights
    st.markdown('<div class="sec-head">💡 Auto-Generated Insights</div>', unsafe_allow_html=True)
    low_att = df[df["Attendance_%"] < 60]
    high_abs = df[df["Absences"] > 15]
    no_fam = df[(df["Family_Support"] == 0) & (df["At_Risk"] == 1)]

    i1 = f"<strong>{len(low_att)}</strong> students have attendance below 60% — the strongest predictor of failure."
    i2 = f"Students studying <strong>4+ hours/day</strong> score on average <strong>{df[df['Study_Hours']>=4]['Exam_Score'].mean():.1f}</strong> vs <strong>{df[df['Study_Hours']<4]['Exam_Score'].mean():.1f}</strong> for those who study less."
    i3 = f"<strong>{len(no_fam)}</strong> at-risk students lack family support — targeted counseling could help."
    i4 = f"<strong>{len(high_abs)}</strong> students have 15+ absences; they average <strong>{high_abs['Exam_Score'].mean():.1f}</strong> exam score."

    for ins in [i1, i2, i3, i4]:
        st.markdown(f'<div class="insight-pill">{ins}</div>', unsafe_allow_html=True)

    # Top at-risk students table
    st.markdown('<div class="sec-head">🚨 Highest-Risk Students</div>', unsafe_allow_html=True)
    top_risk = (
        df[df["At_Risk"] == 1]
        .sort_values("Risk_Score", ascending=False)
        .head(10)[["Student_ID","Grade_Level","Attendance_%","Study_Hours","Exam_Score","Risk_Score","Risk_Level"]]
        .reset_index(drop=True)
    )

    def color_risk(val):
        if val == "High":   return "🔴 High"
        if val == "Medium": return "🟡 Medium"
        return "🟢 Low"

    top_risk["Risk_Level"] = top_risk["Risk_Level"].apply(color_risk)
    st.dataframe(
        top_risk,
        use_container_width=True,
        column_config={
            "Risk_Score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100, format="%.1f"),
            "Attendance_%": st.column_config.ProgressColumn("Attendance %", min_value=0, max_value=100, format="%.1f"),
            "Exam_Score": st.column_config.NumberColumn("Exam Score", format="%.1f"),
        },
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════
#  PAGE: DEEP ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "📈 Deep Analysis":
    st.markdown('<div class="sec-head">🔭 Correlation Analysis</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.pyplot(chart_heatmap(df), use_container_width=True)
    with c2:
        st.markdown("**📌 Key Correlations**")
        corr = df[["Study_Hours","Attendance_%","Prev_Grade","Absences","Health_Score","Exam_Score"]].corr()["Exam_Score"].drop("Exam_Score").sort_values(ascending=False)
        for feat, val in corr.items():
            direction = "↑ Positive" if val > 0 else "↓ Negative"
            bar_color = MINT if val > 0 else "#fca5a5"
            bar_w     = abs(val) * 100
            st.markdown(f"""
            <div style='margin-bottom:.6rem;'>
              <div style='display:flex; justify-content:space-between; font-size:.85rem; margin-bottom:.2rem;'>
                <span style='font-weight:600;'>{feat}</span>
                <span style='color:{"#166534" if val > 0 else "#b91c1c"};'>{direction} ({val:.2f})</span>
              </div>
              <div style='background:#e2e8f0; border-radius:50px; height:6px;'>
                <div style='background:{bar_color}; width:{bar_w:.0f}%; height:100%; border-radius:50px;'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Breakdown by gender
    st.markdown('<div class="sec-head">👥 Demographic Breakdown</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        gender_risk = df.groupby("Gender")["Risk_Score"].mean().sort_values()
        ax.barh(gender_risk.index, gender_risk.values,
                color=[MINT, AZURE, AMBER], edgecolor="none", height=.5)
        set_chart_style(ax, "Avg Risk Score by Gender", "Risk Score", "")
        for v, n in zip(gender_risk.values, gender_risk.index):
            ax.text(v + .3, list(gender_risk.index).index(n), f"{v:.1f}",
                    va="center", fontsize=9, color="#475569")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with c4:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        st_risk = df.groupby("School_Type")["Exam_Score"].agg(["mean","std"])
        x = np.arange(len(st_risk))
        ax.bar(x, st_risk["mean"], yerr=st_risk["std"],
               color=[MINT, AZURE], width=.4, edgecolor="none",
               error_kw=dict(ecolor="#94a3b8", capsize=5))
        ax.set_xticks(x); ax.set_xticklabels(st_risk.index)
        set_chart_style(ax, "Exam Score: Public vs Private", "School Type", "Score (mean ± std)")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # Family support impact
    st.markdown('<div class="sec-head">🏠 Support & Lifestyle Factors</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)

    with c5:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        cats  = ["No Family\nSupport", "Family\nSupport", "No Internet", "Has Internet",
                 "No Extra\nActivities", "Extra\nActivities"]
        means = [
            df[df["Family_Support"]==0]["Exam_Score"].mean(),
            df[df["Family_Support"]==1]["Exam_Score"].mean(),
            df[df["Internet_Access"]==0]["Exam_Score"].mean(),
            df[df["Internet_Access"]==1]["Exam_Score"].mean(),
            df[df["Extra_Curricular"]==0]["Exam_Score"].mean(),
            df[df["Extra_Curricular"]==1]["Exam_Score"].mean(),
        ]
        colors = [DANGER, MINT, DANGER, MINT, AMBER, MINT]
        ax.bar(range(6), means, color=colors, edgecolor="none", width=.6)
        ax.set_xticks(range(6)); ax.set_xticklabels(cats, fontsize=7.5)
        ax.axhline(50, color=AMBER, linestyle="--", linewidth=1.2, alpha=.6)
        set_chart_style(ax, "Support Factors vs Exam Score", "", "Avg Exam Score")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with c6:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        health_data = [df[df["Health_Score"]==h]["Exam_Score"].values for h in range(1,6)]
        bp = ax.violinplot(health_data, positions=range(1,6), showmedians=True,
                           showextrema=False)
        for i, body in enumerate(bp["bodies"]):
            body.set_facecolor([DANGER, "#fca5a5", AMBER, LIGHT_MINT, MINT][i])
            body.set_alpha(.7); body.set_edgecolor("none")
        bp["cmedians"].set_color("#1e293b"); bp["cmedians"].set_linewidth(2)
        set_chart_style(ax, "Health Score vs Exam Performance", "Health Score (1=Poor, 5=Excellent)", "Exam Score")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # Full data explorer
    st.markdown('<div class="sec-head">🗃️ Data Explorer</div>', unsafe_allow_html=True)
    st.dataframe(
        df.drop(columns=["At_Risk","Risk_Score","Risk_Level"]).head(50),
        use_container_width=True, hide_index=True,
    )
    csv = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download Full Dataset (CSV)", csv,
                       "student_data.csv", "text/csv")


# ══════════════════════════════════════════════════════════
#  PAGE: ML MODEL
# ══════════════════════════════════════════════════════════
elif page == "🤖 ML Model":
    st.markdown('<div class="sec-head">🤖 Model Performance</div>', unsafe_allow_html=True)

    # Metric chips
    prec = report["1"]["precision"]
    rec  = report["1"]["recall"]
    f1   = report["1"]["f1-score"]
    acc  = report["accuracy"]

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-chip">🎯 Accuracy <b>{acc*100:.1f}%</b></div>
      <div class="metric-chip">📐 Precision <b>{prec*100:.1f}%</b></div>
      <div class="metric-chip">🔍 Recall <b>{rec*100:.1f}%</b></div>
      <div class="metric-chip">⚖️ F1 Score <b>{f1*100:.1f}%</b></div>
      <div class="metric-chip">📈 AUC-ROC <b>{auc:.3f}</b></div>
      <div class="metric-chip">🔄 CV F1 <b>{cv_score*100:.1f}%</b></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(chart_confusion(cm), use_container_width=True)
    with c2:
        st.pyplot(chart_feature_importance(imp_df), use_container_width=True)

    # Model details
    st.markdown('<div class="sec-head">📋 Model Architecture</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:#fff; border-radius:14px; padding:1.4rem; border:1.5px solid #e2e8f0;'>
          <div style='font-weight:700; margin-bottom:.8rem; color:#1e293b;'>🌲 Random Forest Classifier</div>
          <ul style='margin:0; padding-left:1.2rem; font-size:.88rem; color:#475569; line-height:2;'>
            <li>200 decision trees</li>
            <li>Max depth: 8 levels</li>
            <li>Class-balanced weights</li>
            <li>Feature scaling: StandardScaler</li>
            <li>Train/Test split: 80/20</li>
            <li>Cross-validation: 5-fold</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style='background:#fff; border-radius:14px; padding:1.4rem; border:1.5px solid #e2e8f0;'>
          <div style='font-weight:700; margin-bottom:.8rem; color:#1e293b;'>🎯 Target Variable</div>
          <ul style='margin:0; padding-left:1.2rem; font-size:.88rem; color:#475569; line-height:2;'>
            <li><strong>At Risk</strong>: Exam Score &lt; 50</li>
            <li>Binary classification task</li>
            <li>Features: 8 predictors</li>
            <li>No missing values (synthetic data)</li>
            <li>Balanced class weighting applied</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    # Feature importance table
    st.markdown('<div class="sec-head">📊 Feature Importance Ranking</div>', unsafe_allow_html=True)
    imp_display = imp_df.copy()
    imp_display["Rank"] = range(1, len(imp_display)+1)
    imp_display["Importance %"] = (imp_display["Importance"] * 100).round(2)
    imp_display["Visual"] = imp_display["Importance %"]
    st.dataframe(
        imp_display[["Rank","Feature","Importance %","Visual"]],
        column_config={
            "Visual": st.column_config.ProgressColumn("Importance Bar", min_value=0, max_value=100, format="%.1f%%"),
        },
        hide_index=True, use_container_width=True,
    )


# ══════════════════════════════════════════════════════════
#  PAGE: PREDICT STUDENT
# ══════════════════════════════════════════════════════════
elif page == "🔮 Predict Student":
    st.markdown('<div class="sec-head">🔮 Individual Student Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown("Enter a student's profile below to get an instant risk assessment.")

    with st.container():
        st.markdown('<div class="pred-card">', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            study_h  = st.slider("📚 Daily Study Hours", 1, 5, 3)
            att_pct  = st.slider("📅 Attendance (%)",    40, 100, 75)
            prev_g   = st.slider("📝 Previous Grade",    40, 100, 65)
        with c2:
            absences = st.slider("🚫 Absences (days)", 0, 25, 5)
            health   = st.slider("❤️ Health Score (1–5)", 1, 5, 3)
        with c3:
            fam_sup  = st.selectbox("🏠 Family Support",     ["Yes","No"])
            internet = st.selectbox("🌐 Internet Access",    ["Yes","No"])
            extra    = st.selectbox("🎭 Extra Curricular",   ["Yes","No"])

        predict_btn = st.button("⚡ Run Risk Assessment")
        st.markdown('</div>', unsafe_allow_html=True)

    if predict_btn:
        inp = np.array([[
            study_h, att_pct, prev_g,
            1 if fam_sup=="Yes" else 0,
            1 if internet=="Yes" else 0,
            1 if extra=="Yes" else 0,
            health, absences,
        ]])
        inp_sc = scaler.transform(inp)
        pred   = model.predict(inp_sc)[0]
        prob   = model.predict_proba(inp_sc)[0][1]

        # Risk score
        risk_score = (
            (100 - att_pct) * 0.35
            + (absences / 25) * 25
            + prob * 30
            + ((5 - health) / 5) * 10
        )
        risk_score = min(100, max(0, risk_score))

        if risk_score > 60:
            badge = "risk-high";   label = "🔴 HIGH RISK";   color = "#b91c1c"
        elif risk_score > 30:
            badge = "risk-medium"; label = "🟡 MEDIUM RISK"; color = "#c2410c"
        else:
            badge = "risk-low";    label = "🟢 LOW RISK";    color = "#166534"

        st.markdown(f"""
        <div style='background:#fff; border-radius:18px; padding:2rem; margin-top:1.2rem;
                    border:1.5px solid #e2e8f0; box-shadow:0 4px 20px rgba(0,0,0,.08);'>
          <div style='display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap;'>
            <div style='font-size:3.5rem; line-height:1;'>
              {"⚠️" if pred==1 else "✅"}
            </div>
            <div>
              <div style='font-size:1.5rem; font-weight:800; color:{color};'>{label}</div>
              <div style='font-size:.9rem; color:#64748b; margin-top:.3rem;'>
                Failure probability: <strong>{prob*100:.1f}%</strong> &nbsp;·&nbsp;
                Risk score: <strong>{risk_score:.1f}/100</strong>
              </div>
            </div>
          </div>
          <div style='margin-top:1.2rem; background:#f1f5f9; border-radius:10px; padding:1rem;'>
            <div style='font-size:.85rem; color:#475569; margin-bottom:.5rem;'>Risk Score Gauge</div>
            <div style='background:#e2e8f0; border-radius:50px; height:14px; overflow:hidden;'>
              <div style='background:{"#ef4444" if risk_score>60 else "#f97316" if risk_score>30 else "#52b788"};
                          width:{risk_score:.0f}%; height:100%; border-radius:50px;
                          transition: width 1s ease;'></div>
            </div>
            <div style='display:flex; justify-content:space-between; font-size:.75rem; color:#94a3b8; margin-top:.3rem;'>
              <span>Low Risk (0)</span><span>Medium (30)</span><span>High Risk (60+)</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Recommendations
        st.markdown('<div class="sec-head" style="margin-top:1.5rem;">📋 Personalized Recommendations</div>', unsafe_allow_html=True)
        recs = []
        if att_pct < 75:
            recs.append(("📅", "Attendance Alert", f"Student has {att_pct}% attendance. Target: 85%+. Consider attendance monitoring."))
        if study_h < 3:
            recs.append(("📚", "Study Time", f"Only {study_h}h/day. Encourage a structured 3–4hr daily study routine."))
        if absences > 10:
            recs.append(("🚫", "High Absences", f"{absences} absences recorded. Investigate reasons and provide support."))
        if fam_sup == "No":
            recs.append(("🏠", "Family Engagement", "No family support detected. Assign a school counselor or mentor."))
        if health < 3:
            recs.append(("❤️", "Health & Wellbeing", f"Health score {health}/5. Consider wellness resources or counseling."))
        if prev_g < 55:
            recs.append(("📝", "Academic History", f"Previous grade was {prev_g}. Arrange tutoring or remedial sessions."))
        if not recs:
            recs.append(("🌟", "Great Profile!", "This student shows no major risk indicators. Keep up the positive momentum!"))

        for icon, title, desc in recs:
            bg = LIGHT_LEMON if title != "Great Profile!" else LIGHT_MINT
            border = "#fde68a" if title != "Great Profile!" else "#86efac"
            st.markdown(f"""
            <div style='background:{bg}; border:1.5px solid {border}; border-radius:12px;
                        padding:1rem 1.2rem; margin:.4rem 0; display:flex; gap:.8rem; align-items:flex-start;'>
              <span style='font-size:1.3rem;'>{icon}</span>
              <div>
                <div style='font-weight:700; font-size:.9rem; color:#1e293b;'>{title}</div>
                <div style='font-size:.85rem; color:#475569; margin-top:.2rem;'>{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Input summary
        with st.expander("🔎 View Input Summary"):
            summary_df = pd.DataFrame({
                "Feature": ["Study Hours","Attendance","Previous Grade","Absences",
                            "Health Score","Family Support","Internet Access","Extra Curricular"],
                "Value":   [f"{study_h}h/day", f"{att_pct}%", f"{prev_g}/100",
                            f"{absences} days", f"{health}/5", fam_sup, internet, extra],
                "Impact":  ["High","High","High","Medium","Medium","Low","Low","Low"],
            })
            st.dataframe(summary_df, hide_index=True, use_container_width=True)


# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Built with ❤️ using Streamlit · Python · Scikit-learn · Pandas &nbsp;|&nbsp;
  EduRisk Student Analytics Dashboard &nbsp;|&nbsp; 
  Designed for fresh graduate portfolios 🎓
</div>
""", unsafe_allow_html=True)
