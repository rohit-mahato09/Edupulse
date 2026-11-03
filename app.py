from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="EduPulse", page_icon="🎓", layout="wide")

FEATURES = ["attendance", "study_hours", "cgpa", "backlogs", "assignment_pct"]
NICE_NAMES = {
    "attendance": "Attendance (%)",
    "study_hours": "Study Hours / Day",
    "cgpa": "Previous CGPA",
    "backlogs": "Active Backlogs",
    "assignment_pct": "Assignments Submitted (%)",
}


# ---------------------------------------------------------------
# 1. DATA & MODEL
# ---------------------------------------------------------------
@st.cache_data
def generate_demo_data(n=600, seed=42):
    """Synthetic demo data. Replace with your real college CSV when available."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "attendance": rng.normal(72, 15, n).clip(20, 100).round(1),
            "study_hours": rng.normal(3, 1.8, n).clip(0, 12).round(),
            "cgpa": rng.normal(7, 1.3, n).clip(2, 10).round(1),
            "backlogs": rng.poisson(0.8, n).clip(0, 8),
            "assignment_pct": rng.normal(75, 18, n).clip(10, 100).round(1),
        }
    )
    z = (
        -0.05 * (df["attendance"] - 70)
        - 0.25 * (df["study_hours"] - 3)
        - 0.9 * (df["cgpa"] - 6.5)
        + 0.6 * df["backlogs"]
        - 0.03 * (df["assignment_pct"] - 70)
        + rng.normal(0, 0.8, n)
    )
    df["at_risk"] = (z > np.quantile(z, 0.65)).astype(int)
    return df


@st.cache_resource
def train_model(df):
    X, y = df[FEATURES], df["at_risk"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return model, acc, cm


def risk_level(prob):
    if prob < 30:
        return "Low", "success"
    if prob < 60:
        return "Medium", "warning"
    return "High", "error"


def recommendations(row):
    tips = []
    if row["attendance"] < 75:
        tips.append("Attendance is below 75%. Attend regularly; this is usually the biggest lever.")
    if row["study_hours"] < 3:
        tips.append("Self-study is under 3 hrs/day. Build a fixed daily study slot.")
    if row["cgpa"] < 6.5:
        tips.append("CGPA is low. Consider peer tutoring or remedial classes for weak subjects.")
    if row["backlogs"] > 0:
        tips.append(f"{int(row['backlogs'])} active backlog(s). Make a clearance plan for the next exam cycle.")
    if row["assignment_pct"] < 70:
        tips.append("Assignment submission is low. Set weekly deadlines and track them.")
    if not tips:
        tips.append("All indicators look healthy. Keep up the current routine.")
    return tips


TARGETS = {"attendance": 75, "study_hours": 4, "cgpa": 7, "backlogs": 0, "assignment_pct": 80}


def _risk_of(model, r):
    return round(model.predict_proba(pd.DataFrame([r])[FEATURES])[0][1] * 100, 1)


def _fmt(d):
    return d.strftime("%d %b")


def build_plan(level, entry_date, sem_end):
    """Action plan that fits the time actually left in the semester."""
    days_left = (sem_end - entry_date).days
    if days_left <= 0:
        return [
            "The semester end date has passed. Focus on result review, supplementary / backlog exams "
            "and a fresh plan for the next semester from day 1."
        ]

    if days_left < 14:
        mid = entry_date + timedelta(days=max(1, days_left // 2))
        return [
            f"{_fmt(entry_date)} - {_fmt(mid)} (crash phase): Meet faculty/mentor now, submit all pending "
            "assignments and prioritise exam-relevant topics of weak subjects.",
            f"{_fmt(mid)} - {_fmt(sem_end)} (final phase): Revision with previous-year papers and mock tests. "
            "No new topics.",
        ]

    p1_end = entry_date + timedelta(days=max(1, round(days_left * 0.25)))
    p2_end = entry_date + timedelta(days=max(2, round(days_left * 0.70)))

    if level == "High":
        p1 = "Meet mentor / class advisor, find the root causes and start the top priority action immediately."
        p2 = "Daily study timetable, full class attendance, remedial classes and clearing pending assignments."
    elif level == "Medium":
        p1 = "Fix the weakest indicator first (see priority actions)."
        p2 = "Maintain a weekly self-review of attendance, assignments and study hours."
    else:
        p1 = "Review the current routine and set a stretch goal (higher CGPA or a new skill)."
        p2 = "Keep consistency; try group study or mentoring peers."

    return [
        f"{_fmt(entry_date)} - {_fmt(p1_end)} (start): {p1}",
        f"{_fmt(p1_end)} - {_fmt(p2_end)} (build): {p2}",
        f"{_fmt(p2_end)} - {_fmt(sem_end)} (final): Revision, previous-year papers and mock tests.",
        f"Checkpoint on {_fmt(entry_date + timedelta(days=days_left // 2))}: re-run this prediction "
        "and compare the risk score.",
    ]


def smart_advice(row, risk_pct, level, model, entry_date=None, sem_end=None):
    """Builds a full suggestion + recommendation set for ANY predicted result.

    Returns a dict with: summary, priority (list of (text, est. risk drop)),
    plan (list of timeline steps).
    """
    row = {k: float(v) for k, v in row.items()}
    entry_date = entry_date or date.today()
    sem_end = sem_end or (entry_date + timedelta(days=60))
    days_left = (sem_end - entry_date).days

    # Each lever: (condition that it needs work, what-if scenario, advice text)
    levers = [
        (
            row["attendance"] < TARGETS["attendance"],
            {**row, "attendance": min(100, row["attendance"] + 10)},
            f"Raise attendance from {row['attendance']:.0f}% to at least {TARGETS['attendance']}%. "
            "Attend every class, and meet the faculty for any missed topics.",
        ),
        (
            row["study_hours"] < TARGETS["study_hours"],
            {**row, "study_hours": min(12, row["study_hours"] + 2)},
            f"Increase self-study from {row['study_hours']:.0f} to {TARGETS['study_hours']}+ hrs/day. "
            "Fix a daily slot and use a timetable.",
        ),
        (
            row["cgpa"] < TARGETS["cgpa"],
            {**row, "cgpa": min(10, row["cgpa"] + 0.5)},
            f"CGPA is {row['cgpa']:.1f}. Join peer tutoring or remedial classes for the weakest subjects "
            "and revise previous-year papers.",
        ),
        (
            row["backlogs"] > TARGETS["backlogs"],
            {**row, "backlogs": 0},
            f"Clear the {int(row['backlogs'])} active backlog(s). Prepare a subject-wise clearance plan "
            "for the next exam cycle.",
        ),
        (
            row["assignment_pct"] < TARGETS["assignment_pct"],
            {**row, "assignment_pct": min(100, row["assignment_pct"] + 15)},
            f"Assignment submission is {row['assignment_pct']:.0f}%. Set weekly deadlines "
            f"and aim for {TARGETS['assignment_pct']}%+.",
        ),
    ]

    # Rank by how much the model says the risk would drop
    priority = []
    for needs_work, scenario, text in levers:
        if needs_work:
            drop = max(0.0, round(risk_pct - _risk_of(model, scenario), 1))
            priority.append((text, drop))
    priority.sort(key=lambda t: t[1], reverse=True)

    if not priority:
        priority = [("All indicators are healthy. Maintain the current routine and keep consistency.", 0.0)]

    if level == "High":
        summary = (
            f"High risk ({risk_pct}%). The student needs immediate academic intervention. "
            "Start with the top priority action below and review progress within 2 weeks."
        )
    elif level == "Medium":
        summary = (
            f"Medium risk ({risk_pct}%). The student is manageable but can slip further "
            "without timely correction. Focus on the top 1-2 actions."
        )
    else:
        summary = (
            f"Low risk ({risk_pct}%). The student is on a good track. "
            "Keep the habits going and avoid complacency."
        )

    # Time-aware additions
    if days_left > 0:
        summary += f" Time left in the semester: {days_left} days (~{max(1, days_left // 7)} weeks)."
        if days_left < 30 and level in ("High", "Medium"):
            summary += " Very little time is left, so act this week and escalate to the mentor early."
        if days_left < 30 and row["attendance"] < TARGETS["attendance"]:
            summary += (
                " Attendance may not be fully recoverable in the time left. Talk to the faculty/HOD "
                "about attendance condonation rules."
            )
    else:
        summary += " The semester end date has already passed."

    plan = build_plan(level, entry_date, sem_end)

    return {
        "summary": summary,
        "priority": priority,
        "plan": plan,
        "days_left": days_left,
    }


def render_advice(advice):
    """Show the advice in Streamlit."""
    st.subheader("💡 Suggestion & Recommendation")
    st.info(advice["summary"])
    dl = advice.get("days_left")
    if dl is not None:
        st.metric("⏳ Time left in semester", f"{max(dl, 0)} days (~{max(dl, 0) // 7} weeks)")

    st.markdown("**🎯 Priority actions (biggest risk reduction first)**")
    for i, (text, drop) in enumerate(advice["priority"], 1):
        extra = f"  _(est. risk reduction: ~{drop} pts)_" if drop > 0 else ""
        st.write(f"{i}. {text}{extra}")

    st.markdown("**🗓️ Action plan**")
    for step in advice["plan"]:
        st.write("• " + step)


def advice_to_text(advice):
    """Plain-text version for the downloadable report."""
    lines = ["Summary:", advice["summary"], "", "Priority Actions:"]
    for i, (text, drop) in enumerate(advice["priority"], 1):
        extra = f" (est. risk reduction ~{drop} pts)" if drop > 0 else ""
        lines.append(f"{i}. {text}{extra}")
    lines += ["", "Action Plan:"] + ["- " + s for s in advice["plan"]]
    return "\n".join(lines)


# ---------------------------------------------------------------
# 2. SIDEBAR: DATA SOURCE
# ---------------------------------------------------------------
st.title("🎓 EduPulse")
st.subheader("Student Academic Risk Predictor & Analyser")
st.caption("AI-powered early warning system")

df = generate_demo_data()

with st.sidebar:
    st.header("ℹ️ About")
    st.caption(f"Model trained on {len(df)} sample student records.")
    st.caption("Algorithm: Random Forest Classifier")

st.markdown(
    """
    <style>
    .stApp, [data-testid="stHeader"] { background: #000000; }
    [data-testid="stSidebar"] { background: #0d0d0d; }
    .stApp, .stApp p, .stApp label, .stApp span, .stApp li,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    [data-testid="stCaptionContainer"] { color: #f2f2f2; }
    </style>
    """,
    unsafe_allow_html=True,
)

model, accuracy, cm = train_model(df)

if "history" not in st.session_state:
    st.session_state.history = []

tab1, tab2, tab3 = st.tabs(["👤 Single Student", "📂 Batch Prediction", "📈 Model Insights"])

# ---------------------------------------------------------------
# TAB 1: SINGLE STUDENT
# ---------------------------------------------------------------
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        student_name = st.text_input("Student Name", "Rishav")
        attendance = st.slider("Attendance Rate (%)", 0, 100, 75)
        study_hours = st.slider("Daily Self-Study Hours", 0, 12, 3)
    with c2:
        roll_number = st.text_input("Roll / ID Number", "CSE-101")
        past_cgpa = st.slider("Previous CGPA", 0.0, 10.0, 7.2, 0.1)
        backlogs = st.number_input("Active Backlogs", 0, 15, 0)
    assignment_pct = st.slider("Assignments Submitted (%)", 0, 100, 80)

    d1, d2 = st.columns(2)
    entry_date = d1.date_input("📅 Details Uploaded On", date.today(), key="single_entry")
    sem_end = d2.date_input("🏁 Semester End Date", date.today() + timedelta(days=60), key="single_end")
    _left = (sem_end - entry_date).days
    if _left > 0:
        st.caption(f"⏳ {_left} days (~{_left // 7} weeks) left until the semester ends.")
    else:
        st.warning("Semester end date is on or before the upload date. Please check the dates.")

    if st.button("Generate Performance Analysis", type="primary"):
        row = {
            "attendance": attendance,
            "study_hours": study_hours,
            "cgpa": past_cgpa,
            "backlogs": backlogs,
            "assignment_pct": assignment_pct,
        }
        features = pd.DataFrame([row])[FEATURES]
        risk_pct = round(model.predict_proba(features)[0][1] * 100, 1)
        level, kind = risk_level(risk_pct)

        st.divider()
        getattr(st, kind)(f"**Risk Level: {level}** (Risk probability: {risk_pct}%)")
        st.progress(int(risk_pct))

        advice = smart_advice(row, risk_pct, level, model, entry_date, sem_end)
        render_advice(advice)

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**📊 Student vs Recommended Targets**")
            st.caption("For each factor, compare the Student bar with the Target bar.")
            targets = {"attendance": 75, "study_hours": 4, "cgpa": 7, "backlogs": 0, "assignment_pct": 80}
            compare = pd.DataFrame(
                {"Student": row, "Target": targets}
            ).rename(index=NICE_NAMES)
            st.bar_chart(compare)

            needs_work, doing_well = [], []
            for f in FEATURES:
                val, tgt = row[f], targets[f]
                good = val <= tgt if f == "backlogs" else val >= tgt
                line = f"{NICE_NAMES[f]}: {val:g} (target {tgt})"
                (doing_well if good else needs_work).append(line)

            if needs_work:
                st.warning("⚠️ **Needs improvement**\n\n" + "\n".join("- " + x for x in needs_work))
            if doing_well:
                st.success("✅ **Already good**\n\n" + "\n".join("- " + x for x in doing_well))
            st.caption(
                "Attendance and assignments are out of 100, so their bars look taller than CGPA. "
                "That is fine. Just compare Student vs Target for each factor. "
                "For backlogs, lower is better."
            )

        with g2:
            st.markdown("**🔮 What-if: How can risk be reduced?**")
            st.caption("The model checks what happens to the risk if only one change is made at a time.")
            scenarios = {
                "Now (current)": row,
                "Attendance +10%": {**row, "attendance": min(100, attendance + 10)},
                "Study +2 hrs/day": {**row, "study_hours": min(12, study_hours + 2)},
                "Clear backlogs": {**row, "backlogs": 0},
                "Assignments +15%": {**row, "assignment_pct": min(100, assignment_pct + 15)},
                "All changes together": {
                    **row,
                    "attendance": min(100, attendance + 10),
                    "study_hours": min(12, study_hours + 2),
                    "backlogs": 0,
                    "assignment_pct": min(100, assignment_pct + 15),
                },
            }
            risks = [
                round(model.predict_proba(pd.DataFrame([sc])[FEATURES])[0][1] * 100, 1)
                for sc in scenarios.values()
            ]
            labels = list(scenarios.keys())
            now_risk, all_risk = risks[0], risks[-1]
            single = risks[1:-1]
            best_i = int(np.argmin(single))
            best_label, best_risk = labels[1:-1][best_i], single[best_i]

            if best_risk < now_risk:
                st.success(
                    f"✅ **Best single change:** {best_label}. "
                    f"Risk falls from **{now_risk}%** to **{best_risk}%**."
                )
            else:
                st.info(
                    "No single change reduces the risk on its own. "
                    "Try combining a few changes together."
                )

            if all_risk < now_risk:
                st.write(f"🎯 If **all** the changes are done together, risk can fall to **{all_risk}%**.")
            else:
                st.write("Even with all changes together, the risk stays about the same for this student.")

        # Save to session history
        st.session_state.history.append(
            {"Name": student_name, "Roll No": roll_number, **row, "Risk %": risk_pct, "Level": level,
             "Uploaded On": str(entry_date), "Days Left": (sem_end - entry_date).days}
        )

        report = (
            "STUDENT PERFORMANCE REPORT\n"
            "-------------------------\n"
            f"Name: {student_name}\nRoll No: {roll_number}\n"
            f"Details Uploaded On: {entry_date}\nSemester End Date: {sem_end} "
            f"({(sem_end - entry_date).days} days left)\n"
            f"Attendance: {attendance}%\nStudy Hours: {study_hours} hrs/day\n"
            f"CGPA: {past_cgpa}\nBacklogs: {backlogs}\nAssignments: {assignment_pct}%\n"
            f"Risk Level: {level} ({risk_pct}%)\n\n"
            + advice_to_text(advice)
        )
        st.download_button(
            "📥 Download Report (.txt)", report, file_name=f"report_{student_name}.txt"
        )

    if st.session_state.history:
        st.divider()
        st.subheader("🕘 Session History")
        hist = pd.DataFrame(st.session_state.history)
        st.dataframe(hist, use_container_width=True)
        st.download_button(
            "📥 Download History (.csv)",
            hist.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------
# TAB 2: CLASS / BATCH PREDICTION (type directly, no CSV)
# ---------------------------------------------------------------
with tab2:
    st.write("Enter several students in the table below (click + to add rows).")
    b1, b2 = st.columns(2)
    b_entry = b1.date_input("📅 Details Uploaded On", date.today(), key="batch_entry")
    b_end = b2.date_input("🏁 Semester End Date", date.today() + timedelta(days=60), key="batch_end")
    default_rows = pd.DataFrame(
        {
            "Name": ["Rishav", "Aman", "Priya"],
            "attendance": [75.0, 45.0, 90.0],
            "study_hours": [3, 1, 5],
            "cgpa": [7.2, 4.5, 9.0],
            "backlogs": [0, 3, 0],
            "assignment_pct": [80.0, 40.0, 95.0],
        }
    )
    edited = st.data_editor(default_rows, num_rows="dynamic", use_container_width=True)

    if st.button("Predict for All Students", type="primary"):
        batch = edited.dropna(subset=FEATURES).copy()
        if batch.empty:
            st.warning("Please fill in at least one complete row.")
        else:
            batch["Risk %"] = (model.predict_proba(batch[FEATURES])[:, 1] * 100).round(1)
            batch["Level"] = batch["Risk %"].apply(lambda p: risk_level(p)[0])
            batch = batch.sort_values("Risk %", ascending=False)

            # Suggestion for every student
            advices = []
            for _, r in batch.iterrows():
                feats = {f: r[f] for f in FEATURES}
                advices.append(smart_advice(feats, r["Risk %"], r["Level"], model, b_entry, b_end))
            batch["Top Suggestion"] = [a["priority"][0][0] for a in advices]

            m1, m2, m3 = st.columns(3)
            m1.metric("Students", len(batch))
            m2.metric("High risk", int((batch["Level"] == "High").sum()))
            m3.metric("Average risk", f"{batch['Risk %'].mean():.1f}%")

            st.bar_chart(batch.set_index("Name")["Risk %"])
            st.dataframe(batch, use_container_width=True)

            st.subheader("💡 Suggestions per Student")
            for (_, r), adv in zip(batch.iterrows(), advices):
                with st.expander(f"{r['Name']}: {r['Level']} risk ({r['Risk %']}%)"):
                    render_advice(adv)

# ---------------------------------------------------------------
# TAB 3: MODEL INSIGHTS
# ---------------------------------------------------------------
with tab3:
    st.subheader("📈 How good is this model? (in simple words)")
    st.caption(
        "To check the model fairly, we kept 20% of the students hidden while it was learning. "
        "After learning, the model was tested on those hidden students, like a surprise exam."
    )

    tn, fp, fn, tp = cm.ravel()
    total = int(cm.sum())
    correct = int(tn + tp)
    real_risk = int(tp + fn)
    real_safe = int(tn + fp)
    caught_pct = (tp / real_risk * 100) if real_risk else 0.0

    # ---------- 1. Accuracy ----------
    st.markdown("### 1️⃣ Accuracy: how often is the model right?")
    st.metric("Test Accuracy", f"{accuracy * 100:.1f}%")
    st.write(
        f"In the test, the model looked at **{total} students** and gave the correct answer "
        f"for **{correct}** of them."
    )
    st.caption("Higher is better. 90% means about 9 out of every 10 predictions are right.")

    st.divider()

    # ---------- 2. Feature importance ----------
    st.markdown("### 2️⃣ Important factors: what matters most?")
    imp = pd.Series(model.feature_importances_, index=[NICE_NAMES[f] for f in FEATURES])
    imp = imp.sort_values(ascending=False)
    st.bar_chart(imp)
    st.write(
        f"The model depends most on **{imp.index[0]}** ({imp.iloc[0] * 100:.0f}% of its decision), "
        f"then **{imp.index[1]}** ({imp.iloc[1] * 100:.0f}%). "
        f"**{imp.index[-1]}** matters the least ({imp.iloc[-1] * 100:.0f}%)."
    )
    st.caption("A taller bar means that factor has a bigger effect on whether a student is at risk.")

    st.divider()

    # ---------- 3. Confusion matrix ----------
    st.markdown("### 3️⃣ Right and wrong answers: where did the model make mistakes?")
    k1, k2 = st.columns(2)
    k1.metric("✅ Safe students found safe", int(tn))
    k2.metric("✅ At-risk students caught", int(tp))
    k3, k4 = st.columns(2)
    k3.metric("⚠️ False alarms", int(fp), help="Student was safe, but the model said at risk")
    k4.metric("🚨 At-risk students missed", int(fn), help="Student was at risk, but the model said safe")

    st.write(
        f"Out of **{real_risk}** students who were really at risk, the model caught **{int(tp)}** "
        f"({caught_pct:.0f}%). Out of **{real_safe}** safe students, it wrongly flagged **{int(fp)}**."
    )
    st.caption(
        "A false alarm is a small problem: the student just gets extra attention. "
        "A missed at-risk student is the bigger problem, so we want that number to be low."
    )

    with st.expander("See the full table"):
        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual: Safe", "Actual: At Risk"],
                columns=["Model said: Safe", "Model said: At Risk"],
            )
        )

    st.divider()
    st.info(
        f"In short: the model is right about {accuracy * 100:.0f}% of the time, "
        f"and {imp.index[0]} is the factor it trusts the most."
    )
    st.warning(
        "Note: the students in this demo are computer-generated (not real). "
        "So these numbers are for demonstration. With real college data, they will change."
    )