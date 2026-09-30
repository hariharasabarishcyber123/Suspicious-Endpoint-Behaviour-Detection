import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Endpoint Threat Detection",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #777;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .risk-high {
        background-color: #ffdddd;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }

    .risk-low {
        background-color: #ddffdd;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ Suspicious Endpoint Behaviour Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Cybersecurity Monitoring System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

DATA_FILE = "data/predictions.csv"

try:

    df = pd.read_csv(DATA_FILE)

except FileNotFoundError:

    st.error(
        "predictions.csv not found. Run: python predict.py"
    )

    st.stop()


# ============================================================
# BASIC CLEANING
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)


# ============================================================
# METRICS
# ============================================================

total_events = len(df)

suspicious_events = (
    df["prediction"] == "Suspicious"
).sum()

normal_events = (
    df["prediction"] == "Normal"
).sum()

suspicious_percentage = (
    suspicious_events / total_events * 100
)


# ============================================================
# TOP METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Events",
        f"{total_events:,}"
    )


with col2:

    st.metric(
        "🚨 Suspicious",
        f"{suspicious_events:,}"
    )


with col3:

    st.metric(
        "✅ Normal",
        f"{normal_events:,}"
    )


with col4:

    st.metric(
        "Suspicious Rate",
        f"{suspicious_percentage:.1f}%"
    )


st.divider()


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.title("🔎 Event Filters")


# User filter

users = sorted(
    df["user"]
    .dropna()
    .astype(str)
    .unique()
)

selected_user = st.sidebar.selectbox(
    "User",
    ["All"] + users
)


# PC filter

pcs = sorted(
    df["pc"]
    .dropna()
    .astype(str)
    .unique()
)

selected_pc = st.sidebar.selectbox(
    "Endpoint / PC",
    ["All"] + pcs
)


# Prediction filter

selected_status = st.sidebar.selectbox(
    "Detection Status",
    [
        "All",
        "Suspicious",
        "Normal"
    ]
)


# Risk filter

selected_risk = st.sidebar.selectbox(
    "Risk Level",
    [
        "All",
        "HIGH",
        "MEDIUM",
        "LOW"
    ]
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()


if selected_user != "All":

    filtered_df = filtered_df[
        filtered_df["user"].astype(str)
        == selected_user
    ]


if selected_pc != "All":

    filtered_df = filtered_df[
        filtered_df["pc"].astype(str)
        == selected_pc
    ]


if selected_status != "All":

    filtered_df = filtered_df[
        filtered_df["prediction"]
        == selected_status
    ]


if selected_risk != "All":

    filtered_df = filtered_df[
        filtered_df["risk_level"]
        == selected_risk
    ]


# ============================================================
# CHART SECTION
# ============================================================

left, right = st.columns(2)


# ------------------------------------------------------------
# Prediction distribution
# ------------------------------------------------------------

with left:

    st.subheader(
        "📊 Detection Distribution"
    )

    status_counts = (
        df["prediction"]
        .value_counts()
        .reset_index()
    )

    status_counts.columns = [
        "Status",
        "Count"
    ]

    fig = px.pie(
        status_counts,
        names="Status",
        values="Count",
        hole=0.45
    )

    fig.update_layout(
        height=350,
        showlegend=True
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ------------------------------------------------------------
# Risk distribution
# ------------------------------------------------------------

with right:

    st.subheader(
        "🚨 Risk Distribution"
    )

    risk_counts = (
        df["risk_level"]
        .value_counts()
        .reindex(
            ["HIGH", "MEDIUM", "LOW"],
            fill_value=0
        )
        .reset_index()
    )

    risk_counts.columns = [
        "Risk Level",
        "Count"
    ]

    fig2 = px.bar(
        risk_counts,
        x="Risk Level",
        y="Count",
        text="Count"
    )

    fig2.update_layout(
        height=350
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ============================================================
# ACTIVITY ANALYSIS
# ============================================================

st.subheader(
    "📈 Activity Analysis"
)

activity_counts = (
    df.groupby(
        ["activity", "prediction"]
    )
    .size()
    .reset_index(
        name="count"
    )
)

fig3 = px.bar(
    activity_counts,
    x="activity",
    y="count",
    color="prediction",
    barmode="group"
)

fig3.update_layout(
    height=450,
    xaxis_tickangle=-45
)

st.plotly_chart(
    fig3,
    use_container_width=True
)


# ============================================================
# SUSPICIOUS EVENTS
# ============================================================

st.subheader(
    "🚨 Suspicious Events"
)

suspicious_df = df[
    df["prediction"] == "Suspicious"
].copy()


display_columns = [
    "timestamp",
    "user",
    "pc",
    "activity",
    "resource",
    "risk_score",
    "risk_level"
]


st.dataframe(
    suspicious_df[
        display_columns
    ]
    .sort_values(
        "risk_score",
        ascending=False
    )
    .head(100),
    use_container_width=True,
    height=400
)


# ============================================================
# EVENT INVESTIGATION
# ============================================================

st.divider()

st.subheader(
    "🔍 Event Investigation"
)

if len(filtered_df) == 0:

    st.warning(
        "No events match the selected filters."
    )

else:

    # Give each row an index for selection

    event_options = filtered_df.index.tolist()

    selected_event = st.selectbox(
        "Select an event to investigate",
        event_options
    )

    event = df.loc[selected_event]


    # --------------------------------------------------------
    # Event details
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.write("### Endpoint")

        st.write(
            f"**PC:** {event['pc']}"
        )

        st.write(
            f"**User:** {event['user']}"
        )


    with col2:

        st.write("### Activity")

        st.write(
            f"**Type:** {event['activity']}"
        )

        st.write(
            f"**File:** {event['file']}"
        )


    with col3:

        st.write("### Detection")

        score = float(
            event["risk_score"]
        )

        st.metric(
            "Risk Score",
            f"{score:.0f}/100"
        )

        st.write(
            f"**Status:** {event['prediction']}"
        )

        st.write(
            f"**Risk:** {event['risk_level']}"
        )


    # --------------------------------------------------------
    # Risk visualization
    # --------------------------------------------------------

    st.write("### Risk Score")

    st.progress(
        min(
            max(score / 100, 0),
            1
        )
    )


    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    st.write(
        "### 🧠 Detection Explanation"
    )


    reasons = []


    activity = str(
        event.get(
            "activity",
            ""
        )
    ).lower()


    file_type = str(
        event.get(
            "file",
            ""
        )
    ).lower()


    if activity == "connect":

        reasons.append(
            "Network connection activity detected."
        )


    if activity == "disconnect":

        reasons.append(
            "Network disconnection activity detected."
        )


    if activity == "logon":

        reasons.append(
            "User logon activity detected."
        )


    if activity == "logoff":

        reasons.append(
            "User logoff activity detected."
        )


    if activity == "send email":

        reasons.append(
            "Email transmission activity detected."
        )


    if activity == "consult http":

        reasons.append(
            "HTTP/web resource access detected."
        )


    if file_type == "http":

        reasons.append(
            "HTTP event type detected."
        )


    if file_type == "email":

        reasons.append(
            "Email event type detected."
        )


    if pd.notna(
        event.get("resource")
    ):

        reasons.append(
            "A specific resource was associated with this event."
        )


    if not reasons:

        reasons.append(
            "No additional rule-based indicators were identified."
        )


    for reason in reasons:

        st.write(
            f"• {reason}"
        )


# ============================================================
# DATA DOWNLOAD
# ============================================================

st.divider()

st.subheader(
    "📥 Export Results"
)


csv_data = df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="Download Detection Results",
    data=csv_data,
    file_name="endpoint_detection_results.csv",
    mime="text/csv"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ML-Based Suspicious Endpoint Behaviour Detection | "
    "Academic Cybersecurity Hackathon Prototype"
)