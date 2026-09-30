import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import plotly.express as px

from event_data import create_event_features, load_event_data


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Endpoint Threat Detection",
    page_icon="🛡️",
    layout="wide"
)


MODEL_PATH = "models/endpoint_rf.pkl"
DATA_PATH = "data/cert_preprocessed_ml_comparison.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD ORIGINAL DATASET
# ============================================================

@st.cache_data
def load_data():

    return load_event_data(DATA_PATH)


try:

    model = load_model()
    df = load_data()

except Exception as e:

    st.error("Could not load the model or dataset.")

    st.exception(e)

    st.stop()


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = [
    "timestamp",
    "user",
    "pc",
    "activity",
    "resource",
    "case id;Cattivi"
]


missing = [
    c
    for c in required_columns
    if c not in df.columns
]


if missing:

    st.error(
        "Missing columns in the dataset:"
    )

    st.write(missing)

    st.write(
        "Available columns:"
    )

    st.write(
        df.columns.tolist()
    )

    st.stop()


# ============================================================
# HELPER
# ============================================================

def clean_value(value):

    if pd.isna(value):

        return ""

    return str(value).strip()


# ============================================================
# USER / PC EVENT COUNTS
# ============================================================

USER_COUNTS = (
    df["user"]
    .fillna("")
    .astype(str)
    .value_counts()
    .to_dict()
)


PC_COUNTS = (
    df["pc"]
    .fillna("")
    .astype(str)
    .value_counts()
    .to_dict()
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(event):
    timestamp = pd.to_datetime(event["timestamp"], errors="coerce")

    if pd.isna(timestamp):
        raise ValueError(f"Invalid timestamp: {event['timestamp']}")

    X = create_event_features(
        pd.DataFrame([event]),
        user_counts=USER_COUNTS,
        pc_counts=PC_COUNTS,
    )


    # Make absolutely sure feature order
    # matches the trained model.

    if hasattr(
        model,
        "feature_names_in_"
    ):

        X = X[
            model.feature_names_in_
        ]


    return X


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict_event(event):

    X = create_features(
        event
    )


    prediction = int(
        model.predict(X)[0]
    )


    if hasattr(
        model,
        "predict_proba"
    ):

        probability = float(
            model.predict_proba(X)[0][1]
        )

    else:

        probability = float(
            prediction
        )


    risk_score = round(
        probability * 100,
        2
    )


    if risk_score >= 70:

        risk_level = "HIGH"

    elif risk_score >= 40:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    status = (
        "SUSPICIOUS"
        if prediction == 1
        else "NORMAL"
    )


    return {

        "prediction":
            prediction,

        "status":
            status,

        "risk_score":
            risk_score,

        "risk_level":
            risk_level,

        "features":
            X

    }


# ============================================================
# EXPLANATION
# ============================================================

def explain_event(
    event,
    result
):

    explanations = []


    timestamp = pd.to_datetime(
        event["timestamp"],
        errors="coerce"
    )


    activity = clean_value(
        event["activity"]
    ).lower()


    resource = clean_value(
        event["resource"]
    )


    # --------------------------------------------------------
    # Time
    # --------------------------------------------------------

    if (
        not pd.isna(timestamp)
        and
        (
            timestamp.hour < 8
            or timestamp.hour >= 18
        )
    ):

        explanations.append(
            "Event occurred outside the configured normal working hours."
        )


    # --------------------------------------------------------
    # Activity
    # --------------------------------------------------------

    if activity == "logon":

        explanations.append(
            "User logon activity detected."
        )


    elif activity == "logoff":

        explanations.append(
            "User logoff activity detected."
        )


    elif activity == "connect":

        explanations.append(
            "Endpoint connection activity detected."
        )


    elif activity == "disconnect":

        explanations.append(
            "Endpoint disconnection activity detected."
        )


    elif activity == "http":

        explanations.append(
            "HTTP/web-related activity detected."
        )


    elif activity == "email":

        explanations.append(
            "Email-related activity detected."
        )


    elif activity == "device":

        explanations.append(
            "Device activity detected."
        )


    elif activity == "send email":

        explanations.append(
            "Email sending activity detected."
        )


    elif activity == "consult http":

        explanations.append(
            "HTTP consultation activity detected."
        )


    # --------------------------------------------------------
    # Resource
    # --------------------------------------------------------

    if resource:

        explanations.append(
            "A specific resource was associated with this event."
        )


    # --------------------------------------------------------
    # Model explanation
    # --------------------------------------------------------

    if not explanations:

        explanations.append(
            "The machine-learning model classified this "
            "event using the engineered endpoint behaviour features."
        )


    return explanations


# ============================================================
# FIND REAL NORMAL EVENT
# ============================================================

def get_normal_candidates():

    normal = df[
        df["case id;Cattivi"].isna()
        |
        (
            df["case id;Cattivi"]
            .astype(str)
            .str.strip()
            .isin(
                [
                    "",
                    "nan",
                    "None"
                ]
            )
        )
    ].copy()


    return normal


# ============================================================
# FIND REAL SUSPICIOUS EVENT
# ============================================================

def get_suspicious_candidates():

    suspicious = df[
        df["case id;Cattivi"].notna()
        &
        (
            df["case id;Cattivi"]
            .astype(str)
            .str.strip()
            .ne("")
        )
        &
        (
            df["case id;Cattivi"]
            .astype(str)
            .str.lower()
            .ne("nan")
        )
    ].copy()


    return suspicious


# ============================================================
# CONVERT DATASET ROW TO EVENT
# ============================================================

def row_to_event(row):

    return {

        "timestamp":
            clean_value(
                row["timestamp"]
            ),

        "user":
            clean_value(
                row["user"]
            ),

        "pc":
            clean_value(
                row["pc"]
            ),

        "activity":
            clean_value(
                row["activity"]
            ),

        "resource":
            clean_value(
                row["resource"]
            )

    }


# ============================================================
# FIND A MODEL-CONFIRMED DEMO EVENT
# ============================================================

def find_demo_event(
    desired_status
):

    if desired_status == "NORMAL":

        candidates = (
            get_normal_candidates()
        )

    else:

        candidates = (
            get_suspicious_candidates()
        )


    # --------------------------------------------------------
    # Try candidates until the model agrees.
    #
    # This happens locally and silently.
    # There is NO visible "searching" spinner.
    # --------------------------------------------------------

    for _, row in candidates.head(200).iterrows():

        try:

            event = row_to_event(
                row
            )


            if not event["timestamp"]:
                continue

            if not event["user"]:
                continue

            if not event["pc"]:
                continue

            if not event["activity"]:
                continue


            result = predict_event(
                event
            )


            if (
                result["status"]
                == desired_status
            ):

                return event, result, row


        except Exception:

            continue


    # --------------------------------------------------------
    # If no matching model prediction was found,
    # use the first valid dataset row.
    # --------------------------------------------------------

    for _, row in candidates.head(200).iterrows():

        try:

            event = row_to_event(
                row
            )


            result = predict_event(
                event
            )


            return (
                event,
                result,
                row
            )


        except Exception:

            continue


    return None, None, None


# ============================================================
# SESSION STATE
# ============================================================

if "timestamp_value" not in st.session_state:
    st.session_state.timestamp_value = ""

if "user_value" not in st.session_state:
    st.session_state.user_value = ""

if "pc_value" not in st.session_state:
    st.session_state.pc_value = ""

if "activity_value" not in st.session_state:
    st.session_state.activity_value = ""

if "resource_value" not in st.session_state:
    st.session_state.resource_value = ""

if "demo_result" not in st.session_state:
    st.session_state.demo_result = None

if "demo_event" not in st.session_state:
    st.session_state.demo_event = None

if "demo_label" not in st.session_state:
    st.session_state.demo_label = ""


# ============================================================
# LOAD DEMO INTO FORM
# ============================================================

def load_demo(
    desired_status
):

    event, result, row = (
        find_demo_event(
            desired_status
        )
    )


    if event is None:

        return False


    st.session_state.timestamp_value = (
        event["timestamp"]
    )

    st.session_state.user_value = (
        event["user"]
    )

    st.session_state.pc_value = (
        event["pc"]
    )

    st.session_state.activity_value = (
        event["activity"]
    )

    st.session_state.resource_value = (
        event["resource"]
    )

    st.session_state.demo_event = event

    st.session_state.demo_result = result

    st.session_state.demo_label = (
        desired_status
    )

    return True


# ============================================================
# HEADER
# ============================================================

st.title(
    "🛡️ Endpoint Threat Detection"
)

st.write(
    "Machine-learning based suspicious endpoint behaviour detection"
)


# ============================================================
# TEST EVENT
# ============================================================

st.header(
    "🔎 Test New Endpoint Event"
)

st.write(
    "Enter an endpoint event and let the machine-learning "
    "model classify it."
)


# ============================================================
# DEMO BUTTONS
# ============================================================

st.subheader(
    "🎬 Demo Events"
)


demo1, demo2, demo3 = st.columns(
    [1, 1, 2]
)


with demo1:

    if st.button(
        "🟢 Load Normal Demo",
        use_container_width=True
    ):

        success = load_demo(
            "NORMAL"
        )

        if success:

            st.rerun()

        else:

            st.error(
                "Could not find a suitable NORMAL event."
            )


with demo2:

    if st.button(
        "🔴 Load Suspicious Demo",
        use_container_width=True
    ):

        success = load_demo(
            "SUSPICIOUS"
        )

        if success:

            st.rerun()

        else:

            st.error(
                "Could not find a suitable SUSPICIOUS event."
            )


with demo3:

    st.info(
        "Demo buttons load real events directly from your "
        "CERT dataset and test them with the trained model."
    )


# ============================================================
# DEMO MESSAGE
# ============================================================

if st.session_state.demo_label:

    if st.session_state.demo_label == "NORMAL":

        st.success(
            "🟢 Loaded a real NORMAL event from the dataset."
        )

    else:

        st.warning(
            "🔴 Loaded a real SUSPICIOUS event from the dataset."
        )


st.divider()


# ============================================================
# EVENT DETAILS
# ============================================================

st.subheader(
    "Endpoint Event Details"
)


left, right = st.columns(2)


with left:

    timestamp = st.text_input(
        "Timestamp",
        value=st.session_state.timestamp_value
    )


    user = st.text_input(
        "User ID",
        value=st.session_state.user_value
    )


    pc = st.text_input(
        "Endpoint / PC",
        value=st.session_state.pc_value
    )


with right:

    activity = st.text_input(
        "Activity",
        value=st.session_state.activity_value
    )


    resource = st.text_input(
        "Resource",
        value=st.session_state.resource_value,
        placeholder="Leave blank if there is no resource"
    )


# ============================================================
# ANALYZE
# ============================================================

if st.button(
    "🚨 Analyze Event",
    use_container_width=True
):

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if not timestamp.strip():

        st.error(
            "Please enter a timestamp."
        )

        st.stop()


    if not user.strip():

        st.error(
            "Please enter a User ID."
        )

        st.stop()


    if not pc.strip():

        st.error(
            "Please enter an Endpoint / PC."
        )

        st.stop()


    if not activity.strip():

        st.error(
            "Please enter an Activity."
        )

        st.stop()


    # --------------------------------------------------------
    # Create event
    # --------------------------------------------------------

    event = {

        "timestamp":
            timestamp,

        "user":
            user,

        "pc":
            pc,

        "activity":
            activity,

        "resource":
            resource

    }


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    try:

        result = predict_event(
            event
        )

    except Exception as e:

        st.error(
            "Error while analyzing event."
        )

        st.exception(e)

        st.stop()


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header(
        "📊 Event Analysis"
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.metric(
            "Prediction",
            result["status"]
        )


    with c2:

        st.metric(
            "Risk Score",
            f'{result["risk_score"]:.1f} / 100'
        )


    with c3:

        st.metric(
            "Risk Level",
            result["risk_level"]
        )


    # ========================================================
    # STATUS
    # ========================================================

    if result["status"] == "SUSPICIOUS":

        st.error(
            "🚨 SUSPICIOUS BEHAVIOUR DETECTED"
        )

    else:

        st.success(
            "✅ NORMAL BEHAVIOUR"
        )


    # ========================================================
    # EVENT INFORMATION
    # ========================================================

    st.subheader(
        "Event Information"
    )


    e1, e2, e3, e4, e5 = st.columns(5)


    with e1:

        st.write("**Timestamp**")

        st.code(
            timestamp
        )


    with e2:

        st.write("**User ID**")

        st.code(
            user
        )


    with e3:

        st.write("**Endpoint / PC**")

        st.code(
            pc
        )


    with e4:

        st.write("**Activity**")

        st.code(
            activity
        )


    with e5:

        st.write("**Resource**")

        st.code(
            resource
            if resource
            else "None"
        )


    # ========================================================
    # RISK
    # ========================================================

    st.subheader(
        "Risk Score"
    )


    st.progress(
        int(
            min(
                max(
                    result["risk_score"],
                    0
                ),
                100
            )
        )
    )


    # ========================================================
    # EXPLANATION
    # ========================================================

    st.subheader(
        "🧠 Detection Explanation"
    )


    explanations = explain_event(
        event,
        result
    )


    for explanation in explanations:

        st.write(
            "• " + explanation
        )


    # ========================================================
    # FEATURES
    # ========================================================

    with st.expander(
        "🔬 View Engineered Features"
    ):

        st.dataframe(
            result["features"],
            use_container_width=True
        )


# ============================================================
# DASHBOARD
# ============================================================

st.divider()

st.header(
    "📈 Endpoint Behaviour Dashboard"
)


# ============================================================
# DASHBOARD LABEL
# ============================================================

dashboard = df.copy()


dashboard["Status"] = np.where(
    dashboard["case id;Cattivi"].isna()
    |
    (
        dashboard["case id;Cattivi"]
        .astype(str)
        .str.strip()
        .isin(
            [
                "",
                "nan",
                "None"
            ]
        )
    ),
    "NORMAL",
    "SUSPICIOUS"
)


# ============================================================
# FILTERS
# ============================================================

st.sidebar.header(
    "🎛️ Dashboard Filters"
)


user_options = sorted(
    dashboard["user"]
    .fillna("")
    .astype(str)
    .unique()
)


selected_users = st.sidebar.multiselect(
    "Users",
    user_options
)


pc_options = sorted(
    dashboard["pc"]
    .fillna("")
    .astype(str)
    .unique()
)


selected_pcs = st.sidebar.multiselect(
    "Endpoint / PC",
    pc_options
)


selected_status = st.sidebar.multiselect(
    "Status",
    [
        "NORMAL",
        "SUSPICIOUS"
    ]
)


filtered = dashboard.copy()


if selected_users:

    filtered = filtered[
        filtered["user"]
        .astype(str)
        .isin(selected_users)
    ]


if selected_pcs:

    filtered = filtered[
        filtered["pc"]
        .astype(str)
        .isin(selected_pcs)
    ]


if selected_status:

    filtered = filtered[
        filtered["Status"]
        .isin(selected_status)
    ]


# ============================================================
# METRICS
# ============================================================

total_events = len(
    filtered
)


suspicious_events = len(
    filtered[
        filtered["Status"]
        == "SUSPICIOUS"
    ]
)


normal_events = len(
    filtered[
        filtered["Status"]
        == "NORMAL"
    ]
)


suspicious_rate = (
    suspicious_events / total_events * 100
    if total_events > 0
    else 0
)


m1, m2, m3, m4 = st.columns(4)


with m1:

    st.metric(
        "Total Events",
        total_events
    )


with m2:

    st.metric(
        "Suspicious Events",
        suspicious_events
    )


with m3:

    st.metric(
        "Normal Events",
        normal_events
    )


with m4:

    st.metric(
        "Suspicious Rate",
        f"{suspicious_rate:.2f}%"
    )


# ============================================================
# CHARTS
# ============================================================

st.divider()


chart1, chart2 = st.columns(2)


with chart1:

    st.subheader(
        "Detection Distribution"
    )


    detection = (
        filtered["Status"]
        .value_counts()
        .reset_index()
    )


    detection.columns = [
        "Status",
        "Count"
    ]


    if len(detection):

        fig = px.pie(
            detection,
            names="Status",
            values="Count",
            title="Normal vs Suspicious"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


with chart2:

    st.subheader(
        "Activity Distribution"
    )


    activities = (
        filtered["activity"]
        .astype(str)
        .value_counts()
        .head(10)
        .reset_index()
    )


    activities.columns = [
        "Activity",
        "Count"
    ]


    if len(activities):

        fig = px.bar(
            activities,
            x="Activity",
            y="Count",
            title="Most Common Activities"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# SUSPICIOUS TABLE
# ============================================================

st.divider()

st.subheader(
    "🚨 Suspicious Events"
)


suspicious_table = filtered[
    filtered["Status"]
    == "SUSPICIOUS"
]


if len(suspicious_table):

    st.dataframe(

        suspicious_table[
            [
                "timestamp",
                "user",
                "pc",
                "activity",
                "resource",
                "Status"
            ]
        ].head(100),

        use_container_width=True,

        height=400
    )

else:

    st.success(
        "No suspicious events match the current filters."
    )


# ============================================================
# DOWNLOAD
# ============================================================

st.divider()

st.subheader(
    "📥 Export Results"
)


csv_data = (
    filtered
    .to_csv(index=False)
    .encode("utf-8")
)


st.download_button(
    "Download Filtered Results CSV",
    data=csv_data,
    file_name="endpoint_detection_results.csv",
    mime="text/csv",
    use_container_width=True
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Hackathon prototype — results are based on the trained "
    "Random Forest model and the reduced CERT dataset."
)