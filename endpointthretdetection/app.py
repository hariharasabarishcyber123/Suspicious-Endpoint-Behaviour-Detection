import streamlit as st
import pandas as pd
import plotly.express as px

from inference import model, predict_event, explain_event
from event_data import create_event_features, load_event_data


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Endpoint Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    :root {
        --canvas: #0b1110;
        --surface: #111a18;
        --surface-raised: #17221f;
        --line: #293a35;
        --text: #e7f0ec;
        --muted: #93a69e;
        --mint: #9be6c1;
        --mint-deep: #193b30;
        --danger: #ff8178;
    }

    .stApp {
        color: var(--text);
        background-color: var(--canvas);
        background-image:
            linear-gradient(rgba(155, 230, 193, 0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(155, 230, 193, 0.025) 1px, transparent 1px);
        background-size: 36px 36px;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 16, 0.88);
    }

    [data-testid="stSidebar"] {
        background: #101816;
        border-right: 1px solid var(--line);
    }

    .block-container {
        max-width: 1560px;
        padding-top: 2.25rem;
        padding-bottom: 3rem;
    }

    .app-heading {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.25rem 0 1.4rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid var(--line);
    }

    .brand-mark {
        display: grid;
        place-items: center;
        flex: 0 0 3.25rem;
        height: 3.25rem;
        border: 1px solid #416c5a;
        border-radius: 8px;
        background: var(--mint-deep);
        font-size: 1.55rem;
    }

    .heading-copy {
        min-width: 0;
    }

    .title {
        margin: 0;
        color: var(--text);
        font-family: "Bahnschrift", "Trebuchet MS", sans-serif;
        font-size: 2.15rem;
        font-weight: 700;
        line-height: 1.12;
    }

    .eyebrow {
        margin-bottom: 0.35rem;
        color: var(--mint);
        font-family: "Bahnschrift", "Trebuchet MS", sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
    }

    .model-indicator {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-left: auto;
        padding: 0.5rem 0.7rem;
        border: 1px solid var(--line);
        border-radius: 6px;
        color: var(--muted);
        font-size: 0.72rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .live-dot {
        width: 0.5rem;
        height: 0.5rem;
        border-radius: 50%;
        background: var(--mint);
        box-shadow: 0 0 10px rgba(155, 230, 193, 0.55);
    }

    .subtitle {
        margin: 0.45rem 0 0;
        color: var(--muted);
        font-size: 0.95rem;
    }

    h1, h2, h3 {
        color: var(--text);
        font-family: "Bahnschrift", "Trebuchet MS", sans-serif;
    }

    div[data-testid="stMetric"] {
        min-height: 6.2rem;
        padding: 0.8rem 1rem;
        border: 1px solid var(--line);
        border-left: 3px solid var(--mint);
        border-radius: 6px;
        background: var(--surface);
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    div[data-testid="stMetricValue"] {
        color: var(--text);
        font-family: "Bahnschrift", "Trebuchet MS", sans-serif;
    }

    [data-testid="stForm"] {
        padding: 1.15rem 1.25rem 1.25rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: rgba(17, 26, 24, 0.92);
    }

    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div,
    [data-baseweb="select"] > div {
        border-color: var(--line);
        border-radius: 6px;
        background: var(--surface-raised);
    }

    [data-baseweb="input"] > div:focus-within,
    [data-baseweb="textarea"] > div:focus-within,
    [data-baseweb="select"] > div:focus-within {
        border-color: var(--mint);
        box-shadow: 0 0 0 1px var(--mint);
    }

    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        min-height: 2.75rem;
        border: 1px solid #426555;
        border-radius: 6px;
        background: #15251f;
        color: var(--text);
        font-weight: 650;
        transition: border-color 140ms ease, background 140ms ease, transform 140ms ease;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        border-color: var(--mint);
        background: #1b342a;
        color: #ffffff;
        transform: translateY(-1px);
    }

    [data-testid="stAlert"] {
        border-radius: 6px;
    }

    [data-testid="stDataFrame"],
    [data-testid="stPlotlyChart"] {
        border: 1px solid var(--line);
        border-radius: 6px;
        overflow: hidden;
    }

    hr {
        border-color: var(--line);
    }

    @media (max-width: 700px) {
        .block-container {
            padding: 1.25rem 1rem 2rem;
        }

        .app-heading {
            align-items: flex-start;
            flex-wrap: wrap;
        }

        .title {
            font-size: 1.65rem;
        }

        .model-indicator {
            margin-left: 4.25rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD ORIGINAL EVENT DATA
# ============================================================

@st.cache_data
def load_original_data():

    return load_event_data(
        "data/cert_preprocessed_ml_comparison.csv"
    )


# ============================================================
# LOAD MODEL PREDICTIONS
# ============================================================

@st.cache_data
def load_predictions(events):

    features = create_event_features(events)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    results = pd.DataFrame(
        {
            "prediction": [
                "Suspicious" if prediction == 1 else "Normal"
                for prediction in predictions
            ],
            "risk_score": (probabilities * 100).round(2),
        }
    )

    if "id" in events.columns:
        results["id"] = events["id"].values

    return results


try:

    original_df = load_original_data()

    predictions_df = load_predictions(original_df)

except Exception as e:

    st.error(
        "Could not load the required dataset files."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SHOW DATA INFORMATION
# ============================================================

# The original dataset contains the real event information.
# Predictions are calculated from the current model at load time.


# ============================================================
# FIND COLUMNS
# ============================================================

def find_column(dataframe, possible_names):

    for name in possible_names:

        if name in dataframe.columns:

            return name

    return None


# Original event columns

ORIGINAL_TIMESTAMP = find_column(
    original_df,
    [
        "timestamp",
        "Timestamp",
        "datetime",
        "date"
    ]
)

ORIGINAL_USER = find_column(
    original_df,
    [
        "user",
        "User",
        "user_id"
    ]
)

ORIGINAL_PC = find_column(
    original_df,
    [
        "pc",
        "PC",
        "endpoint"
    ]
)

ORIGINAL_ACTIVITY = find_column(
    original_df,
    [
        "activity",
        "Activity"
    ]
)

ORIGINAL_RESOURCE = find_column(
    original_df,
    [
        "resource",
        "Resource"
    ]
)

ORIGINAL_ID = find_column(
    original_df,
    [
        "id",
        "ID",
        "event_id"
    ]
)


# Prediction columns

PREDICTION_COLUMN = find_column(
    predictions_df,
    [
        "prediction",
        "Prediction",
        "status",
        "Status"
    ]
)

RISK_SCORE_COLUMN = find_column(
    predictions_df,
    [
        "risk_score",
        "Risk Score"
    ]
)

PREDICTION_ID = find_column(
    predictions_df,
    [
        "id",
        "ID",
        "event_id"
    ]
)


# ============================================================
# VALIDATION
# ============================================================

required_original_columns = [

    ORIGINAL_TIMESTAMP,
    ORIGINAL_USER,
    ORIGINAL_PC,
    ORIGINAL_ACTIVITY

]

if any(
    column is None
    for column in required_original_columns
):

    st.error(
        "The original dataset does not contain the expected "
        "timestamp/user/pc/activity columns."
    )

    st.write(
        "Original dataset columns:"
    )

    st.write(
        list(original_df.columns)
    )

    st.stop()


# ============================================================
# BUILD COMBINED DATASET
# ============================================================

def build_combined_dataset():

    """
    Combine the REAL event information from the original
    CERT dataset with the model results.

    Priority:
    1. Match using event ID if available.
    2. Otherwise match rows by position.
    """

    events = original_df.copy()

    predictions = predictions_df.copy()

    # --------------------------------------------------------
    # Match by ID if both files have an ID column
    # --------------------------------------------------------

    if (
        ORIGINAL_ID is not None
        and PREDICTION_ID is not None
    ):

        events["_join_id"] = (
            events[ORIGINAL_ID]
            .astype(str)
        )

        predictions["_join_id"] = (
            predictions[PREDICTION_ID]
            .astype(str)
        )

        # Only bring prediction columns
        prediction_columns = []

        if PREDICTION_COLUMN:

            prediction_columns.append(
                PREDICTION_COLUMN
            )

        if RISK_SCORE_COLUMN:

            prediction_columns.append(
                RISK_SCORE_COLUMN
            )

        if prediction_columns:

            pred_part = predictions[
                ["_join_id"] + prediction_columns
            ].copy()

            pred_part = pred_part.drop_duplicates(
                subset="_join_id"
            )

            combined = events.merge(
                pred_part,
                on="_join_id",
                how="left"
            )

        else:

            combined = events.copy()

    else:

        # ----------------------------------------------------
        # Fallback: same row order
        # ----------------------------------------------------

        combined = events.copy()

        if len(predictions) == len(combined):

            if PREDICTION_COLUMN:

                combined[
                    "_prediction"
                ] = predictions[
                    PREDICTION_COLUMN
                ].values

            if RISK_SCORE_COLUMN:

                combined[
                    "_risk_score"
                ] = predictions[
                    RISK_SCORE_COLUMN
                ].values

        else:

            st.warning(
                "The event dataset and calculated predictions "
                "have different row counts. Showing the available "
                "event data and model results."
            )

    return combined


combined_df = build_combined_dataset()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_string(value):

    if value is None:

        return ""

    try:

        if pd.isna(value):

            return ""

    except Exception:

        pass

    return str(value).strip()


def valid_timestamp(value):

    try:

        result = pd.to_datetime(
            value,
            errors="coerce"
        )

        return not pd.isna(result)

    except Exception:

        return False


def normalize_status(value):

    value = safe_string(value).lower()

    if value in [
        "normal",
        "0",
        "false"
    ]:

        return "NORMAL"

    if value in [
        "suspicious",
        "1",
        "true",
        "anomaly"
    ]:

        return "SUSPICIOUS"

    return value.upper()


def get_prediction_value(row):

    # Normal prediction column

    if PREDICTION_COLUMN:

        return normalize_status(
            row[PREDICTION_COLUMN]
        )

    # Fallback prediction column

    if "_prediction" in row.index:

        return normalize_status(
            row["_prediction"]
        )

    return ""


def get_risk_score(row):

    value = None

    if RISK_SCORE_COLUMN:

        value = row[RISK_SCORE_COLUMN]

    elif "_risk_score" in row.index:

        value = row["_risk_score"]

    try:

        if pd.isna(value):

            raise ValueError

        return float(value)

    except Exception:

        status = get_prediction_value(row)

        if status == "SUSPICIOUS":

            return 100.0

        return 0.0


def get_risk_level(score):

    if score >= 70:

        return "HIGH"

    elif score >= 40:

        return "MEDIUM"

    return "LOW"


# ============================================================
# CREATE CLEAN EVENT VIEW
# ============================================================

dashboard_df = combined_df.copy()


dashboard_df["display_timestamp"] = (
    dashboard_df[
        ORIGINAL_TIMESTAMP
    ]
    .astype(str)
)


dashboard_df["display_user"] = (
    dashboard_df[
        ORIGINAL_USER
    ]
    .astype(str)
)


dashboard_df["display_pc"] = (
    dashboard_df[
        ORIGINAL_PC
    ]
    .astype(str)
)


dashboard_df["display_activity"] = (
    dashboard_df[
        ORIGINAL_ACTIVITY
    ]
    .astype(str)
)


if ORIGINAL_RESOURCE:

    dashboard_df["display_resource"] = (
        dashboard_df[
            ORIGINAL_RESOURCE
        ]
        .fillna("")
        .astype(str)
    )

else:

    dashboard_df["display_resource"] = ""


dashboard_df["display_status"] = (
    dashboard_df
    .apply(
        get_prediction_value,
        axis=1
    )
)


dashboard_df["display_risk"] = (
    dashboard_df
    .apply(
        get_risk_score,
        axis=1
    )
)


dashboard_df["risk_level"] = (
    dashboard_df[
        "display_risk"
    ]
    .apply(
        get_risk_level
    )
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {

    "timestamp_value": "",

    "user_value": "",

    "pc_value": "",

    "activity_type": "Other",

    "activity_value": "",

    "resource_value": "",

    "demo_loaded": False,

    "demo_event": None,

    "demo_result": None,

    "demo_message": ""

}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ============================================================
# ACTIVITY OPTIONS
# ============================================================

STANDARD_ACTIVITIES = [

    "Logon",
    "Logoff",
    "Connect",
    "Disconnect",
    "http",
    "email",
    "device",
    "send email",
    "consult http"

]

ACTIVITY_OPTIONS = (
    STANDARD_ACTIVITIES
    + ["Other"]
)


# ============================================================
# GET REAL EVENT FROM DATASET
# ============================================================

def row_to_event(row):

    return {

        "timestamp":
            safe_string(
                row[ORIGINAL_TIMESTAMP]
            ),

        "user":
            safe_string(
                row[ORIGINAL_USER]
            ),

        "pc":
            safe_string(
                row[ORIGINAL_PC]
            ),

        "activity":
            safe_string(
                row[ORIGINAL_ACTIVITY]
            ),

        "resource":
            safe_string(
                row[ORIGINAL_RESOURCE]
            )
            if ORIGINAL_RESOURCE
            else ""

    }


# ============================================================
# FIND DEMO EVENT
# ============================================================

def find_demo_row(desired_status):

    matches = dashboard_df[
        dashboard_df[
            "display_status"
        ]
        == desired_status
    ]

    if len(matches) == 0:

        return None

    # Prefer a complete valid event

    for index in matches.index:

        row = dashboard_df.loc[
            index
        ]

        event = row_to_event(
            row
        )

        if (
            valid_timestamp(
                event["timestamp"]
            )
            and event["user"]
            and event["pc"]
            and event["activity"]
        ):

            return row

    return matches.iloc[0]


# ============================================================
# LOAD DEMO
# ============================================================

def load_demo(row):

    event = row_to_event(
        row
    )

    status = get_prediction_value(
        row
    )

    risk_score = get_risk_score(
        row
    )

    result = {

        "prediction":
            1 if status == "SUSPICIOUS"
            else 0,

        "status":
            status,

        "risk_score":
            risk_score,

        "risk_level":
            get_risk_level(
                risk_score
            ),

        "features":
            pd.DataFrame()

    }

    # --------------------------------------------------------
    # Put REAL values into form
    # --------------------------------------------------------

    st.session_state.timestamp_value = (
        event["timestamp"]
    )

    st.session_state.user_value = (
        event["user"]
    )

    st.session_state.pc_value = (
        event["pc"]
    )

    st.session_state.resource_value = (
        event["resource"]
    )

    activity = event["activity"]

    matched = None

    for standard in STANDARD_ACTIVITIES:

        if (
            activity.lower()
            == standard.lower()
        ):

            matched = standard

            break

    if matched:

        st.session_state.activity_type = (
            matched
        )

        st.session_state.activity_value = (
            activity
        )

    else:

        st.session_state.activity_type = (
            "Other"
        )

        st.session_state.activity_value = (
            activity
        )

    st.session_state.demo_loaded = True

    st.session_state.demo_event = event

    st.session_state.demo_result = result

    st.session_state.demo_message = (
        f"Loaded a real {status} event "
        f"from the dataset."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-heading">'
    '<div class="brand-mark">🛡️</div>'
    '<div class="heading-copy">'
    '<div class="eyebrow">ENDPOINT SECURITY / THREAT MONITOR</div>'
    '<h1 class="title">Endpoint Threat Detection</h1>'
    '<div class="subtitle">Machine-learning based suspicious endpoint behaviour detection</div>'
    '</div>'
    '<div class="model-indicator"><span class="live-dot"></span>MODEL READY</div>'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# TEST EVENT
# ============================================================

st.header(
    "🔎 Test New Endpoint Event"
)

st.write(
    "Enter an endpoint event and let the "
    "machine-learning model classify it."
)


# ============================================================
# DEMO BUTTONS
# ============================================================

st.subheader(
    "🎬 Demo Events"
)


col1, col2, col3 = st.columns(
    [1, 1, 2]
)


# ============================================================
# NORMAL DEMO
# ============================================================

with col1:

    if st.button(
        "🟢 Load Normal Demo",
        use_container_width=True
    ):

        row = find_demo_row(
            "NORMAL"
        )

        if row is None:

            st.error(
                "No NORMAL event was found."
            )

        else:

            load_demo(
                row
            )

            st.rerun()


# ============================================================
# SUSPICIOUS DEMO
# ============================================================

with col2:

    if st.button(
        "🔴 Load Suspicious Demo",
        use_container_width=True
    ):

        row = find_demo_row(
            "SUSPICIOUS"
        )

        if row is None:

            st.error(
                "No SUSPICIOUS event was found."
            )

        else:

            load_demo(
                row
            )

            st.rerun()


# ============================================================
# INFO
# ============================================================

with col3:

    st.info(
        "Demo events use the original CERT event fields "
        "and the corresponding model prediction."
    )


if st.session_state.demo_message:

    st.success(
        st.session_state.demo_message
    )


st.divider()


# ============================================================
# EVENT FORM
# ============================================================

with st.form(
    "event_analysis_form"
):

    st.subheader(
        "Endpoint Event Details"
    )

    left, right = st.columns(2)


    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    with left:

        timestamp = st.text_input(
            "Timestamp",
            value=st.session_state.timestamp_value,
            placeholder="2010-07-28 14:00:00"
        )

        user = st.text_input(
            "User ID",
            value=st.session_state.user_value,
            placeholder="EHBO824"
        )

        pc = st.text_input(
            "Endpoint / PC",
            value=st.session_state.pc_value,
            placeholder="PC-0141"
        )


    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    with right:

        current_activity = (
            st.session_state.activity_type
        )

        if current_activity not in ACTIVITY_OPTIONS:

            current_activity = "Other"


        activity_type = st.selectbox(
            "Activity",
            ACTIVITY_OPTIONS,
            index=ACTIVITY_OPTIONS.index(
                current_activity
            )
        )


        if activity_type == "Other":

            activity = st.text_input(
                "Enter Activity",
                value=st.session_state.activity_value,
                placeholder="Example: 179;0"
            )

            st.caption(
                "Custom activities can be entered here, "
                "for example: 179;0"
            )

        else:

            activity = activity_type


        resource = st.text_input(
            "Resource",
            value=st.session_state.resource_value,
            placeholder="Leave blank if no resource"
        )


    submitted = st.form_submit_button(
        "🚨 Analyze Event",
        use_container_width=True
    )


# ============================================================
# UPDATE STATE
# ============================================================

st.session_state.timestamp_value = timestamp

st.session_state.user_value = user

st.session_state.pc_value = pc

st.session_state.activity_type = activity_type

st.session_state.activity_value = activity

st.session_state.resource_value = resource


# ============================================================
# ANALYZE EVENT
# ============================================================

if submitted:

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not timestamp.strip():

        st.error(
            "Please enter a timestamp."
        )

        st.stop()


    if not valid_timestamp(
        timestamp
    ):

        st.error(
            f"Invalid timestamp: {timestamp}"
        )

        st.info(
            "Use a format such as "
            "2010-07-28 14:00:00"
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
            "Please enter an activity."
        )

        st.stop()


    # --------------------------------------------------------
    # EVENT
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
    # CHECK WHETHER THIS IS THE LOADED DEMO
    # --------------------------------------------------------

    same_demo = False


    if (
        st.session_state.demo_loaded
        and
        st.session_state.demo_event
    ):

        demo = (
            st.session_state.demo_event
        )

        same_demo = (

            event["timestamp"]
            == demo["timestamp"]

            and

            event["user"]
            == demo["user"]

            and

            event["pc"]
            == demo["pc"]

            and

            event["activity"]
            == demo["activity"]

            and

            event["resource"]
            == demo["resource"]

        )


    # --------------------------------------------------------
    # DEMO EVENT
    # --------------------------------------------------------

    if same_demo:

        result = (
            st.session_state.demo_result
        )

        st.info(
            "Using the stored model result for "
            "this dataset event."
        )


    # --------------------------------------------------------
    # MANUAL EVENT
    # --------------------------------------------------------

    else:

        st.session_state.demo_loaded = False

        st.session_state.demo_event = None

        try:

            result = predict_event(
                event
            )

        except Exception as e:

            st.error(
                "An error occurred while analyzing the event."
            )

            st.exception(e)

            st.stop()


    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Event Analysis"
    )


    r1, r2, r3 = st.columns(3)


    with r1:

        st.metric(
            "Prediction",
            result["status"]
        )


    with r2:

        st.metric(
            "Risk Score",
            f'{result["risk_score"]:.1f} / 100'
        )


    with r3:

        st.metric(
            "Risk Level",
            result["risk_level"]
        )


    # ========================================================
    # ALERT
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
    # EVENT DETAILS
    # ========================================================

    st.subheader(
        "Event Details"
    )


    e1, e2, e3, e4 = st.columns(4)


    with e1:

        st.write("**User**")

        st.code(
            user
        )


    with e2:

        st.write("**Endpoint / PC**")

        st.code(
            pc
        )


    with e3:

        st.write("**Activity**")

        st.code(
            activity
        )


    with e4:

        st.write("**Resource**")

        if resource.strip():

            st.code(
                resource
            )

        else:

            st.code(
                "None"
            )


    # ========================================================
    # RISK
    # ========================================================

    st.subheader(
        "Risk Score"
    )


    risk_score = float(
        result["risk_score"]
    )


    st.progress(
        min(
            max(
                risk_score / 100,
                0.0
            ),
            1.0
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


    if explanations:

        for explanation in explanations:

            st.write(
                "• " + explanation
            )

    else:

        st.write(
            "• The machine-learning model classified "
            "the event using engineered endpoint "
            "behaviour features."
        )


    # ========================================================
    # FEATURES
    # ========================================================

    if (
        "features" in result
        and
        result["features"] is not None
        and
        len(result["features"]) > 0
    ):

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
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header(
    "🎛️ Dashboard Filters"
)


users = sorted(
    dashboard_df[
        "display_user"
    ]
    .dropna()
    .unique()
    .tolist()
)


selected_users = st.sidebar.multiselect(
    "Users",
    users
)


pcs = sorted(
    dashboard_df[
        "display_pc"
    ]
    .dropna()
    .unique()
    .tolist()
)


selected_pcs = st.sidebar.multiselect(
    "Endpoints / PCs",
    pcs
)


selected_status = st.sidebar.multiselect(
    "Detection Status",
    [
        "NORMAL",
        "SUSPICIOUS"
    ]
)


selected_risk = st.sidebar.multiselect(
    "Risk Level",
    [
        "LOW",
        "MEDIUM",
        "HIGH"
    ]
)


# ============================================================
# FILTER
# ============================================================

filtered_df = dashboard_df.copy()


if selected_users:

    filtered_df = filtered_df[
        filtered_df[
            "display_user"
        ]
        .isin(
            selected_users
        )
    ]


if selected_pcs:

    filtered_df = filtered_df[
        filtered_df[
            "display_pc"
        ]
        .isin(
            selected_pcs
        )
    ]


if selected_status:

    filtered_df = filtered_df[
        filtered_df[
            "display_status"
        ]
        .isin(
            selected_status
        )
    ]


if selected_risk:

    filtered_df = filtered_df[
        filtered_df[
            "risk_level"
        ]
        .isin(
            selected_risk
        )
    ]


# ============================================================
# METRICS
# ============================================================

total_events = len(
    filtered_df
)


suspicious_events = len(
    filtered_df[
        filtered_df[
            "display_status"
        ]
        == "SUSPICIOUS"
    ]
)


normal_events = len(
    filtered_df[
        filtered_df[
            "display_status"
        ]
        == "NORMAL"
    ]
)


if total_events:

    suspicious_rate = (
        suspicious_events
        / total_events
        * 100
    )

else:

    suspicious_rate = 0


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


c1, c2 = st.columns(2)


with c1:

    st.subheader(
        "Detection Distribution"
    )

    counts = (
        filtered_df[
            "display_status"
        ]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "Status",
        "Count"
    ]

    if len(counts):

        fig = px.pie(
            counts,
            names="Status",
            values="Count",
            title="Normal vs Suspicious"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


with c2:

    st.subheader(
        "Risk Level Distribution"
    )

    risk_counts = (
        filtered_df[
            "risk_level"
        ]
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = [
        "Risk Level",
        "Count"
    ]

    if len(risk_counts):

        fig = px.bar(
            risk_counts,
            x="Risk Level",
            y="Count",
            title="Risk Levels"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# ACTIVITY ANALYSIS
# ============================================================

st.divider()

st.subheader(
    "📊 Activity Analysis"
)


activity_counts = (
    filtered_df[
        "display_activity"
    ]
    .value_counts()
    .head(15)
    .reset_index()
)


activity_counts.columns = [
    "Activity",
    "Count"
]


if len(activity_counts):

    fig = px.bar(
        activity_counts,
        x="Activity",
        y="Count",
        title="Most Common Endpoint Activities"
    )

    fig.update_layout(
        xaxis_tickangle=-45
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# SUSPICIOUS EVENTS
# ============================================================

st.divider()

st.subheader(
    "🚨 Suspicious Events"
)


suspicious_table = filtered_df[
    filtered_df[
        "display_status"
    ]
    == "SUSPICIOUS"
].copy()


show_columns = [

    "display_timestamp",

    "display_user",

    "display_pc",

    "display_activity",

    "display_resource",

    "display_status",

    "display_risk",

    "risk_level"

]


if len(suspicious_table):

    st.dataframe(
        suspicious_table[
            show_columns
        ].head(100),
        use_container_width=True,
        height=400
    )

else:

    st.success(
        "No suspicious events match the current filters."
    )


# ============================================================
# EVENT INVESTIGATION
# ============================================================

st.divider()

st.subheader(
    "🔍 Event Investigation"
)


if len(filtered_df):

    selected_index = st.selectbox(
        "Select an event",
        filtered_df.index.tolist(),
        format_func=lambda x:
        f"Event {x}"
    )


    selected_event = filtered_df.loc[
        selected_index
    ]


    investigation = pd.DataFrame(
        {
            "Field": [

                "Timestamp",
                "User",
                "Endpoint / PC",
                "Activity",
                "Resource",
                "Prediction",
                "Risk Score",
                "Risk Level"

            ],

            "Value": [

                selected_event[
                    "display_timestamp"
                ],

                selected_event[
                    "display_user"
                ],

                selected_event[
                    "display_pc"
                ],

                selected_event[
                    "display_activity"
                ],

                selected_event[
                    "display_resource"
                ],

                selected_event[
                    "display_status"
                ],

                selected_event[
                    "display_risk"
                ],

                selected_event[
                    "risk_level"
                ]

            ]
        }
    )


    st.dataframe(
        investigation,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DOWNLOAD
# ============================================================

st.divider()

st.subheader(
    "📥 Export Results"
)


csv_data = (
    filtered_df
    .to_csv(index=False)
    .encode("utf-8")
)


st.download_button(
    label="Download Filtered Results CSV",
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
    "⚠️ Hackathon prototype: risk scores and classifications "
    "are based on the trained model and the reduced CERT dataset. "
    "They should not be interpreted as production-grade "
    "security decisions."
)