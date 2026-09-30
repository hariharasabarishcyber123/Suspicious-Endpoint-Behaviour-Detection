import csv

import pandas as pd


FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "is_after_hours",
    "is_logon",
    "is_logoff",
    "is_connect",
    "is_disconnect",
    "is_http",
    "is_email",
    "is_device",
    "is_send_email",
    "is_consult_http",
    "has_url",
    "resource_present",
    "activity_length",
    "activity_has_number",
    "user_event_count",
    "pc_event_count",
]


def load_event_data(file_path):
    with open(file_path, encoding="utf-8-sig", newline="") as source:
        rows = csv.reader(source)
        columns = next(rows, None)

        if columns is None:
            raise ValueError("The event dataset is empty.")

        records = []

        for line_number, row in enumerate(rows, start=2):
            if len(row) == len(columns):
                records.append(dict(zip(columns, row)))
                continue

            is_short_normal_row = (
                len(columns) == 9
                and len(row) == 7
                and columns[0] in ("", "Unnamed: 0")
                and columns[1:4] == ["file", "id", "timestamp"]
                and columns[4:8] == ["user", "pc", "activity", "resource"]
                and columns[8] == "case id;Cattivi"
                and not pd.isna(pd.to_datetime(row[2], errors="coerce"))
            )

            if not is_short_normal_row:
                raise ValueError(
                    f"Unexpected event row format on line {line_number}: "
                    f"expected {len(columns)} fields, got {len(row)}."
                )

            records.append(
                {
                    columns[0]: row[0],
                    columns[1]: "",
                    columns[2]: row[1],
                    columns[3]: row[2],
                    columns[4]: row[5],
                    columns[5]: row[3],
                    columns[6]: row[4],
                    columns[7]: "",
                    columns[8]: row[6],
                }
            )

    return pd.DataFrame.from_records(records, columns=columns).replace("", pd.NA)


def create_event_features(events, user_counts=None, pc_counts=None):
    activity = events["activity"].fillna("").astype(str)
    activity_lower = activity.str.lower()
    file_type = events.get("file", activity).fillna("").astype(str).str.lower()
    timestamp = pd.to_datetime(
        events["timestamp"],
        errors="coerce",
        format="mixed",
    )
    has_url = activity.str.contains(
        r"https?://|www\.",
        case=False,
        regex=True,
    )

    if user_counts is None:
        user_event_count = events.groupby("user")["user"].transform("count")
    else:
        user_event_count = (
            events["user"].fillna("").astype(str).map(user_counts).fillna(0)
        )

    if pc_counts is None:
        pc_event_count = events.groupby("pc")["pc"].transform("count")
    else:
        pc_event_count = (
            events["pc"].fillna("").astype(str).map(pc_counts).fillna(0)
        )

    features = pd.DataFrame(index=events.index)
    features["hour"] = timestamp.dt.hour
    features["day_of_week"] = timestamp.dt.dayofweek
    features["day_of_month"] = timestamp.dt.day
    features["month"] = timestamp.dt.month
    features["is_weekend"] = (timestamp.dt.dayofweek >= 5).astype(int)
    features["is_after_hours"] = (
        (timestamp.dt.hour < 8) | (timestamp.dt.hour >= 18)
    ).astype(int)
    features["is_logon"] = ((file_type == "logon") | (activity_lower == "logon")).astype(int)
    features["is_logoff"] = (activity_lower == "logoff").astype(int)
    features["is_connect"] = (activity_lower == "connect").astype(int)
    features["is_disconnect"] = (activity_lower == "disconnect").astype(int)
    features["is_http"] = (
        (file_type == "http") | (activity_lower == "http") | has_url
    ).astype(int)
    features["is_email"] = ((file_type == "email") | (activity_lower == "email")).astype(int)
    features["is_device"] = ((file_type == "device") | (activity_lower == "device")).astype(int)
    features["is_send_email"] = (activity_lower == "send email").astype(int)
    features["is_consult_http"] = (activity_lower == "consult http").astype(int)
    features["has_url"] = has_url.astype(int)

    resource = events.get("resource", pd.Series("", index=events.index))
    features["resource_present"] = (
        resource.fillna("").astype(str).str.strip().ne("")
    ).astype(int)
    features["activity_length"] = activity.str.len()
    features["activity_has_number"] = activity.str.contains(r"\d", regex=True).astype(int)
    features["user_event_count"] = user_event_count
    features["pc_event_count"] = pc_event_count

    return features[FEATURE_COLUMNS].fillna(0)