from inference import predict_event, explain_event


normal_event = {
    "timestamp": "2010-02-23 15:43:45",
    "user": "SBN0971",
    "pc": "PC-0151",
    "activity": "consult http",
    "resource": "",
}

normal_result = predict_event(normal_event)
assert normal_result["status"] == "NORMAL"
assert int(normal_result["features"].iloc[0]["hour"]) == 15

print("Normal demo prediction:", normal_result["status"])


event = {
    "timestamp": "2010-07-28 23:30:00",
    "user": "EHBO824",
    "pc": "PC-0141",
    "activity": "Logon",
    "resource": "EHBO824"
}


result = predict_event(event)

print("\n==============================")
print("NEW EVENT ANALYSIS")
print("==============================")

print("Prediction :", result["status"])
print("Risk Score :", result["risk_score"])
print("Risk Level :", result["risk_level"])

print("\nExplanation:")

for explanation in explain_event(event, result):
    print("-", explanation)