import requests

url = "http://127.0.0.1:5000/api/predict"

data = {
    "team1": "Mumbai Indians",
    "team2": "Chennai Super Kings",
    "venue": "Wankhede Stadium",
    "season": 2026,
    "toss_winner": "Mumbai Indians",
    "toss_decision": "bat"
}

response = requests.post(
    url,
    json=data
)

print("Status:", response.status_code)
print("Response:")

print(response.json())