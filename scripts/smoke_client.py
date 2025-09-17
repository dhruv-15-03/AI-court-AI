import json
from urllib import request

url = "http://127.0.0.1:5002/api/analyze"

data = {
    "case_type": "Criminal",
    "parties": "State vs Accused",
    "violence_level": "Serious injury",
    "weapon": "Yes",
    "police_report": "Yes",
    "witnesses": "Yes",
    "premeditation": "Premeditated",
}

req = request.Request(url, data=json.dumps(data).encode("utf-8"), headers={"Content-Type": "application/json"})
with request.urlopen(req, timeout=10) as resp:
    print(resp.status, resp.reason)
    body = resp.read().decode("utf-8")
    print(body)
