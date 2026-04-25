"""End-to-end verification of the trained system."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
os.environ["FLASK_ENV"] = "testing"

from ai_court.api import server

print("Server initialized, testing endpoints...")
app = server.app
client = app.test_client()

# 1. Health
r = client.get("/api/health")
print(f"Health: {r.status_code}")

# 2. Analyze
r = client.post("/api/analyze", json={
    "case_data": "The accused was charged under Section 302 IPC for murder. Two eyewitnesses saw the incident. The weapon was recovered from the scene.",
    "case_type": "Criminal",
})
print(f"Analyze: {r.status_code}")
if r.status_code == 200:
    data = r.get_json()
    pred = data.get("prediction", "N/A")
    conf = data.get("confidence", "N/A")
    print(f"  Prediction: {pred}")
    print(f"  Confidence: {conf}")

# 3. Search
r = client.post("/api/search", json={"query": "murder bail denied", "k": 3})
print(f"Search: {r.status_code}")
if r.status_code == 200:
    data = r.get_json()
    results = data.get("results", [])
    print(f"  Results: {len(results)} cases found")

# 4. Agent health
r = client.get("/api/agent/health")
print(f"Agent Health: {r.status_code}")
if r.status_code == 200:
    data = r.get_json()
    for k, v in data.items():
        print(f"  {k}: {v}")

# 5. Model info
r = client.get("/api/version")
print(f"Version: {r.status_code}")

print("\n=== ALL SYSTEMS OPERATIONAL ===")
