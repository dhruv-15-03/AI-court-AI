import os
import sys
import json
import requests

BASE_URL = os.environ.get("SMOKE_URL") or os.environ.get("DEV_URL", "http://127.0.0.1:5002")
API = f"{BASE_URL.rstrip('/')}/api"

def get(path: str):
    r = requests.get(f"{API}{path}", timeout=10)
    r.raise_for_status()
    return r

def post(path: str, payload: dict):
    r = requests.post(f"{API}{path}", json=payload, timeout=20)
    r.raise_for_status()
    return r

def main():
    print(f"[smoke] Target: {API}")
    print("[smoke] /health:", get("/health").status_code)
    print("[smoke] /version:", get("/version").status_code)
    print("[smoke] /metrics/model:", get("/metrics/model").status_code)

    sample_text = "The appeal is allowed. The conviction is set aside and the appellant is acquitted."
    print("[smoke] /analyze:", post("/analyze", {"text": sample_text}).status_code)

    try:
        r = post("/search", {"query": "bail granted", "k": 3})
        print("[smoke] /search:", r.status_code, json.dumps(r.json(), indent=2)[:300])
    except requests.HTTPError as he:
        if he.response is not None and he.response.status_code == 503:
            print("[smoke] /search disabled or index missing (503)")
        else:
            raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[smoke] FAILED:", e)
        sys.exit(1)
