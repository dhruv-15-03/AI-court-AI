"""Quick verification — no LLM calls, just structural checks."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
os.environ["FLASK_ENV"] = "testing"

P = 0; F = 0
def ok(n, c, d=""):
    global P, F
    if c: P += 1; print(f"  [PASS] {n}")
    else: F += 1; print(f"  [FAIL] {n} — {d}")

print("=" * 60)
print("  PRODUCT VERIFICATION (quick)")
print("=" * 60)

# Imports
print("\n[1] IMPORTS")
try:
    from ai_court.agent.pipeline import LegalAgentPipeline; ok("Agent Pipeline", True)
except Exception as e: ok("Agent Pipeline", False, str(e))
try:
    from ai_court.agent.session import SessionManager; ok("Session Manager", True)
except Exception as e: ok("Session Manager", False, str(e))
try:
    from ai_court.llm.client import LLMClient; ok("LLM Client", True)
except Exception as e: ok("LLM Client", False, str(e))
try:
    from ai_court.documents.processor import DocumentProcessor, format_documents_for_llm; ok("Doc Processor", True)
except Exception as e: ok("Doc Processor", False, str(e))
try:
    from ai_court.documents.court_docs import CourtDocumentGenerator, DOCUMENT_TYPES
    ok(f"Court Docs ({len(DOCUMENT_TYPES)} types)", len(DOCUMENT_TYPES) >= 10)
except Exception as e: ok("Court Docs", False, str(e))
try:
    from ai_court.active_learning.loop import ActiveLearningQueue; ok("Active Learning", True)
except Exception as e: ok("Active Learning", False, str(e))

# Files
print("\n[2] MODEL + DATA FILES")
ok("RF model", os.path.exists("models/legal_case_classifier.pkl"))
ok("Search index", os.path.exists("models/search_index.pkl"))
ok("BERT model", os.path.isdir("models/production/bert_classifier"))
import glob
ok(f"Raw CSVs ({len(glob.glob('data/raw/*.csv'))})", len(glob.glob("data/raw/*.csv")) > 40)
ok(f"Statute files ({len(glob.glob('data/statutes/*.json'))})", len(glob.glob("data/statutes/*.json")) >= 5)
ok("Master CSV", os.path.exists("data/processed/all_cases_master.csv"))
if os.path.exists("models/metrics.json"):
    m = json.load(open("models/metrics.json"))
    rf = m.get("models", {}).get("rf", {})
    ok(f"RF accuracy: {rf.get('accuracy', 0):.1%}", rf.get("accuracy", 0) > 0.8)
if os.path.exists("models/production/bert_classifier/metrics.json"):
    bm = json.load(open("models/production/bert_classifier/metrics.json"))
    ok(f"BERT accuracy: {bm.get('accuracy', 0):.1%}", bm.get("accuracy", 0) > 0.75)
    ok(f"BERT classes: {bm.get('num_classes', 0)}", bm.get("num_classes", 0) == 11)

# Server
print("\n[3] SERVER")
from ai_court.api import server, state
app = server.app
ok("Flask app", app is not None)
ok("Classifier", state.classifier is not None)
ok("Search index", state.search_index is not None)
ok("LLM client", state.llm_client is not None)
ok("Statute corpus", state.statute_corpus is not None and state.statute_corpus.loaded)
ok("Agent pipeline", state.agent_pipeline is not None)
ok("Session manager", state.session_manager is not None)

# Endpoints (no LLM calls)
print("\n[4] ENDPOINTS (fast, no LLM)")
c = app.test_client()
r = c.get("/api/health"); ok(f"/api/health → {r.status_code}", r.status_code == 200)
r = c.post("/api/analyze", json={"case_data": "Murder IPC 302", "case_type": "Criminal"})
ok(f"/api/analyze → {r.status_code}", r.status_code == 200)
r = c.post("/api/search", json={"query": "murder bail", "k": 3}); ok(f"/api/search → {r.status_code}", r.status_code == 200)
r = c.get("/api/agent/health"); ok(f"/api/agent/health → {r.status_code}", r.status_code == 200)
r = c.get("/api/agent/document-types"); ok(f"/api/agent/document-types → {r.status_code}", r.status_code == 200)
r = c.post("/api/agent/analyze", json={"query": "murder charge", "tier": "free"})
ok(f"/api/agent/analyze (free) → {r.status_code}", r.status_code == 200)

# Document processing
print("\n[5] DOCUMENT PROCESSING")
proc = DocumentProcessor()
doc = proc.process_file(file_bytes=b"FIR No 123/2024. Section 302 IPC. Date 15-03-2024. Accused vs State.", filename="fir.txt")
ok("Text parsing", len(doc.text) > 20)
ok(f"Detected type: {doc.doc_type_guess}", doc.doc_type_guess == "fir")
ok(f"Sections: {doc.sections_mentioned}", len(doc.sections_mentioned) > 0)
ok("LLM context format", len(format_documents_for_llm([doc])) > 50)

# LLM config
print("\n[6] LLM CONFIG")
from dotenv import load_dotenv; load_dotenv()
ok("GITHUB_TOKEN set", bool(os.getenv("GITHUB_TOKEN")))

# GPU
print("\n[7] GPU")
try:
    import torch
    ok(f"CUDA: {torch.cuda.is_available()}", torch.cuda.is_available())
    if torch.cuda.is_available():
        ok(f"GPU: {torch.cuda.get_device_name(0)}", True)
except: ok("CUDA", False, "torch import failed")

# Frontend
print("\n[8] FRONTEND FILES")
fe = os.path.join("..", "AI-Courtroom", "frontend", "src")
ok("AILawyer.jsx", os.path.exists(os.path.join(fe, "pages", "AILawyer.jsx")))
ok("agentService in api.js", "agentService" in open(os.path.join(fe, "services", "api.js")).read())
ok("/ai-lawyer route", "ai-lawyer" in open(os.path.join(fe, "App.js")).read())
ok("Sidebar AI Lawyer link", "AI Lawyer" in open(os.path.join(fe, "components", "Sidebar.jsx")).read())

# Java backend
print("\n[9] JAVA BACKEND FILES")
jb = os.path.join("..", "AI-Courtroom", "backend", "demo", "src", "main", "java", "com", "example", "demo")
ok("AIAgentController.java", os.path.exists(os.path.join(jb, "Controller", "AIAgentController.java")))
aj = open(os.path.join(jb, "Controller", "AIAgentController.java")).read()
ok("  /analyze endpoint", '"/analyze"' in aj)
ok("  /chat endpoint", '"/chat"' in aj)
ok("  /upload-documents", "upload-documents" in aj)
ok("  /generate-document", "generate-document" in aj)
ok("  Subscription auth", "subscriptionService" in aj)
ok("SubscriptionPlan AI Lawyer", "AI Lawyer" in open(os.path.join(jb, "Classes", "SubscriptionPlan.java")).read())

# Summary
print("\n" + "=" * 60)
print(f"  RESULTS: {P} PASSED | {F} FAILED")
print("=" * 60)
if F == 0:
    print("\n  ALL CHECKS PASSED")
else:
    print(f"\n  {F} issue(s) found")
