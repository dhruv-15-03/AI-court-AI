"""Health check for all agent components."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

print("Testing imports...")
from ai_court.agent.pipeline import LegalAgentPipeline
from ai_court.agent.session import SessionManager
from ai_court.llm.client import LLMClient
from ai_court.corpus.statutes import StatuteCorpus
from ai_court.rag.pipeline import rag_query
from ai_court.active_learning.loop import ActiveLearningQueue
from ai_court.model.preprocessor import TextPreprocessor
print("All imports OK\n")

# Statute corpus
sc = StatuteCorpus()
pub = [x for x in dir(sc) if not x.startswith("_")]
print(f"StatuteCorpus methods: {pub}")
if hasattr(sc, "_acts"):
    print(f"  Acts loaded: {len(sc._acts)}")
    for act_id in sc._acts:
        act = sc._acts[act_id]
        nsections = len(act.get("sections", []))
        print(f"    {act_id}: {nsections} sections")

# Session manager
sm = SessionManager()
print(f"\nSessionManager: OK")

# Active learning
alq = ActiveLearningQueue()
print(f"ActiveLearningQueue: {len(alq)} items")

# Check if model exists
model_path = "models/legal_case_classifier.pkl"
print(f"\nModel file exists: {os.path.exists(model_path)}")

# Check search index
search_path = "models/search_index.pkl"
print(f"Search index exists: {os.path.exists(search_path)}")

# Check .env for GITHUB_TOKEN
from dotenv import load_dotenv
load_dotenv()
has_token = bool(os.getenv("GITHUB_TOKEN"))
print(f"GITHUB_TOKEN configured: {has_token}")

print("\n=== HEALTH CHECK COMPLETE ===")
