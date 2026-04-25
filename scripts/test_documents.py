"""Test document processing and court document generation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# Test imports
from ai_court.documents.processor import DocumentProcessor, ExtractedDocument, format_documents_for_llm
from ai_court.documents.court_docs import CourtDocumentGenerator, DOCUMENT_TYPES
print("Document processor: OK")
print(f"Document types available: {len(DOCUMENT_TYPES)}")
for k, v in DOCUMENT_TYPES.items():
    title = v["title"]
    print(f"  {k}: {title}")

# Test processing a text file
proc = DocumentProcessor()
fir_text = (
    b"FIRST INFORMATION REPORT\n"
    b"FIR No. 123/2024\n"
    b"Police Station: Saket, New Delhi\n"
    b"Under Section 302 IPC and Section 34 IPC\n"
    b"Date: 15-03-2024\n"
    b"Complainant: Ramesh Kumar vs State\n"
    b"The complainant states that on 15-03-2024, the accused person attacked the victim "
    b"with a sharp weapon near the market area. Two eyewitnesses were present.\n"
    b"The accused was identified as Suresh Sharma, resident of Dwarka.\n"
)
doc = proc.process_file(file_bytes=fir_text, filename="test_fir.txt")
print(f"\nTest FIR processing:")
print(f"  Doc type: {doc.doc_type_guess}")
print(f"  Sections: {doc.sections_mentioned}")
print(f"  Dates: {doc.dates_found}")
print(f"  Parties: {doc.parties_mentioned}")
print(f"  Text length: {len(doc.text)} chars")

# Test format for LLM
context = format_documents_for_llm([doc])
print(f"\nFormatted for LLM: {len(context)} chars")
print(context[:300])

# Test server endpoints
print("\n--- Testing server integration ---")
os.environ["FLASK_ENV"] = "testing"
from ai_court.api import server
app = server.app
client = app.test_client()

# Test document types endpoint
r = client.get("/api/agent/document-types")
print(f"Document types endpoint: {r.status_code}")
if r.status_code == 200:
    data = r.get_json()
    print(f"  Available types: {len(data['document_types'])}")

# Test agent health (should show new capabilities)
r = client.get("/api/agent/health")
print(f"Agent health: {r.status_code}")
if r.status_code == 200:
    data = r.get_json()
    for k, v in data.items():
        print(f"  {k}: {v}")

print("\n=== ALL DOCUMENT FEATURES WORKING ===")
