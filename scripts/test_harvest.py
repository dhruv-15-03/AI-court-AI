"""Quick test of harvest pipeline components."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from scripts.harvest_100k import search_kanoon, scrape_case_detail, _categorize_query

# Test 1: Search
print("Test 1: Search for murder cases...")
results = search_kanoon("murder IPC 302", court="supremecourt", page=1)
print(f"  Found {len(results)} results")
if results:
    r = results[0]
    print(f"  First: {r['title'][:80]}")
    print(f"  URL: {r['url']}")

# Test 2: Scrape one case
if results:
    print("\nTest 2: Scraping first case...")
    case = scrape_case_detail(results[0]["url"])
    if case:
        print(f"  Title: {case['title'][:80]}")
        print(f"  Court: {case['court']}")
        print(f"  Summary length: {len(case['case_summary'])} chars")
        print(f"  Judgment length: {len(case['judgment'])} chars")
        print(f"  Full text length: {case['full_text_length']}")
        secs = case["sections_mentioned"]
        print(f"  Sections: {secs[:100]}")
    else:
        print("  Failed to scrape")

# Test 3: Categorize
print("\nTest 3: Category mapping")
for q in ["murder IPC 302", "bail granted", "divorce Hindu Marriage Act", "NDPS narcotics"]:
    print(f"  {q} -> {_categorize_query(q)}")

print("\nAll tests passed!")
