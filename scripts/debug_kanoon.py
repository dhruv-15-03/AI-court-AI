"""Debug Indian Kanoon search page."""
import requests
from bs4 import BeautifulSoup

url = "https://indiankanoon.org/search/?formInput=murder+IPC+302&pagenum=1"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
}
r = requests.get(url, headers=headers, timeout=30)
print(f"Status: {r.status_code}")
print(f"Length: {len(r.text)}")

soup = BeautifulSoup(r.text, "html.parser")

# Try various selectors
for sel in ["div.result_title a", ".result_title a", "a.result_title", "div.result a"]:
    tags = soup.select(sel)
    print(f"Selector '{sel}': {len(tags)} results")
    for t in tags[:3]:
        print(f"  {t.get('href','')} -> {t.get_text(strip=True)[:60]}")

# Find all /doc/ links
all_doc_links = soup.find_all("a", href=lambda h: h and "/doc/" in h)
print(f"\nAll /doc/ links: {len(all_doc_links)}")
for a in all_doc_links[:5]:
    print(f"  {a.get('href','')} -> {a.get_text(strip=True)[:60]}")

# Check for blocking
text_lower = r.text.lower()
if "captcha" in text_lower:
    print("\nWARNING: CAPTCHA detected!")
if "robot" in text_lower:
    print("\nWARNING: Robot detection!")
if "javascript" in text_lower and "enable" in text_lower:
    print("\nWARNING: JavaScript required!")

# Save debug HTML
with open("logs/debug_kanoon_search.html", "w", encoding="utf-8") as f:
    f.write(r.text)
print("\nDebug HTML saved to logs/debug_kanoon_search.html")
