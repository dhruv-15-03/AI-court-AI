import requests
import re
import time
import os
import random
import logging
import pandas as pd
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("legal_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

BASE_URL = "https://indiankanoon.org"
SEARCH_URL = f"{BASE_URL}/search/?formInput="

# Configuration for Hugging Face Inference API
# Replace with your actual Hugging Face API token
HUGGINGFACE_API_TOKEN = "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"
# You can use different summarization models by changing this URL
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

def get_case_links(query, pages=1):
    """Get links to cases using only requests (no browser automation)"""
    links = []
    encoded_query = quote_plus(query)

    for page in range(1, pages + 1):
        url = f"{SEARCH_URL}{encoded_query}&pagenum={page}"
        logger.info(f"🔍 Searching page {page}: {url}")

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'DNT': '1',  # Do Not Track
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }

        try:
            # Add timeout and retries
            for attempt in range(3):
                try:
                    response = requests.get(url, headers=headers, timeout=15)
                    break
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    logger.warning(f"⚠️ Request attempt {attempt+1} failed: {e}")
                    if attempt < 2:  # Don't sleep after last attempt
                        time.sleep(3)

            if response.status_code == 200:
                logger.info(f"✅ Got page content (length: {len(response.text)} bytes)")

                # Save HTML for debugging
                with open(f"debug_page_{page}.html", "w", encoding="utf-8") as f:
                    f.write(response.text)

                # First try to extract using BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")

                # Try multiple selectors
                result_blocks = []

                selectors = [
                    "div.result_title > a",
                    ".result_title a",
                    "a.result_title",
                    ".docsearch-result a",
                    ".search_result a"
                ]

                for selector in selectors:
                    result_blocks = soup.select(selector)
                    if result_blocks:
                        logger.info(f"✅ Found results with selector: {selector}")
                        break

                page_links = []
                for tag in result_blocks:
                    href = tag.get("href", "")
                    if href and href.startswith("/doc/") and "fragment" not in href:
                        full_url = BASE_URL + href
                        logger.info(f"Found case link: {full_url}")
                        if full_url not in [item.get("url") for item in links]:
                            page_links.append({
                                "url": full_url,
                                "title": tag.get_text(strip=True)
                            })

                # If that didn't work, use regex as a backup
                if not page_links:
                    logger.info("Using regex fallback to find links")
                    # Pattern to match document links and their titles
                    doc_links = re.findall(r'href="(/doc/[^"]+)".*?>(.*?)</a>', response.text)

                    for href, title in doc_links:
                        if "fragment" not in href:
                            full_url = BASE_URL + href
                            if full_url not in [item.get("url") for item in links]:
                                page_links.append({
                                    "url": full_url,
                                    "title": BeautifulSoup(title, "html.parser").get_text(strip=True)
                                })

                links.extend(page_links)
                logger.info(f"✅ Found {len(page_links)} cases on page {page}")
            else:
                logger.error(f"❌ HTTP error: {response.status_code}")

        except Exception as e:
            logger.error(f"❌ Error processing page {page}: {e}")

        if page < pages:
            delay = random.uniform(2, 4)
            logger.info(f"Waiting {delay:.1f} seconds before next page...")
            time.sleep(delay)

    return links

def get_case_content(url):
    logger.info(f"📄 Getting case: {url}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    try:
        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers, timeout=45)  # Extended timeout for large documents
                break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                logger.warning(f"⚠️ Request attempt {attempt+1} failed: {e}")
                if attempt < 2:  # Don't sleep after last attempt
                    time.sleep(3)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            title_element = soup.select_one("div.judgments_title")
            title = title_element.text.strip() if title_element else "Unknown Case"
            judgment_text = ""
            judgment_element = soup.find("pre")
            if judgment_element:
                judgment_text = judgment_element.get_text()
            if not judgment_text or len(judgment_text) < 500:
                for selector in ["#judgments_text", ".judgments_text", "div.judgments", "div.doc-content"]:
                    element = soup.select_one(selector)
                    if element:
                        judgment_text = element.get_text()
                        if len(judgment_text) > 500:  # Found substantial text
                            break
            if not judgment_text or len(judgment_text) < 500:
                main_content = soup.select_one("div.judments_container, div.judgments_container, div.content, main")
                if main_content:
                    judgment_text = main_content.get_text()
            if not judgment_text or len(judgment_text) < 500:
                body = soup.find("body")
                if body:
                    for tag in body.select("nav, header, footer, script, style"):
                        tag.decompose()
                    judgment_text = body.get_text()
            if judgment_text:
                judgment_text = re.sub(r'\n\s*\n', '\n\n', judgment_text)
                judgment_text = judgment_text.strip()
                safe_title = re.sub(r'[^\w]+', '_', title[:30])
                file_name = f"debug_case_{safe_title}.txt"
                with open(file_name, "w", encoding="utf-8") as f:
                    f.write(judgment_text)

                return {
                    "title": title,
                    "text": judgment_text,
                    "url": url
                }
            else:
                logger.warning(f"⚠️ No judgment text found at {url}")
                return {
                    "title": title,
                    "text": "Judgment text not found",
                    "url": url
                }
        else:
            logger.error(f"❌ HTTP error: {response.status_code}")
            return None

    except Exception as e:
        logger.error(f"❌ Error fetching case: {e}")
        return None

def extract_judgment_section(text):
    if not text or text == "Judgment text not found":
        return "No judgment text available"
    judgment_patterns = [
        r'(?:ORDER|JUDGMENT|CONCLUSION|HELD|THEREFORE)(.*?)(?:\n\n|$)',
        r'(?:we\s+hold|we\s+direct|it\s+is\s+ordered|it\s+is\s+hereby|accordingly|the\s+appeal\s+is|the\s+petition\s+is|we\s+find\s+that)(.*?)(?:\n\n|$)',
        r'(?:for\s+the\s+foregoing\s+reasons|in\s+view\s+of\s+the\s+above|in\s+the\s+light\s+of\s+the\s+above)(.*?)(?:\n\n|$)'
    ]
    judgment_texts = []
    for pattern in judgment_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            judgment_texts.append(match.group(0).strip())

    if judgment_texts:
        final_judgment = " ".join(judgment_texts[-3:])
        return final_judgment

    lines = text.split("\n")
    last_lines = [line.strip() for line in lines[-30:] if line.strip()][-15:]
    return " ".join(last_lines)

def get_case_summary(case_text, target_length=1500):
    try:
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        max_input_length = 1024  # Most models have input length limitations
        truncated_text = case_text[:max_input_length] if len(case_text) > max_input_length else case_text

        payload = {
            "inputs": truncated_text,
            "parameters": {
                "max_length": min(200, target_length // 5),  # Approx 200 words ≈ 1500 characters
                "min_length": 50,
                "do_sample": False
            }
        }

        response = requests.post(
            HUGGINGFACE_API_URL,
            json=payload,
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            # Different models might have different response formats
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and "summary_text" in result[0]:
                    summary = result[0]["summary_text"]
                else:
                    summary = result[0]
            elif isinstance(result, dict) and "summary_text" in result:
                summary = result["summary_text"]
            else:
                summary = str(result)

            if summary:
                return summary
            else:
                logger.warning("⚠️ API returned no summary")
        else:
            logger.error(f"❌ API error: {response.status_code}, Message: {response.text}")

    except Exception as e:
        logger.error(f"❌ Error with Hugging Face API: {e}")
    logger.info("Using fallback summarization method")
    sentences = re.split(r'(?<=[.!?]) +', case_text)
    intro = ' '.join(sentences[:5])
    judgment_section = extract_judgment_section(case_text)
    return f"{intro}\n\n[...]\n\n{judgment_section}"

def create_dataset(query, pages, output_csv):

    case_links = get_case_links(query, pages)

    if not case_links:
        logger.error("❌ No case links found")
        return 0

    logger.info(f"✅ Found {len(case_links)} cases. Getting details...")
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    data = []
    for i, case in enumerate(case_links):
        logger.info(f"Processing case {i+1}/{len(case_links)}: {case['title']}")

        try:
            # Get case content
            case_data = get_case_content(case['url'])

            if case_data and case_data['text'] != "Judgment text not found":
                # Extract judgment section
                judgment_text = extract_judgment_section(case_data['text'])

                # Get case summary - try to process the full text in chunks if needed
                full_text = case_data['text']
                case_summary = ""
                if len(full_text) > 10000:
                    logger.info(f"Long text detected ({len(full_text)} chars), processing in sections")
                    beginning = full_text[:3000]
                    beginning_summary = get_case_summary(beginning)
                    judgment_summary = get_case_summary(judgment_text) if len(judgment_text) > 100 else judgment_text
                    case_summary = f"{beginning_summary}\n\n[...]\n\n{judgment_summary}"
                else:
                    case_summary = get_case_summary(full_text)
                record = {
                    "id": i + 1,
                    "title": case_data['title'],
                    "url": case['url'],
                    "case_summary": case_summary,
                    "judgment": judgment_text
                }

                data.append(record)
                logger.info(f"✅ Successfully processed case {i+1}")
                if i % 5 == 0 and i > 0:
                    temp_df = pd.DataFrame(data)
                    temp_df.to_csv(f"temp_{output_csv}", index=False)
                    logger.info(f"💾 Saved intermediate results ({len(data)} cases)")
            else:
                logger.warning(f"⚠️ Skipping case {i+1} due to missing content")

        except Exception as e:
            logger.error(f"❌ Error processing case {i+1}: {e}")

        if i < len(case_links) - 1:
            delay = random.uniform(2, 4)
            time.sleep(delay)
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        logger.info(f"💾 Successfully saved {len(data)} cases to '{output_csv}'")
    else:
        logger.error("❌ No data collected, dataset not created")

    return len(data)
def setup_local_summarizer():
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization")
        return summarizer
    except ImportError:
        logger.warning("⚠️ Transformers library not available. API or fallback method will be used.")
        return None

def summarize_with_local_model(text, summarizer, max_length=15000):
    if not summarizer:
        return None

    try:
        max_chunk_length = 10240
        if len(text) > max_chunk_length:
            chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
            summaries = []

            for chunk in chunks[:3]:
                summary = summarizer(chunk, max_length=max_length//3, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])

            return " ".join(summaries)
        else:
            summary = summarizer(text, max_length=max_length, min_length=100, do_sample=False)
            return summary[0]['summary_text']
    except Exception as e:
        logger.error(f"❌ Error with local summarization: {e}")
        return None

if __name__ == "__main__":
    QUERY = "cases on rape and murder"
    PAGES = 2
    OUTPUT_CSV = "data/rape_murder_cases_summarized.csv"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("legal_scraper.log"),
            logging.StreamHandler()
        ]
    )
    os.makedirs(os.path.dirname(OUTPUT_CSV) if os.path.dirname(OUTPUT_CSV) else '.', exist_ok=True)

    try:
        logger.info(f"🚀 Starting legal case scraper for query: '{QUERY}'")
        logger.info(f"Using Hugging Face model: {HUGGINGFACE_API_URL}")
        test_text = "This is a test to verify the Hugging Face API connection."
        try:
            headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
            response = requests.post(
                HUGGINGFACE_API_URL,
                json={"inputs": test_text},
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                logger.info("✅ Hugging Face API connection successful")
            else:
                logger.warning(f"⚠️ Hugging Face API test failed: {response.status_code}, {response.text}")
        except Exception as e:
            logger.warning(f"⚠️ Hugging Face API test failed: {e}")

        total_cases = create_dataset(QUERY, PAGES, OUTPUT_CSV)
        logger.info(f"🏁 Scraping completed - {total_cases} cases processed")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")