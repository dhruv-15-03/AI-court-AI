"""
CONTINUOUS COLLECTOR - RUNS UNTIL 5,000 CASES
Keeps repeating queries with randomization until target reached
"""

import requests
from bs4 import BeautifulSoup
import time
import sqlite3
from pathlib import Path
from typing import List
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import re
import random
from urllib.parse import quote
import threading

# Setup
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continuous_5000.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TARGET_CASES = 100000  # Will run until you stop it manually

# Quality tracking statistics
class CollectionStats:
    """Track collection quality metrics"""
    def __init__(self):
        self.lock = threading.Lock()
        self.total_scraped = 0
        self.accepted = 0
        self.rejected_no_outcome = 0
        self.rejected_short_text = 0
        self.rejected_duplicate = 0
    
    def record_accepted(self):
        with self.lock:
            self.total_scraped += 1
            self.accepted += 1
    
    def record_rejected(self, reason: str):
        with self.lock:
            self.total_scraped += 1
            if reason == "no_outcome":
                self.rejected_no_outcome += 1
            elif reason == "short_text":
                self.rejected_short_text += 1
            elif reason == "duplicate":
                self.rejected_duplicate += 1
    
    def get_summary(self):
        with self.lock:
            total = self.total_scraped
            if total == 0:
                return "No cases processed yet"
            
            acceptance_rate = (self.accepted / total) * 100 if total > 0 else 0
            return (f"Scraped: {total} | Accepted: {self.accepted} ({acceptance_rate:.1f}%) | "
                   f"Rejected: No outcome={self.rejected_no_outcome}, "
                   f"Short text={self.rejected_short_text}, Duplicate={self.rejected_duplicate}")

stats = CollectionStats()

class Database:
    """Database handler with auto-stop at target"""
    
    def __init__(self):
        self.db_path = "data/legal_cases_10M.db"
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.buffer = []
        self.buffer_size = 30
        self.stop_collection = False
        self.last_count = 0
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                case_id TEXT PRIMARY KEY,
                title TEXT,
                court TEXT,
                date TEXT,
                case_number TEXT,
                judges TEXT,
                parties TEXT,
                decision_text TEXT,
                outcome TEXT,
                citations TEXT,
                laws_cited TEXT,
                source TEXT,
                url TEXT,
                scraped_at TEXT,
                text_hash TEXT UNIQUE
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcome ON cases(outcome)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_court ON cases(court)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_scraped ON cases(scraped_at)")
        conn.commit()
        conn.close()
    
    def get_current_count(self):
        """Get current case count"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cases")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def add_case(self, case_data):
        """Add case to buffer"""
        if self.stop_collection:
            return False
            
        with self.lock:
            self.buffer.append(case_data)
            
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
            
            return not self.stop_collection
    
    def _flush_buffer(self):
        """Flush buffer to database"""
        if not self.buffer:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            added = 0
            
            for case in self.buffer:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO cases 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, case)
                    if cursor.rowcount > 0:
                        added += 1
                except Exception:
                    continue
            
            conn.commit()
            conn.close()
            
            if added > 0:
                current = self.get_current_count()
                remaining = max(0, TARGET_CASES - current)
                percentage = (current / TARGET_CASES) * 100
                
                logger.info(f"[DB] +{added} | Total: {current} ({percentage:.1f}%) | Remaining: {remaining}")
                
                # Check if we hit target
                if current >= TARGET_CASES:
                    self.stop_collection = True
                    logger.info(f"\n🎯 TARGET REACHED! {current} cases!")
            
            self.buffer = []
            
        except Exception as e:
            logger.error(f"Database error: {e}")
    
    def close(self):
        """Final flush"""
        with self.lock:
            self._flush_buffer()

# Initialize
db = Database()

def get_case_hash(text: str) -> str:
    """Generate hash for deduplication"""
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def extract_case_data(soup, url: str):
    """Extract case data from page"""
    try:
        title = soup.find('h1', class_='doctitle')
        title = title.get_text(strip=True) if title else "Unknown Case"
        
        # Court
        court_elem = soup.find('div', class_='docsource')
        court = court_elem.get_text(strip=True) if court_elem else "Unknown"
        if 'supreme court' in court.lower():
            court = "Supreme Court"
        elif 'high court' in court.lower():
            court = "High Court"
        
        # Date
        meta = soup.find('div', class_='doc_meta')
        date = "Unknown"
        if meta:
            text = meta.get_text()
            date_match = re.search(r'\d{1,2}\s+\w+\s+\d{4}', text)
            if date_match:
                date = date_match.group()
        
        # Decision text
        decision = soup.find('div', class_='judgments')
        if not decision:
            decision = soup.find('div', id='judgment')
        decision_text = decision.get_text(separator=' ', strip=True)[:5000] if decision else ""
        
        # Outcome detection (EXPANDED - capture more legal terminology)
        outcome = None
        text_lower = decision_text.lower()
        
        # Convicted - Expanded keywords
        convicted_keywords = ['convicted', 'conviction upheld', 'conviction confirmed', 
                            'sentenced', 'guilty', 'punishment', 'imprisonment awarded',
                            'sentence affirmed', 'conviction stands', 'held guilty']
        if any(w in text_lower for w in convicted_keywords):
            outcome = "Convicted"
        
        # Acquitted - Expanded keywords
        elif any(w in text_lower for w in ['acquitted', 'acquittal', 'not guilty', 
                                            'discharged', 'exonerated', 'absolved']):
            outcome = "Acquitted"
        
        # Dismissed - Expanded keywords
        elif any(w in text_lower for w in ['appeal dismissed', 'petition dismissed', 
                                            'writ dismissed', 'dismissed', 'rejected',
                                            'cannot be entertained']):
            outcome = "Dismissed"
        
        # Allowed - Expanded keywords
        elif any(w in text_lower for w in ['appeal allowed', 'petition allowed', 
                                            'writ allowed', 'allowed', 'granted',
                                            'prayer granted', 'relief granted']):
            outcome = "Allowed"
        
        # Partly Allowed
        elif any(w in text_lower for w in ['partly allowed', 'partially allowed',
                                            'partly granted', 'partially granted']):
            outcome = "Partly Allowed"
        
        # Remanded
        elif any(w in text_lower for w in ['remanded', 'remand', 'sent back',
                                            'remitted back', 'restored to file']):
            outcome = "Remanded"
        
        # ✅ QUALITY FILTER: Reject cases without clear outcomes
        if outcome is None:
            logger.debug(f"Rejected (no outcome): {title[:50]}...")
            stats.record_rejected("no_outcome")
            return None
        
        # Validation: Minimum text length (reduced from 200 to 100)
        if len(decision_text.strip()) < 100:
            logger.debug(f"Rejected (short text): {title[:50]}...")
            stats.record_rejected("short_text")
            return None
        
        # Generate ID and hash
        case_id = hashlib.sha256(url.encode()).hexdigest()[:16]
        text_hash = get_case_hash(title + decision_text)
        
        # Record successful extraction
        stats.record_accepted()
        
        return (
            case_id, title, court, date, "",
            "", "", decision_text, outcome, "", "",
            "IndianKanoon", url, datetime.now().isoformat(),
            text_hash
        )
        
    except Exception as e:
        return None

def scrape_case(url: str):
    """Scrape single case"""
    if db.stop_collection:
        return None
        
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=8)
        
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.content, 'html.parser')
            case_data = extract_case_data(soup, url)
            
            if case_data:
                if db.add_case(case_data):
                    return case_data
                else:
                    return None  # Target reached
            
    except Exception:
        return None
    
    return None

def get_search_results(query: str, page: int = 1) -> List[str]:
    """Get case URLs from search"""
    if db.stop_collection:
        return []
        
    try:
        url = f"https://indiankanoon.org/search/?formInput={quote(query)}&pagenum={page}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return []
        
        soup = BeautifulSoup(resp.content, 'html.parser')
        links = soup.find_all('a', href=True)
        
        case_urls = []
        for link in links:
            href = link['href']
            if '/doc/' in href and href.startswith('/'):
                full_url = f"https://indiankanoon.org{href}"
                case_urls.append(full_url)
        
        return list(set(case_urls))[:25]  # Max 25 per page
        
    except Exception:
        return []

# MASSIVE EXPANDED QUERIES - Maximum variety for unique cases
BASE_QUERIES = [
    # IPC sections (comprehensive)
    "IPC 302", "IPC 376", "IPC 420", "IPC 498A", "IPC 304B",
    "IPC 307", "IPC 406", "IPC 323", "IPC 506", "IPC 354",
    "IPC 120B", "IPC 201", "IPC 212", "IPC 341", "IPC 363",
    "IPC 392", "IPC 395", "IPC 397", "IPC 452", "IPC 467",
    "IPC 468", "IPC 471", "IPC 34", "IPC 109", "IPC 114",
    "IPC 147", "IPC 148", "IPC 149", "IPC 326", "IPC 379",
    "IPC 380", "IPC 384", "IPC 411", "IPC 427", "IPC 447",
    
    # Crime types (detailed)
    "murder", "rape", "fraud", "corruption", "dowry death",
    "kidnapping", "assault", "theft", "robbery", "extortion",
    "cheating", "forgery", "bribery", "defamation", "harassment",
    "sexual assault", "sexual harassment", "domestic violence", "child abuse",
    "drug trafficking", "money laundering", "cyber crime", "cyberstalking",
    "criminal conspiracy", "criminal intimidation", "attempt to murder",
    "culpable homicide", "grievous hurt", "wrongful confinement",
    "criminal breach of trust", "criminal misappropriation",
    
    # Case outcomes (varied)
    "bail granted", "bail denied", "bail rejected", "bail application",
    "conviction upheld", "conviction reversed", "acquittal", "acquitted",
    "sentence reduced", "sentence enhanced", "sentence suspended",
    "appeal allowed", "appeal dismissed", "appeal pending",
    "anticipatory bail", "regular bail", "interim bail",
    "quashing of charges", "discharge petition", "framing of charges",
    
    # Courts (all major)
    "Supreme Court India", "Supreme Court criminal", "Supreme Court appeal",
    "Delhi High Court", "Bombay High Court", "Madras High Court",
    "Calcutta High Court", "Karnataka High Court", "Gujarat High Court",
    "Rajasthan High Court", "Madhya Pradesh High Court", "Punjab High Court",
    "Allahabad High Court", "Patna High Court", "Orissa High Court",
    
    # Years (comprehensive)
    "criminal 2024", "criminal 2023", "criminal 2022", "criminal 2021", "criminal 2020",
    "criminal 2019", "criminal 2018", "criminal 2017", "criminal 2016", "criminal 2015",
    "judgment 2024", "judgment 2023", "judgment 2022", "judgment 2021",
    
    # Specific legal terms
    "section 313 CrPC", "section 164 CrPC", "section 482 CrPC",
    "NDPS Act", "POCSO Act", "Prevention of Corruption Act",
    "dowry prohibition", "SC ST Act", "arms act",
    "motor vehicle accident", "criminal negligence", "rash driving",
    
    # Case types
    "criminal appeal", "criminal revision", "criminal misc",
    "writ petition criminal", "special leave petition criminal",
    "bail application", "anticipatory bail application",
    "discharge application", "appeal against conviction",
    "appeal against acquittal", "appeal against sentence",
    
    # Specific crimes
    "honor killing", "acid attack", "gang rape", "custodial death",
    "fake encounter", "police torture", "illegal detention",
    "sedition", "rioting", "unlawful assembly", "affray",
    "public nuisance", "trespass", "house breaking",
    
    # Evidence related
    "dying declaration", "circumstantial evidence", "eye witness",
    "forensic evidence", "DNA evidence", "confession",
    "admission", "recovery", "test identification",
]

def worker_task(query: str, start_page: int = 1):
    """Worker task - scrapes until told to stop, optimized for uniqueness"""
    if db.stop_collection:
        return 0
        
    collected = 0
    page = start_page
    max_pages = 30  # Balanced depth for speed
    empty_pages = 0  # Track empty pages
    
    while page <= max_pages and not db.stop_collection:
        urls = get_search_results(query, page)
        
        if not urls:
            empty_pages += 1
            if empty_pages >= 3:  # Stop after 3 empty pages
                break
            page += 1
            continue
        
        empty_pages = 0  # Reset counter
        
        for url in urls:
            if db.stop_collection:
                break
                
            result = scrape_case(url)
            if result:
                collected += 1
            
            # Optimized delay for speed
            time.sleep(random.uniform(0.08, 0.2))
        
        page += 1
        time.sleep(random.uniform(0.15, 0.4))  # Faster page transitions
    
    return collected

def main():
    """Main collection loop - CONTINUOUS until target"""
    
    # Get current count
    current_count = db.get_current_count()
    remaining = TARGET_CASES - current_count
    
    print("\n" + "╔" + "═" * 70 + "╗")
    print("║" + " " * 10 + "CONTINUOUS COLLECTOR - UNLIMITED MODE" + " " * 23 + "║")
    print("╚" + "═" * 70 + "╝")
    print(f"\nCurrent: {current_count} cases")
    print(f"Mode: CONTINUOUS (runs until you stop it)")
    print(f"\nPerformance:")
    print(f"  • Workers: 12 parallel threads (OPTIMIZED)")
    print(f"  • Pages per query: Up to 30 pages (balanced depth)")
    print(f"  • Queries: {len(BASE_QUERIES)} unique queries with randomization")
    print(f"  • Queries per round: 30 (maximum variety)")
    print(f"  • Speed: OPTIMIZED for continuous unique case extraction")
    print(f"\nUniqueness Strategy:")
    print(f"  ✓ Random starting pages (1-10)")
    print(f"  ✓ Shuffled query order each round")
    print(f"  ✓ Hash-based deduplication")
    print(f"  ✓ {len(BASE_QUERIES)} diverse search terms")
    print(f"\nStart: {datetime.now().strftime('%H:%M:%S')}")
    print("\n🔄 COLLECTING CONTINUOUSLY...\n")
    
    round_num = 1
    
    try:
        while not db.stop_collection:
            current = db.get_current_count()
            if current >= TARGET_CASES:
                break
            
            logger.info(f"\n🔄 ROUND {round_num} - Current: {current}/{TARGET_CASES}")
            logger.info(f"📊 Quality Stats: {stats.get_summary()}")
            
            # Shuffle queries for maximum variety and uniqueness
            queries = BASE_QUERIES.copy()
            random.shuffle(queries)
            
            # Use more queries per round for better coverage
            queries_this_round = min(30, len(queries))  # 30 different queries per round
            
            with ThreadPoolExecutor(max_workers=12) as executor:  # Increased workers
                # Submit queries with random starting pages for uniqueness
                futures = []
                for query in queries[:queries_this_round]:
                    if db.stop_collection:
                        break
                    
                    # Random starting page (1-10) for variety
                    start_page = random.randint(1, 10)
                    future = executor.submit(worker_task, query, start_page)
                    futures.append(future)
                    time.sleep(0.15)  # Faster staggering
                
                # Wait for completion
                for future in as_completed(futures):
                    if db.stop_collection:
                        logger.info("\n🎯 TARGET REACHED - Stopping workers...")
                        break
                    
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Worker error: {e}")
            
            round_num += 1
            time.sleep(0.5)  # Shorter pause for continuous flow
    
    except KeyboardInterrupt:
        logger.info("\n⚠️ Collection stopped by user")
    
    finally:
        # Final flush
        db.close()
        
        # Final stats
        final_count = db.get_current_count()
        
        print("\n" + "╔" + "═" * 70 + "╗")
        print("║" + " " * 25 + "COLLECTION COMPLETE!" + " " * 26 + "║")
        print("╚" + "═" * 70 + "╝")
        print(f"\nResults:")
        print(f"  • Total cases in DB: {final_count}")
        print(f"  • New cases collected: {final_count - current_count}")
        print(f"  • Rounds completed: {round_num - 1}")
        print(f"\n📊 Quality Metrics:")
        print(f"  {stats.get_summary()}")
        acceptance_rate = (stats.accepted / stats.total_scraped * 100) if stats.total_scraped > 0 else 0
        print(f"  • Acceptance Rate: {acceptance_rate:.1f}%")
        print(f"  • All cases have CLEAR OUTCOMES (no Unknown/NULL) ✅")
        
        if final_count >= TARGET_CASES:
            print(f"\n🎯 TARGET ACHIEVED! {final_count} cases collected!")
            print("\n✅ Ready for GPU training!")
        else:
            remaining = TARGET_CASES - final_count
            print(f"\n⚠️ {remaining} cases remaining to reach target.")

if __name__ == "__main__":
    main()
