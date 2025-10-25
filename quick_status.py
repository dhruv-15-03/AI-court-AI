"""
Quick Status Check for Continuous Collection
Run this anytime to see current progress
"""

import sqlite3
from datetime import datetime
import os
from pathlib import Path

DB_PATH = "data/legal_cases_10M.db"
LOG_PATH = "logs/continuous_5000.log"

def get_stats():
    if not os.path.exists(DB_PATH):
        print("❌ Database not found!")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total
    cursor.execute("SELECT COUNT(*) FROM cases")
    total = cursor.fetchone()[0]
    
    # Last 5 min
    cursor.execute("SELECT COUNT(*) FROM cases WHERE scraped_at >= datetime('now', '-5 minutes')")
    last_5min = cursor.fetchone()[0]
    
    # Last 30 min
    cursor.execute("SELECT COUNT(*) FROM cases WHERE scraped_at >= datetime('now', '-30 minutes')")
    last_30min = cursor.fetchone()[0]
    
    # Last hour
    cursor.execute("SELECT COUNT(*) FROM cases WHERE scraped_at >= datetime('now', '-1 hour')")
    last_hour = cursor.fetchone()[0]
    
    conn.close()
    
    # Log status
    log_status = "🔴 STOPPED"
    if os.path.exists(LOG_PATH):
        log_file = Path(LOG_PATH)
        last_modified = datetime.fromtimestamp(log_file.stat().st_mtime)
        idle_min = (datetime.now() - last_modified).total_seconds() / 60
        
        if idle_min < 1:
            log_status = "🟢 ACTIVE"
        elif idle_min < 5:
            log_status = "🟡 SLOW"
    
    # Display
    print("\n" + "═" * 70)
    print(f"  CONTINUOUS COLLECTION - {datetime.now().strftime('%I:%M:%S %p')}")
    print("═" * 70)
    
    print(f"\n📊 TOTAL CASES: {total:,}")
    
    print(f"\n📈 COLLECTION RATE:")
    print(f"   Last 5 min:  +{last_5min} cases")
    print(f"   Last 30 min: +{last_30min} cases")
    print(f"   Last hour:   +{last_hour} cases")
    
    if last_30min > 0:
        rate_hour = last_30min * 2
        print(f"\n🚀 CURRENT RATE: ~{rate_hour} cases/hour")
    
    print(f"\n🔄 STATUS: {log_status}")
    
    # Milestones
    milestones = [1000, 2000, 3000, 4000, 5000, 10000, 20000, 50000, 100000]
    next_milestone = None
    for m in milestones:
        if total < m:
            next_milestone = m
            break
    
    if next_milestone:
        remaining = next_milestone - total
        print(f"\n🎯 NEXT MILESTONE: {next_milestone:,} cases")
        print(f"   Remaining: {remaining:,} cases")
        
        if last_hour > 0:
            eta_hours = remaining / last_hour
            if eta_hours < 1:
                print(f"   ETA: ~{eta_hours * 60:.0f} minutes")
            else:
                print(f"   ETA: ~{eta_hours:.1f} hours")
    
    print("═" * 70 + "\n")

if __name__ == "__main__":
    get_stats()
