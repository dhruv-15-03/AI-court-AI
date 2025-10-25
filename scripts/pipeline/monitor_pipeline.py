"""
Real-Time Pipeline Monitor
=========================
Professional dashboard for monitoring 5-hour training cycles.

Shows:
- Current cycle progress
- Cases collected
- Training status
- GPU utilization
- ETA to next batch

Usage:
    python monitor_pipeline.py
"""

import os
import sys
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import json

# Setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "data" / "legal_cases_10M.db"
LOGS_DIR = PROJECT_ROOT / "logs"


def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_case_stats():
    """Get case statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Total cases
        cursor.execute("SELECT COUNT(*) FROM cases")
        total = cursor.fetchone()[0]
        
        # By court
        cursor.execute("""
            SELECT court, COUNT(*) 
            FROM cases 
            GROUP BY court 
            ORDER BY COUNT(*) DESC 
            LIMIT 5
        """)
        courts = cursor.fetchall()
        
        # By outcome
        cursor.execute("""
            SELECT outcome, COUNT(*) 
            FROM cases 
            GROUP BY outcome 
            ORDER BY COUNT(*) DESC
        """)
        outcomes = cursor.fetchall()
        
        # Recent cases
        cursor.execute("""
            SELECT COUNT(*) 
            FROM cases 
            WHERE scraped_at > datetime('now', '-1 hour')
        """)
        recent_1h = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM cases 
            WHERE scraped_at > datetime('now', '-5 hour')
        """)
        recent_5h = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total': total,
            'courts': courts,
            'outcomes': outcomes,
            'recent_1h': recent_1h,
            'recent_5h': recent_5h
        }
    except Exception as e:
        return None


def get_training_status():
    """Get training status"""
    metadata_file = PROJECT_ROOT / "models" / "production" / "training_metadata.json"
    
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return metadata
        except:
            return None
    return None


def get_latest_cycle_log():
    """Get latest cycle log"""
    cycle_logs = list(LOGS_DIR.glob("cycle_*.json"))
    
    if cycle_logs:
        latest_log = max(cycle_logs, key=lambda p: p.stat().st_mtime)
        try:
            with open(latest_log, 'r') as f:
                return json.load(f)
        except:
            return None
    return None


def format_number(n):
    """Format number with commas"""
    return f"{n:,}"


def display_dashboard():
    """Display real-time dashboard"""
    clear_screen()
    
    # Header
    print("=" * 80)
    print(" " * 25 + "AI COURT - PIPELINE MONITOR")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")
    
    # Case Statistics
    stats = get_case_stats()
    if stats:
        print("📊 CASE STATISTICS")
        print("-" * 80)
        print(f"Total Cases:        {format_number(stats['total'])}")
        print(f"Last 1 Hour:        {format_number(stats['recent_1h'])} cases")
        print(f"Last 5 Hours:       {format_number(stats['recent_5h'])} cases")
        
        if stats['recent_1h'] > 0:
            rate_per_hour = stats['recent_1h']
            print(f"Collection Rate:    ~{format_number(rate_per_hour)} cases/hour")
            
            # ETA to milestones
            to_1k = max(0, 1000 - stats['total'])
            to_10k = max(0, 10000 - stats['total'])
            to_100k = max(0, 100000 - stats['total'])
            
            if rate_per_hour > 0:
                eta_1k = to_1k / rate_per_hour if to_1k > 0 else 0
                eta_10k = to_10k / rate_per_hour if to_10k > 0 else 0
                eta_100k = to_100k / rate_per_hour if to_100k > 0 else 0
                
                print("")
                print("🎯 MILESTONES")
                print("-" * 80)
                if to_1k > 0:
                    print(f"To 1K cases:        {to_1k} cases ({eta_1k:.1f} hours)")
                else:
                    print("To 1K cases:        ✓ COMPLETE")
                    
                if to_10k > 0:
                    print(f"To 10K cases:       {to_10k} cases ({eta_10k:.1f} hours / {eta_10k/24:.1f} days)")
                else:
                    print("To 10K cases:       ✓ COMPLETE")
                    
                if to_100k > 0:
                    print(f"To 100K cases:      {to_100k} cases ({eta_100k/24:.1f} days)")
                else:
                    print("To 100K cases:      ✓ COMPLETE")
        
        print("")
        print("🏛 TOP COURTS")
        print("-" * 80)
        for court, count in stats['courts'][:5]:
            pct = (count / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"{court[:40]:40} {format_number(count):>8} ({pct:5.1f}%)")
        
        print("")
        print("⚖️ OUTCOMES")
        print("-" * 80)
        for outcome, count in stats['outcomes']:
            pct = (count / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"{outcome[:40]:40} {format_number(count):>8} ({pct:5.1f}%)")
    else:
        print("❌ Database not accessible")
    
    # Training Status
    print("")
    training = get_training_status()
    if training:
        print("🎓 TRAINING STATUS")
        print("-" * 80)
        print(f"Last Trained:       {training.get('timestamp', 'Unknown')}")
        print(f"Cases Used:         {format_number(training.get('total_cases', 0))}")
        print(f"Accuracy:           {training.get('accuracy', 0):.4f}")
        print(f"Macro F1:           {training.get('f1_macro', 0):.4f}")
        print(f"Weighted F1:        {training.get('f1_weighted', 0):.4f}")
        print(f"Model Path:         {training.get('model_path', 'Unknown')}")
    else:
        print("🎓 TRAINING STATUS")
        print("-" * 80)
        print("No training completed yet")
    
    # Cycle Status
    print("")
    cycle = get_latest_cycle_log()
    if cycle:
        print("🔄 LATEST CYCLE")
        print("-" * 80)
        print(f"Cycle Number:       #{cycle.get('cycle_number', 'Unknown')}")
        print(f"Start Time:         {cycle.get('start_time', 'Unknown')}")
        print(f"Duration:           {cycle.get('duration_hours', 0):.2f} hours")
        print(f"Status:             {'✓ SUCCESS' if cycle.get('success') else '✗ FAILED'}")
    else:
        print("🔄 CYCLE STATUS")
        print("-" * 80)
        print("No cycles completed yet")
    
    # Footer
    print("")
    print("=" * 80)
    print("Press Ctrl+C to exit | Updates every 30 seconds")
    print("=" * 80)


def main():
    """Main monitoring loop"""
    print("Starting pipeline monitor...")
    print("Press Ctrl+C to exit")
    time.sleep(2)
    
    try:
        while True:
            display_dashboard()
            time.sleep(30)  # Update every 30 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")


if __name__ == "__main__":
    main()
