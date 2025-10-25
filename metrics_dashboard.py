"""
Real-Time Metrics Dashboard for AI-Court Project
Displays comprehensive metrics for collection, training, and model performance
"""

import sqlite3
import os
import json
from datetime import datetime
from pathlib import Path

DB_PATH = "data/legal_cases_10M.db"
METRICS_PATH = "models/metrics.json"
METADATA_PATH = "models/production/training_metadata.json"

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_section(title):
    """Print section header"""
    print(f"\n{'─' * 80}")
    print(f"  {title}")
    print(f"{'─' * 80}")

def get_collection_metrics():
    """Get data collection metrics"""
    if not os.path.exists(DB_PATH):
        return None
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total cases
    cursor.execute("SELECT COUNT(*) FROM cases")
    total = cursor.fetchone()[0]
    
    # By court
    cursor.execute("""
        SELECT court, COUNT(*) as count
        FROM cases 
        GROUP BY court 
        ORDER BY count DESC 
        LIMIT 5
    """)
    courts = cursor.fetchall()
    
    # By outcome
    cursor.execute("""
        SELECT outcome, COUNT(*) as count
        FROM cases 
        GROUP BY outcome 
        ORDER BY count DESC 
        LIMIT 5
    """)
    outcomes = cursor.fetchall()
    
    # Recent activity
    cursor.execute("SELECT COUNT(*) FROM cases WHERE scraped_at >= datetime('now', '-1 hour')")
    last_hour = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM cases WHERE scraped_at >= datetime('now', '-24 hours')")
    last_day = cursor.fetchone()[0]
    
    # Date range
    cursor.execute("SELECT MIN(scraped_at), MAX(scraped_at) FROM cases")
    date_range = cursor.fetchone()
    
    conn.close()
    
    return {
        'total': total,
        'courts': courts,
        'outcomes': outcomes,
        'last_hour': last_hour,
        'last_day': last_day,
        'date_range': date_range
    }

def get_model_metrics():
    """Get model performance metrics"""
    if not os.path.exists(METRICS_PATH):
        return None
    
    with open(METRICS_PATH, 'r') as f:
        return json.load(f)

def get_training_metadata():
    """Get latest training metadata"""
    if not os.path.exists(METADATA_PATH):
        return None
    
    with open(METADATA_PATH, 'r') as f:
        return json.load(f)

def calculate_milestones(total):
    """Calculate progress to milestones"""
    milestones = [1000, 5000, 10000, 20000, 50000, 100000]
    
    completed = [m for m in milestones if total >= m]
    next_milestone = None
    for m in milestones:
        if total < m:
            next_milestone = m
            break
    
    return completed, next_milestone

def main():
    """Display comprehensive metrics dashboard"""
    
    print_header("📊 AI-COURT METRICS DASHBOARD")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========== COLLECTION METRICS ==========
    print_section("📈 DATA COLLECTION METRICS")
    
    collection = get_collection_metrics()
    if collection:
        total = collection['total']
        print(f"\n🎯 TOTAL CASES: {total:,}")
        
        # File size
        if os.path.exists(DB_PATH):
            size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
            print(f"📦 DATABASE SIZE: {size_mb:.2f} MB")
        
        # Collection rate
        if collection['last_hour'] > 0:
            print(f"⚡ LAST HOUR: +{collection['last_hour']} cases (~{collection['last_hour']} cases/hour)")
        if collection['last_day'] > 0:
            rate_per_hour = collection['last_day'] / 24
            print(f"📅 LAST 24 HOURS: +{collection['last_day']} cases (~{rate_per_hour:.0f} cases/hour avg)")
        
        # Date range
        if collection['date_range'][0]:
            print(f"📆 COLLECTION PERIOD: {collection['date_range'][0][:10]} to {collection['date_range'][1][:10]}")
        
        # Milestones
        completed, next_milestone = calculate_milestones(total)
        print(f"\n✅ MILESTONES COMPLETED: {', '.join([f'{m:,}' for m in completed])}")
        if next_milestone:
            remaining = next_milestone - total
            percentage = (total / next_milestone) * 100
            print(f"🎯 NEXT MILESTONE: {next_milestone:,} cases")
            print(f"   Progress: {total:,} / {next_milestone:,} ({percentage:.1f}%)")
            print(f"   Remaining: {remaining:,} cases")
            if collection['last_hour'] > 0:
                eta_hours = remaining / collection['last_hour']
                if eta_hours < 1:
                    print(f"   ETA: ~{eta_hours * 60:.0f} minutes")
                else:
                    print(f"   ETA: ~{eta_hours:.1f} hours")
        
        # Court distribution
        print(f"\n🏛️  TOP COURTS:")
        for court, count in collection['courts']:
            percentage = (count / total) * 100
            bar_length = int(percentage / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"   {court or 'Unknown':30s} [{bar}] {count:5,} ({percentage:5.1f}%)")
        
        # Outcome distribution
        print(f"\n⚖️  TOP OUTCOMES:")
        for outcome, count in collection['outcomes']:
            percentage = (count / total) * 100
            bar_length = int(percentage / 2)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            print(f"   {outcome or 'Unknown':30s} [{bar}] {count:5,} ({percentage:5.1f}%)")
    else:
        print("\n⚠️  No collection data available")
    
    # ========== MODEL METRICS ==========
    print_section("🤖 MODEL PERFORMANCE METRICS")
    
    training_meta = get_training_metadata()
    if training_meta:
        print(f"\n📊 LATEST MODEL:")
        print(f"   Timestamp: {training_meta.get('timestamp', 'N/A')}")
        print(f"   Training Cases: {training_meta.get('total_cases', 'N/A'):,}")
        print(f"   Device: {training_meta.get('device', 'N/A')}")
        print(f"   Model Path: {training_meta.get('model_path', 'N/A')}")
        
        print(f"\n🎯 PERFORMANCE:")
        accuracy = training_meta.get('accuracy', 0) * 100
        f1_macro = training_meta.get('f1_macro', 0) * 100
        f1_weighted = training_meta.get('f1_weighted', 0) * 100
        
        print(f"   Accuracy:     {accuracy:6.2f}% {'✅' if accuracy >= 70 else '⚠️' if accuracy >= 60 else '❌'}")
        print(f"   F1 Weighted:  {f1_weighted:6.2f}% {'✅' if f1_weighted >= 70 else '⚠️' if f1_weighted >= 45 else '❌'}")
        print(f"   F1 Macro:     {f1_macro:6.2f}% {'✅' if f1_macro >= 60 else '⚠️' if f1_macro >= 20 else '❌'}")
        
        # Targets
        print(f"\n🎯 TARGETS:")
        print(f"   Accuracy:     70%+ (Current: {accuracy:.1f}%)")
        print(f"   F1 Weighted:  70%+ (Current: {f1_weighted:.1f}%)")
        print(f"   F1 Macro:     60%+ (Current: {f1_macro:.1f}%)")
    else:
        print("\n⚠️  No model metrics available")
    
    # ========== SYSTEM STATUS ==========
    print_section("⚙️  SYSTEM STATUS")
    
    # Check critical files
    files_status = {
        "Database": (DB_PATH, os.path.exists(DB_PATH)),
        "Model": ("models/legal_case_classifier.pkl", os.path.exists("models/legal_case_classifier.pkl")),
        "Search Index": ("models/search_index.pkl", os.path.exists("models/search_index.pkl")),
        "Collector Script": ("scripts/continuous_collector.py", os.path.exists("scripts/continuous_collector.py")),
        "Trainer Script": ("scripts/pipeline/batch_trainer.py", os.path.exists("scripts/pipeline/batch_trainer.py")),
        "API Server": ("app.py", os.path.exists("app.py")),
    }
    
    print()
    for name, (path, exists) in files_status.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {name:20s} {path if exists else 'MISSING'}")
    
    # ========== RECOMMENDATIONS ==========
    print_section("💡 RECOMMENDATIONS")
    
    if collection:
        total = collection['total']
        
        if total < 5000:
            print("\n   📌 Continue collecting to 5,000 cases for better model")
            print("   📌 Expected accuracy at 5,000: 70%+")
        elif total < 10000:
            print("\n   ✅ Good dataset size! Consider training Model #2")
            print(f"   📌 Command: python scripts/pipeline/batch_trainer.py --batch_size {min(1000, total)} --force")
            print("   📌 Expected accuracy: 70-75%")
        elif total < 50000:
            print("\n   ✅ Excellent dataset! Train for production use")
            print(f"   📌 Command: python scripts/pipeline/batch_trainer.py --batch_size {min(5000, total)} --force")
            print("   📌 Expected accuracy: 75-80%")
        else:
            print("\n   🎉 Enterprise-ready dataset!")
            print("   📌 Deploy to production")
            print("   📌 Expected accuracy: 80-85%")
        
        # Training recommendation
        if training_meta:
            training_cases = training_meta.get('total_cases', 0)
            if total > training_cases * 2:
                print(f"\n   ⚠️  Model trained on {training_cases:,} cases, but {total:,} cases available")
                print("   📌 Retrain model with new data for better accuracy!")
    
    print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
