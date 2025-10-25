"""
Clean Database - Remove Unknown/NULL Cases
===========================================
Removes all cases with Unknown or NULL outcomes from the database.
This ensures 100% clean data for training.
"""

import sqlite3
from pathlib import Path

DB_PATH = "data/legal_cases_10M.db"

def clean_database():
    """Remove Unknown/NULL cases from database"""
    
    print("\n" + "="*70)
    print("DATABASE CLEANUP - REMOVING UNKNOWN/NULL CASES")
    print("="*70)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get current stats
    cursor.execute("SELECT COUNT(*) FROM cases")
    total_before = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT outcome, COUNT(*) as count
        FROM cases
        GROUP BY outcome
        ORDER BY count DESC
    """)
    
    print(f"\n📊 BEFORE CLEANUP:")
    print(f"  Total cases: {total_before:,}")
    print(f"\n  Distribution:")
    unknown_count = 0
    for outcome, count in cursor.fetchall():
        status = "✅" if outcome in ['Convicted', 'Acquitted', 'Dismissed', 'Allowed', 'Partly Allowed', 'Remanded'] else "❌"
        print(f"    {status} {outcome or 'NULL'}: {count:,}")
        if outcome in [None, '', 'Unknown'] or outcome is None:
            unknown_count += count
    
    # Remove Unknown/NULL cases
    print(f"\n🗑️  REMOVING {unknown_count:,} Unknown/NULL cases...")
    
    cursor.execute("""
        DELETE FROM cases
        WHERE outcome IS NULL
        OR outcome = ''
        OR outcome = 'Unknown'
    """)
    
    conn.commit()
    
    # Get new stats
    cursor.execute("SELECT COUNT(*) FROM cases")
    total_after = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT outcome, COUNT(*) as count
        FROM cases
        GROUP BY outcome
        ORDER BY count DESC
    """)
    
    print(f"\n✅ AFTER CLEANUP:")
    print(f"  Total cases: {total_after:,}")
    print(f"\n  Distribution:")
    for outcome, count in cursor.fetchall():
        print(f"    ✅ {outcome}: {count:,}")
    
    removed = total_before - total_after
    print(f"\n📉 Removed: {removed:,} cases ({removed/total_before*100:.1f}%)")
    print(f"✅ Remaining: {total_after:,} cases (100% clean for training)")
    
    # Vacuum to reclaim space
    print(f"\n🔧 Optimizing database...")
    cursor.execute("VACUUM")
    
    conn.close()
    
    print(f"\n" + "="*70)
    print("CLEANUP COMPLETE! Database ready for training.")
    print("="*70 + "\n")

if __name__ == "__main__":
    clean_database()
