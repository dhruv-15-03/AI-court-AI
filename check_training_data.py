import sqlite3

conn = sqlite3.connect('data/legal_cases_10M.db')
cursor = conn.cursor()

# Total cases
cursor.execute('SELECT COUNT(*) FROM cases')
total = cursor.fetchone()[0]
print(f'Total cases in DB: {total:,}')

# Cases by outcome
cursor.execute('''
    SELECT outcome, COUNT(*) as count 
    FROM cases 
    GROUP BY outcome 
    ORDER BY count DESC
''')
print('\nCases by outcome:')
for outcome, count in cursor.fetchall():
    print(f'  {outcome or "NULL/Unknown"}: {count:,}')

# Cases with valid outcomes (not NULL/Unknown)
cursor.execute('''
    SELECT COUNT(*) 
    FROM cases 
    WHERE outcome IS NOT NULL 
    AND outcome != '' 
    AND outcome != 'Unknown'
''')
valid_cases = cursor.fetchone()[0]
print(f'\nCases with valid outcomes: {valid_cases:,}')

# Check what the trainer loaded
cursor.execute('''
    SELECT outcome, COUNT(*) as count
    FROM cases
    WHERE outcome IN ('Convicted', 'Acquitted', 'Dismissed', 'Allowed', 'Partly Allowed', 'Remanded')
    GROUP BY outcome
    ORDER BY count DESC
''')
print('\nCases loaded by trainer (6 outcomes):')
trainer_total = 0
for outcome, count in cursor.fetchall():
    print(f'  {outcome}: {count:,}')
    trainer_total += count
print(f'\nTotal trained on: {trainer_total:,}')
print(f'Missing cases: {total - trainer_total:,}')

conn.close()
