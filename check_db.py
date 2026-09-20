import sqlite3

conn = sqlite3.connect("customer_service.db")
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print("=== YOUR ACTUAL DATABASE SCHEMA ===\n")
for (table_name,) in tables:
    print(f"Table: {table_name}")
    cursor.execute(f"PRAGMA table_info({table_name});")
    for col in cursor.fetchall():
        pk = " PRIMARY KEY" if col[5] else ""
        print(f"  - {col[1]:<20} {col[2]}{pk}")
    print()

conn.close()