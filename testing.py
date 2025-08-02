import sqlite3
import numpy as np

conn = sqlite3.connect("faces.db")
cursor = conn.cursor()

cursor.execute("SELECT user_id, frontal_emb, left_emb, right_emb FROM faces")
rows = cursor.fetchall()

for row in rows:
    print("User:", row[0], "| Registered at:", row[1])

