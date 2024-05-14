import sqlite3
import os
import sys
from datetime import datetime

def insert_emotion(emotion):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("INSERT INTO emotions (time, emotion) VALUES (?, ?)", (current_time, emotion))
    conn.commit()

def get_emotions():
    cur.execute("SELECT * FROM emotions")
    rows = cur.fetchall()
    for row in rows:
        print(row)
        
os.makedirs('db',exist_ok=True)

if len(sys.argv) > 2:
    user = sys.argv[1]
    emotion = sys.argv[2]
    conn = sqlite3.connect('db/'+ user +'.db')
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='emotions'")
    table_exists = cur.fetchone()

    if not table_exists:
        cur.execute('''CREATE TABLE emotions (
                        id INTEGER PRIMARY KEY,
                        time TIMESTAMP,
                        emotion TEXT
                        )''')
    insert_emotion(emotion)
    cur.close()
    conn.close()
else:
    print("Error: arg incorrcet")
# get_emotions()


