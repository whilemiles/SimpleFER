import sqlite3
import os
import sys

os.makedirs('db',exist_ok=True)

def new_user(name, password):
    cur.execute("SELECT COUNT(*) FROM users WHERE name=?", (name,))
    user_count = cur.fetchone()[0]
    
    if user_count > 0:
        print("User with name '{}' already exists.".format(name))
        exit(101)
    else:
        cur.execute("INSERT INTO users (name, password) VALUES (?, ?)", (name, password))
        conn.commit()
        print("User '{}' added successfully.".format(name))


if len(sys.argv) > 2:
    userName = sys.argv[1]
    password = sys.argv[2]
    
    conn = sqlite3.connect('db/userManagement.db')
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    table_exists = cur.fetchone()

    if not table_exists:
        cur.execute('''CREATE TABLE users (
                        id INTEGER PRIMARY KEY,
                        name TEXT,
                        password TEXT
                        )''')
        
    new_user(userName, password)
    
exit(0)