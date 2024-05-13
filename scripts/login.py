import sqlite3
import os
import sys

if len(sys.argv) > 2:
    userName = sys.argv[1]
    password = sys.argv[2]
    conn = sqlite3.connect('db/userManagement.db')
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM users WHERE name=? AND password=?", (userName, password))
    user_count = cur.fetchone()[0]
    
    if user_count > 0:
        print("Authentication successful. Welcome, {}!".format(userName))
        exit(0)
    else:
        print("Authentication failed. Invalid username or password.")
        exit(102)