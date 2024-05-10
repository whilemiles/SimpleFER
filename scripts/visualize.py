import sqlite3
import os
import sys
from datetime import datetime
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline

# os.makedirs('build/db',exist_ok=True)
# os.makedirs('build/visual_pics',exist_ok=True)

os.makedirs('db',exist_ok=True)
os.makedirs('visual_pics',exist_ok=True)

if len(sys.argv) > 1:
    user = sys.argv[1]
    conn = sqlite3.connect('db/'+ user +'.db')
    cur = conn.cursor()
    
    # cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='emotions'")
    # table_exists = cur.fetchone()

    # if not table_exists:
    #     cur.execute('''CREATE TABLE emotions (
    #                     id INTEGER PRIMARY KEY,
    #                     time TIMESTAMP,
    #                     emotion TEXT
    #                     )''')
        
    cur.execute("SELECT * FROM emotions")
    rows = cur.fetchall()
    if len(rows) == 0:
        print("1")
        exit(-1)
    
    mark_data = {
        "Angry" : [],
        "Happy" : [],
        "Sad" : [],
        "Surprise" : [],
        "Fear" : [],
        "Disgust" : []
    }

    cnt = 1;
    for data in rows:
        timestamp = data[1]
        emotion = data[2]
        if emotion == 'angry':
            mark_data["Angry"].append(cnt)
        elif emotion == 'happy':
            mark_data["Happy"].append(cnt)
        elif emotion == 'sad':
            mark_data["Sad"].append(cnt)
        elif emotion == 'surprise':
            mark_data["Surprise"].append(cnt)
        elif emotion == 'fear':
            mark_data["Fear"].append(cnt)
        elif emotion == 'disgust':
            mark_data["Disgust"].append(cnt)
        cnt += 1
        
    cur.close()
    conn.close()

    time_line = [i for i in range(1, cnt + 1)]

    label_data = {label: [] for label in ["Angry", "Happy", "Sad", "Surprise", "Fear", "Disgust"]}

    for label in label_data:
        label_data[label] = [0 for i in range(1, cnt + 1)]

    for lable, mark in mark_data.items():
        intensity = 1;
        last_idx = -1;
        for idx in mark:
            if idx - last_idx == 1:
                intensity += 1
            else:
                intensity = 1
            last_idx = idx
            label_data[lable][idx] = intensity

    smoothed_data = {}
    for label, data in label_data.items():
        x = np.arange(1, len(data) + 1)
        y = np.array(data)
        f = interp1d(x, y, kind='cubic')
        # f = make_interp_spline(x, y)

        x_new = np.linspace(1, len(data), 1000)
        y_smooth = f(x_new)
        y_smooth = np.maximum(y_smooth, 0)
        y_smooth[y_smooth < 0.2] = 0
        smoothed_data[label] = (x_new, y_smooth)

    for label, (x_new, y_smooth) in smoothed_data.items():
        plt.plot(x_new, y_smooth, label=label)

    plt.legend()
    plt.grid(True)

    # plt.show()

    plt.savefig('visual_pics/curve' +  datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png')
    plt.clf()

    emotion_lengths = {emotion: len(indices) for emotion, indices in mark_data.items()}

    labels = list(emotion_lengths.keys())
    sizes = list(emotion_lengths.values())

    plt.pie(sizes, labels=labels, explode=(0,0,0,0,0,0), autopct='%1.1f%%', startangle=140,labeldistance = 1000)
    plt.axis('equal') 
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", title="Emotions", fontsize='medium')

    plt.savefig('visual_pics/pie' +  datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png')
    # plt.show()


