# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

raw_data = pd.read_csv('inputs/trending.csv')
raw_categories = pd.read_json('inputs/categories.json')

# mapping catogory ID to name
categories = {int(category['id']): category['snippet']['title']
              for category in raw_categories['items']}

data = {}
labels = {}
count = 0
total = len(raw_data.values)
for i in raw_data.values:
    count += 1
    print("Preprocessing Data: ", int((count/total)*100), "%", end='\r')
    if i[0] not in data.keys():
        title_caps_ratio = sum(1 for c in i[2] if c.isupper()) / len(i[2])
        no_of_tags = sum(1 for c in i[6] if c == '|') + 1
        desc_caps_ratio = sum(1 for c in i[15] if c.isupper()) / len(i[15])
        data[i[0]] = {
            'title_caps_ratio': title_caps_ratio,
            'title_len': len(i[2]),
            'category': i[4],
            'no_of_tags': no_of_tags,
            'no_of_views_first_day': i[7],
            'no_of_ratings(likes+deslikes)_first_day': i[8]+i[9],
            'no_of_comments_first_day': i[10],
            'description_caps_ratio': desc_caps_ratio
        }
        labels[i[0]] = {'days_trending': len(
            raw_data[raw_data.video_id == i[0]].values)}
        if count == 20:
            break #remove later

for i, j in data.items():
    print(j.keys())
    print(j.values(),labels[i].keys(),labels[i].values())
