# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import datetime as dt


def trainHandMadeANN(feature_set):
    pass


def trainKerasANN(feature_set):
    pass


def testAccuracyHandMadeModel(split_ratio, data):
    pass


def testAccuracyKerasModel(split_ratio, data):
    pass


def predictUsingHandMadeModel(feature_instance, data):
    pass


def predictUsingKerasModel(feature_instance, data):
    pass


# reading dataset
raw_data = pd.read_csv('inputs/trending.csv')
raw_categories = pd.read_json('inputs/categories.json')

# mapping catogory ID to name
categories = {int(category['id']): category['snippet']['title']
              for category in raw_categories['items']}

data = {}
labels = {}
count = 0
total = len(raw_data.values)

print(end="\n\n")
print('-----------------------YTrendNet-----------------------')
print('An Artificial Neural Network which finds how long a')
print('YouTube Video will trend for given first day statistics')
print('-------------------------------------------------------')
print(end="\n\n")

channels = []

for i in raw_data.values:

    count += 1
    print("Preprocessing Data: ", int((count/total)*100), "%", end='\r')

    if i[0] not in data.keys():

        # for no. of channels
        if i[3] not in channels:
            channels.append(i[3])

        title_caps_ratio = sum(1 for c in i[2] if c.isupper()) / len(i[2])
        no_of_tags = sum(1 for c in i[6] if c == '|') + 1

        try:
            desc_caps_ratio = sum(1 for c in i[15] if c.isupper()) / len(i[15])
            no_of_links_in_desc = i[15].count("http")
        except:
            desc_caps_ratio = -1
            no_of_links_in_desc = -1

        first_trend_date = dt.datetime.strptime(i[1], '%y.%d.%m')

        pub_date = dt.datetime.strptime(i[5], '%Y-%m-%dT%H:%M:%S.%fZ')

        delay_to_trend = abs(
            (first_trend_date.date()-pub_date.date()).days)

        data[i[0]] = {
            'title_caps_ratio': title_caps_ratio,
            'title_len': len(i[2]),
            'category': i[4],
            'no_of_tags': no_of_tags,
            'delay_to_trend': delay_to_trend,
            'hour_of_day_published': pub_date.hour,
            'no_of_views_first_day': i[7],
            'no_of_ratings_first_day': i[8]+i[9],
            'likes_to_deslikes_ratio': i[8]/i[9] if i[9] != 0 else i[8],
            'no_of_comments_first_day': i[10],
            'description_caps_ratio': desc_caps_ratio,
            'no_of_links_in_desc': no_of_links_in_desc
        }

        labels[i[0]] = {'days_trending': len(
            raw_data[raw_data.video_id == i[0]].values)}
        if count == 200:
            break  # remove later

print("Preprocessing Data: Completed!", end="\n\n")
print('No. of video categories: ', len(categories))
print("No. of YouTube channels: ", len(channels), end="\n\n")
print("Options:")
print("1. Test accuracy on Artificial Neural Network built by hand (Cross-Entropy, Softmax, Sigmoid, Gradient Descent) with 80/20 split")
print("2. Test accuracy on Artificial Neural Network built using Keras and TensorFlow with 80/20 split")
print("3. Predict how long a video will trend for using Artificial Neural Network built by hand")
print("4. Predict how long a video will trend for using Artificial Neural Network built using Keras and TensorFlow", end="\n\n")
print("Choose an option: ", end="")
ch = input()
print(ch)

# for i, j in data.items():
#     print(j.keys())
#     print(j.values(), labels[i].keys(), labels[i].values())
