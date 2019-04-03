# importing libraries
import numpy as np
import pandas as pd
import math
import datetime as dt
from sklearn.model_selection import train_test_split
import numpy_ann


def trainHandMadeANN(feature_set, labels):
    pass


def trainKerasANN(feature_set, labels):
    pass


def testAccuracyHandMadeModel(split_ratio, data, labels):
    pass


def testAccuracyKerasModel(split_ratio, data, labels):
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
print('------------------YTrendNet------------------')
print('An Artificial Neural Network which finds how')
print('long a YouTube Video will trend for given the')
print('statistics of the video on the first day')
print('---------------------------------------------')
print(end="\n\n")

max_title_len = 1
max_no_of_tags = 1
max_delay_to_trend = 1
max_no_of_views_first_day = 1
max_no_of_ratings_first_day = 1
max_no_of_comments_first_day = 1
max_no_of_links_in_desc = 1

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

        delay_to_trend = abs((first_trend_date.date()-pub_date.date()).days)

        # finding global maximas for normalisation

        if len(i[2]) > max_title_len:
            max_title_len = len(i[2])

        if no_of_tags > max_no_of_tags:
            max_no_of_tags = no_of_tags

        if delay_to_trend > max_delay_to_trend:
            max_delay_to_trend = delay_to_trend

        if i[7] > max_no_of_views_first_day:
            max_no_of_views_first_day = i[7]

        if i[8] + i[9] > max_no_of_ratings_first_day:
            max_no_of_ratings_first_day = i[8] + i[9]

        if i[10] > max_no_of_comments_first_day:
            max_no_of_comments_first_day = i[10]

        if no_of_links_in_desc > max_no_of_links_in_desc:
            max_no_of_links_in_desc = no_of_links_in_desc

        data[i[0]] = {
            'title_caps_ratio': title_caps_ratio,
            'title_len': len(i[2]),
            'category': i[4],
            'channel_title': i[3],
            'no_of_tags': no_of_tags,
            'delay_to_trend': delay_to_trend,
            'hour_of_day_published': pub_date.hour,
            'no_of_views_first_day': i[7],
            'no_of_ratings_first_day': i[8]+i[9],
            'likes_to_total_ratings_ratio': i[8]/(i[8] + i[9]) if i[8]+i[9] != 0 else -1,
            'no_of_comments_first_day': i[10],
            'description_caps_ratio': desc_caps_ratio,
            'no_of_links_in_desc': no_of_links_in_desc
        }

        labels[i[0]] = {'days_trending': len(
            raw_data[raw_data.video_id == i[0]].values)}
        # if count == 200:
        #     break  # remove later

print("Preprocessing Data: Completed!", end="\n\n")

arr = []
labs = []

# converting data and labels into python lists
for i, j in data.items():
    arr.append(list(j.values()))
    labs.append((labels[i])['days_trending'])


# one-hot encoding categories and channel titles and converting to numpy arrays

cat_enc = np.zeros((len(arr), max(list(categories.keys()))))

chan_enc = np.zeros((len(arr), len(channels)))

labels_encoded = np.zeros((len(arr), max(labs) + 1))

hour_enc = np.zeros((len(arr), 24))

for i in range(len(arr)):

    print("One-hot Encoding Data: ", int(((i+1)/len(arr))*100), "%", end='\r')

    cat_enc[i, arr[i][2]-1] = 1

    labels_encoded[i, labs[i]] = 1

    hour_enc[i, arr[i][6]] = 1

    try:
        chan_enc[i, channels.index(arr[i][3])] = 1
    except:
        pass

print("One-Hot Encoding Data: Completed!", end="\n\n")

data_enc = []

for i in range(len(arr)):

    print("Normalising and Converting to Numpy array: ", int(
        ((i+1)/len(arr))*100), "%", end='\r')

    data_enc.append([arr[i][0]] + [arr[i][1]/max_title_len] + [arr[i][4]/max_no_of_tags] + [arr[i][5]/max_delay_to_trend] +
                    [arr[i][7]/max_no_of_views_first_day] + [arr[i][8]/max_no_of_ratings_first_day] + [arr[i][9]] +
                    [arr[i][10]/max_no_of_comments_first_day] + [arr[i][11]] + [arr[i][12]/max_no_of_links_in_desc] +
                    cat_enc[i].tolist() + chan_enc[i].tolist() + hour_enc[i].tolist())

data_encoded = np.array(data_enc)

print("Normalising and Converting to Numpy array: Completed!", end="\n\n")
print('No. of video categories: ', len(categories))
print('Highest Category ID: ', max(list(categories.keys())))
print('Max no. of days a video was trending: ', max(labs))
print('No. of instances:', data_encoded.shape[0])
print("No. of nodes at the input layer: ", data_encoded.shape[1])
print("No. of nodes at the output layer: ", max(labs) + 1)
print('No. of YouTube channels: ', len(channels), end="\n\n")
print("Options:")
print("1. Test accuracy on Artificial Neural Network built with pure Numpy (Cross-Entropy, Softmax, Sigmoid, Gradient Descent) with 80/20 split")
print("2. Test accuracy on Artificial Neural Network built using Keras and TensorFlow with 80/20 split", end="\n\n")
print("Choose an option: ", end="")

print()

# random_state promises deterministic split
train_data, test_data, train_labels, test_labels = train_test_split(data_encoded, labels_encoded, test_size=0.2, random_state=20)

model = numpy_ann.ANN(train_data, train_labels)

epochs = 20
for i in range(epochs):
    print("epoch: ", i, " of ", epochs, end="\t")
    model.feedforward()
    model.backpropogate()

def get_acc(x, y):
    acc = 0
    for xx,yy in zip(x, y):
        s = model.predict(xx)
        #print(s," ",np.argmax(yy))
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100
	
print("Training accuracy : ", get_acc(train_data, train_labels), " %")
print("Test accuracy : ", get_acc(test_data, test_labels), " %")

# ch = input()
# if ch == 1:
#     testAccuracyHandMadeModel(0.8, data_encoded, labels_encoded)
# elif ch == 2:
#     testAccuracyKerasModel(0.8, data_encoded, labels_encoded)
# else:
#     print("Invalid Option")
