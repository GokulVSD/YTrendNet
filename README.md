 <h1 align="center"> <br>
  YTrendNet
  <br> </h1>


<p align="center">
  <a href="https://drive.google.com/file/d/1mzz654f2AXHC5jvPufGkYkrHZkChnpto/view?usp=sharing">
    Project Report
  </a>
  </p>

## Abstract
	
Artificial neural networks (ANN) are computing systems vaguely inspired by the biological neural networks that constitute animal brains. The neural network itself is not an algorithm, but rather a framework for many different machine learning algorithms to work together and process complex data inputs. Such systems "learn" to perform tasks by considering examples, generally without being programmed with any task-specific rules.
YouTube is an American video-sharing website headquartered in San Bruno, California. Three former PayPal employees Chad Hurley, Steve Chen, and Jawed Karim created the service in February 2005. Google bought the site in November 2006 for US$1.65 billion; YouTube now operates as one of Google's subsidiaries.
YouTube allows users to upload, view, rate, share, add to playlists, report, comment on videos, and subscribe to other 
users. It offers a wide variety of user-generated and corporate media videos. 

Available content includes video clips, TV show clips, music videos, short and documentary films, audio recordings, movie trailers, live streams, and other content such as video blogging, short original videos, and educational videos. Most of the content on YouTube is uploaded by individuals, but media corporations including CBS, the BBC, Vevo, and Hulu offer some of their material via YouTube as part of the YouTube partnership program.
Trending helps viewers see what’s happening on YouTube and in the world. Some trends are predictable, like a new song from a popular artist or a new movie trailer. Others are surprising, like a viral video. Trending aims to surface videos that a wide range of viewers will appreciate.
Trending is not personalized. Trending displays the same list of trending videos in each country to all users, except for India. In India, Trending displays the same list of trending videos for each of the 9 most common Indic languages.

The list of trending videos is updated roughly every 15 minutes. With each update, videos may move up, down, or stay in the same position in the list.
Amongst the many new videos on YouTube on any given day, Trending can only show a limited number. Trending aims to surface videos that are appealing to a wide range of viewers, are not misleading or sensational and capture the breadth of what’s happening on YouTube and in the world.
Trending aims to balance all of these considerations. To achieve this, Trending considers many signals, including (but not limited to):

* View count
* The rate of growth in views
* Where views are coming from (including outside of YouTube)
* The age of the video

These claims are made by YouTube itself, with respect to the factors pertaining to which videos end up on the Trending page. No claims pertaining to how long a video will stay on the Trending page exist, neither have the factors involved been disclosed.
	The aim of YTrendNet is to find relevant features in the feature set representing meta-data of videos over a period of time, such that these features have strong correlations to how long a particular video stays on the Trending page.

## Data Set

This dataset includes several months (and counting) of data on daily trending YouTube videos. Data is included for the US, GB, DE, CA, and FR regions (USA, Great Britain, Germany, Canada, and France, respectively), with up to 200 listed trending videos per day.
The dataset also includes data from RU, MX, KR, JP and IN regions (Russia, Mexico, South Korea, Japan and India respectively) over the same time period.
Each region’s data is in a separate file. Data includes the video title, channel title, publish time, tags, views, likes and dislikes, description, and comment count.
The data also includes a category_id field, which varies between regions. To retrieve the categories for a specific video, find it in the associated JSON. One such file is included for each of the five regions in the dataset.
The dataset contains the following features:

* video_id
* trending_date
* title
* channel_title
* category_id
* publish_time
* tags
* views
* likes
* deslikes
* trending_date
* comment_count
* thumbnail_link
* comments_disabled
* ratings_disabled
* video_error_or_removed
* description

## Preprocessing

The dataset was preprocessed into the following features:

* title_caps_ratio
* title_len
* category
* channel_title
* no_of_tags
* delay_to_trend
* hour_of_day_published
* no_of_views_first_day
* no_of_ratings_first_day
* likes_to_total_ratings_ratio
* no_of_comments_first_day
* description_caps_ratio
* no_of_links_in_desc

 The preprocessing procedure involved merging videos with the same video_id and counting the number of merged instances, so as to derive the target variable that we wish to model, ‘no_of_days_trending’
The dataset used contained over 50,000 records, with a record for each day that a video trended on. Once the feature set was preprocessed, normalised features and the target variable was calculated on the entire dataset, the number of records reduced to over 7000. The dataset can be split based on user input, and the desired number of epochs to train for can also be specified by the user.

## Implementation

YTrendNet is a dual Implementation of Artificial Neural Networks using activation functions Sigmoid and Softmax, loss metric Cross-Entropy, and with Gradient Descent/Adam as the optimisation procedure to adjust the weights of the network. The purpose of the ANN is to find (classify) how long a YouTube Video will stay trending given the statistics of the video on the first day it trended.
	The network consists of 4 layers with over 2200 nodes in input layer, 3000 nodes in each hidden layer and 45 nodes in the output layer. Most of the input nodes are due to one-hot encoding of the feature set. The 45 output nodes correspond to up to 45 days that a video can trend for.
	Two ANNs have been implemented. The first one is built manually using NumPy, which is a library that gives support for mathematical functions on multi-dimensional vectors. The second one is built using Keras, using TensorFlow backend. Keras is a high level machine learning library. TensorFlow is a library that supports efficient dataflow and differential programming in parallel. The features from the dataset was extracted with the help of Pandas, a Python library that helps with the manipulation and interpretation of datasets. To split the dataset into training and testing sets, a Python library named sklearn was used.
	During the implementation and testing phase, we noticed low levels of accuracy, and hence we made provisions for per execution manipulation of training epochs and split ratios, along with options to choose between the two models.
	Sigmoid was chosen as the activation function for all layers except the last layer.

In the Keras model, an option to use ReLU was provided so as to allow for overfitting.

The last layer uses Softmax for activation.

<br />

## Contributors
<p><strong>Gokul Vasudeva</strong>   <a href="https://github.com/gokulvsd">github.com/gokulvsd</a></p>
<p><strong>Anusha A</strong>   <a href="https://github.com/anushab05">github.com/anushab05</a></p>

<br />

