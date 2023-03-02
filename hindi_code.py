import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
from collections import Counter
from englisttohindi.englisttohindi import EngtoHindi

# Education

# Most frequent hashtags used for education posts 
education_tags = ["#education", "#school", "#edtech", "#edchat",
                  "#learning", "#exam", "#outdoorlearning", "#K12", 
                  "#topper", "#edleadership", "#marks",  
                  "#literacy", "#scichat", "#mathchat", "#edreform", 
                  "#coding", "#highschool", "#teaching", "#study"]

# Crime

# Most frequent hashtags used for crime posts 
crime_tags = ["#crime", "#murder", "#kidnap", "#rap",
              "#violence", "#childabuse", "#fraud", "#robbery",
              "#sexualharasment", "#terrorist", "#stalking", "#corruption", 
              "#lawbreaking", "#lawlessness", "#gangsterism", "#racketeering",
              "#wrongdoing", "#outlawry", "#phishing", "#arrested"]

# Transport

# Most frequent hashtags used for transport posts 
transport_tags = ["#transport", "#transportation", "#travel", "#supplychain",
                  "#cargo", "#shipping", "#train", "#i mport", 
                  "#export", "#truck", "#trucker", "#auto"
                  "#logistics", "#scania", "#trucking", "#delivery"]

# Health

# Most frequent hashtags used for health posts 
health_tags = ["#health", "#fitness", "#fit", "#healthy",
               "#psychology", "#mentalhealth", "#pharma", "#healthissues",
               "#wellness", "#healthtech", "#biotech", "#exercise",
               "#mhealth", "#workout", "#mediation", "#godhealth"]

# Hygiene

# Most frequent hashtags used for hygiene posts 
hygiene_tag = ["#hygiene", "#food", "#sanitation", "#dentalhygiene",
               "#cleanliness", "#water", "#clean", "#diet",
               "#handwash", "#selfcare", "#staysafe", "#skincare"]

all_tags = {
    'education': education_tags,
    'crime': crime_tags,
    'transport': transport_tags,
    'health': health_tags,
    'hygiene': hygiene_tag
}

for tag in all_tags:
    arr = []
    for sub_tag in all_tags[tag]:
        arr.append(EngtoHindi(sub_tag).convert)
    all_tags[tag] = arr

# Web scraping
# importing libraries
from tqdm.notebook import tqdm
import snscrape.modules.twitter as sntwitter
from itertools import permutations
import warnings

warnings.filterwarnings('ignore')

# cleaning text
import re
import nltk
from langdetect import detect
import emoji

# clean data
def clean_data(text):
    text = emoji.get_emoji_regexp().sub("", text)
    text = text.lower()
    text = re.sub('((www.[^s]+)|(https?://[^s]+))','',text)
    text = re.sub('((www.[^s]+)|(https?://[^s]+))','',text)
    text = re.sub('@[^s]+','',text)
    text = re.sub('[s]+', ' ', text)
    text = re.sub(r'#([^s]+)', r'1', text)
    text = re.sub(r'[-.!:?\'\"\/]', r'', text)
    text = text.strip('\'\"')
    
    return text

posts = []
titles = []

# Twitter posts
def twitter_posts(hashtags):
    tweets = {}
    no_tweets = 1000
    for tag in hashtags:
        if detect(tag) == "hi":
            scraper = sntwitter.TwitterSearchScraper("#"+tag)
            tweet = []
            for i, t in tqdm(enumerate(scraper.get_items()), total = no_tweets, desc = tag):
                data = t.content
                tweet.append(clean_data(data))
                if i > no_tweets:
                    break
            tweets[tag] = list(set(tweet))
    return tweets   

def add_to_dataframe(title, post):
    for val in post.values():
        for ele in val:
            posts.append(ele)
            titles.append(title)
            
def get_total_no_posts(title, post):
    total_values = 0
    for l in post.values():
        total_values += len(l)
    print(title, " total posts: ", total_values)

data = twitter_posts(all_tags['education'])
data.keys()
get_total_no_posts('education', data)
add_to_dataframe('education', data)

data = twitter_posts(all_tags['crime'])
data.keys()
get_total_no_posts('crime', data) 
add_to_dataframe('crime', data)

data = twitter_posts(all_tags['transport'])
data.keys()
get_total_no_posts('transport', data)
add_to_dataframe('transport', data)

data = twitter_posts(all_tags['health'])
data.keys()
get_total_no_posts('health', data)
add_to_dataframe('health', data)

data = twitter_posts(all_tags['hygiene'])
data.keys()
get_total_no_posts('hygiene', data)
add_to_dataframe('hygiene', data)

df = pd.DataFrame(list(zip(posts, titles)), columns = ['post', 'type'])

df.head()

df.shape

df['type'].unique()

# WORD-COUNT
df['word_count'] = df['post'].apply(lambda x: len(str(x).split()))
print(df[df['type']=='education']['word_count'].mean()) #Education tweets

df['word_count'] = df['post'].apply(lambda x: len(str(x).split()))
print(df[df['type']=='crime']['word_count'].mean()) #Crime tweets

df['word_count'] = df['post'].apply(lambda x: len(str(x).split()))
print(df[df['type']=='transport']['word_count'].mean()) #Transport tweets

df['word_count'] = df['post'].apply(lambda x: len(str(x).split()))
print(df[df['type']=='health']['word_count'].mean()) #Health tweets

df['word_count'] = df['post'].apply(lambda x: len(str(x).split()))
print(df[df['type']=='hygiene']['word_count'].mean()) #Hygiene tweets

mean_word_type = []
for tag in df['type'].unique():
    mean_word_type.append(df[df['type']==tag]['word_count'].mean())
    
# creating the bar plot
plt.bar(df['type'].unique(), mean_word_type, color ='maroon', width = 0.4)
plt.xlabel("Tags")
plt.ylabel("Mean words")
plt.title("Mean of number of words in tags")
plt.show()

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder

labelencoder_y = LabelEncoder()
df['type'] = labelencoder_y.fit_transform(df['type'])
labelencoder_name_mapping = dict(zip(labelencoder_y.classes_, labelencoder_y.transform(labelencoder_y.classes_)))
print(labelencoder_name_mapping)

df.head()

df = df.drop(['word_count'], axis=1)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Vectorisation
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7)
X = vectorizer.fit_transform(posts).toarray()
#df = pd.DataFrame(data=X, columns=vectorizer.get_feature_names_out())

# LSA (Latent semantic analysis)
from time import time
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

lsa = make_pipeline(TruncatedSVD(n_components=100), Normalizer(copy=False))
t0 = time()
X = lsa.fit_transform(X)
explained_variance = lsa[0].explained_variance_ratio_.sum()

print(f"LSA done in {time() - t0:.3f} s")
print(f"Explained variance of the SVD step: {explained_variance * 100:.1f}%")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Naive Bayes Classifier

# Fitting Naive bayes classifier to the Training set
from sklearn.naive_bayes import GaussianNB
nbc = GaussianNB()
nbc.fit(X_train, y_train)

y_pred = nbc.predict(X_test)

from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, accuracy_score

print("Naive Bayes Classifier: ", r2_score(y_test, y_pred))

cm_nb = confusion_matrix(y_test, y_pred)
print(cm_nb)

acc_nb = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc_nb)

# Support Vector Classifier

# Fitting classifier to the Training set
from sklearn.svm import SVC
svc = SVC(kernel='rbf', random_state=0)
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, accuracy_score

print("Support Vector Classifier: ", r2_score(y_test, y_pred))

cm_svc = confusion_matrix(y_test, y_pred)
print(cm_svc)

acc_svc = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc_svc)

# Logistic Classifier

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression

lc = LogisticRegression(random_state=0)
lc.fit(X_train, y_train)

y_pred = lc.predict(X_test)

from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, accuracy_score

print("Logistic Classifier: ", r2_score(y_test, y_pred))

cm_lc = confusion_matrix(y_test, y_pred)
print(cm_lc)

acc_lc = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc_lc)

# XGB Classifier

from xgboost import XGBClassifier

# Training the XGB Classifier model on the Training set
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, accuracy_score

print("XGB Classifier: ", r2_score(y_test, y_pred))

cm_xgb = confusion_matrix(y_test, y_pred)
print(cm_xgb)

acc_xgb = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc_xgb)

# KNN Classifier

from sklearn.neighbors import KNeighborsClassifier as KNN

# Training the KNN Classifier model on the Training set
knn = KNN(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix, accuracy_score

print("KNN Classifier: ", r2_score(y_test, y_pred))

cm_knn = confusion_matrix(y_test, y_pred)
print(cm_knn)

acc_knn = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % acc_knn)

max_acc = max(acc_knn, acc_lc, acc_nb, acc_svc, acc_xgb)

model_eng = knn
if max_acc == acc_lc:
    print("\n\nBest model is Logistic Regression\n\n")
    model_eng = lc
elif max_acc == acc_nb:
    print("\n\nBest model is Naive Bayes Classifier\n\n")
    model_eng = nbc
elif max_acc == acc_svc:
    print("\n\nBest model is Support Vector Classifier\n\n")
    model_eng = svc
elif max_acc == acc_xgb:
    print("\n\nBest model is XGB Classifier\n\n")
    model_eng = xgb
else:
    print("\n\nBest model is KNeighbors Classifier\n\n")


import joblib as jbl

jbl.dump(model_eng, 'hindi_model.pkl')
jbl.dump(vectorizer, 'hindi_vectorizer.pkl')
jbl.dump(lsa, 'hindi_lsa.pkl')
jbl.dump(labelencoder_name_mapping, 'hindi_labelencoder_name_mapping.pkl')