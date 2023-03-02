import joblib as jbl
import pandas as pd

x = input("Choose a language:\n1. English\n2. Hindi\n")

feed = input("Enter the feed: ")

if x == "1":
    model = jbl.load('eng_model.pkl')  
    vectorizer = jbl.load('eng_vectorizer.pkl')
    lsa = jbl.load('eng_lsa.pkl')
    labelencoder_name_mapping = jbl.load('eng_labelencoder_name_mapping.pkl')
    # cleaning text
    import re
    import nltk

    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    # clean data
    def clean_data(text):
        content = re.sub('[^a-zA-Z]', ' ', text)
        content = content.lower()
        content = content.split()

        ps = PorterStemmer()
        content = [ps.stem(word) for word in content if not word in set(stopwords.words('english') + ['u', 'r'])]
        content = ' '.join(content)
        return content

    feed = clean_data(feed)
    feed = [feed]
    
    feed = vectorizer.transform(feed).toarray()
    y = pd.DataFrame(data=feed, columns=vectorizer.get_feature_names_out())
    y = lsa.transform(y)

    result = model.predict(y)

    print(list(filter(lambda x: labelencoder_name_mapping[x] == result, labelencoder_name_mapping))[0])
elif x == "2":
    print("Yes")
    model = jbl.load('hindi_model.pkl')  
    vectorizer = jbl.load('hindi_vectorizer.pkl')
    lsa = jbl.load('hindi_lsa.pkl')
    labelencoder_name_mapping = jbl.load('hindi_labelencoder_name_mapping.pkl')
    import emoji
    import re

    def clean_data(text):
        text = emoji.get_emoji_regexp().sub("", text)
        text = text.lower()
        text = re.sub('((www.[^s]+)|(https?://[^s]+))', '', text)
        text = re.sub('((www.[^s]+)|(https?://[^s]+))', '', text)
        text = re.sub('@[^s]+', '', text)
        text = re.sub('[s]+', ' ', text)
        text = re.sub(r'#([^s]+)', r'1', text)
        text = re.sub(r'[-.!:?\'\"\/]', r'', text)
        text = text.strip('\'\"')

        return text
    
    feed = clean_data(feed)
    feed = [feed]
    
    feed = vectorizer.transform(feed).toarray()
    y = pd.DataFrame(data=feed, columns=vectorizer.get_feature_names_out())
    y = lsa.transform(y)

    result = model.predict(y)
    print(result)

    print(list(filter(lambda x: labelencoder_name_mapping[x] == result, labelencoder_name_mapping))[0])
