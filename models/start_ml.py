# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from Custom_Transformers import MessageTransformer


# load data from database
engine = create_engine('sqlite:///data/Vasilis_db.db')
df = pd.read_sql_table('Emergency_Messages', engine)
df = df.drop(columns = ['id','original'], axis=1)


# drop rows with NaN messages
df = df.dropna(subset=['message'])

# our dataset still contains some null values
#print(df.isnull().sum())

# so drop them
df = df.dropna()

# We can observe that the 'related' category also contains double's(2) that does not make sense - we will turn this 2s to 1s
#df.related.value_counts()

df.loc[df.related==2, 'related'] = 1


X = df[['message','genre']]
Y = df.drop(columns=['message','genre'], axis=1)


def tokenize(text):
    """ Function that tokenizes and lemmatizes text

    :param text:     input text to be processed(str)
    :return:         cleaned_tokens(str)
    """

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # Text normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    #tokenization
    words = word_tokenize (text)
    # lemmatization
    cleaned_tokens = [lemmatizer.lemmatize(word).strip() for word in words if word not in stop_words]

    return cleaned_tokens



text_transformer = Pipeline([
    ('messagetransformer', MessageTransformer(tokenize)),
])
genre_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('text', text_transformer, [0]),
    ('genre', genre_transformer, [1])
])
pipeline_2 = Pipeline([
    ('preprocess', preprocessor),
    ('clf', RandomForestClassifier(random_state=33))
])

#####################################################################################

pipeline = Pipeline([
     ('vect', CountVectorizer(tokenizer=tokenize)),
     ('tfidf', TfidfTransformer()),
     ('clf', RandomForestClassifier(random_state=33))
])




X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=33)

pipeline_2.fit(X_train, y_train)
print('DONE TRAINING')
preds = pipeline_2.predict(X_test)
print(preds)

