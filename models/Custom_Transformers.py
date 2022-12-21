from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class MessageTransformer(BaseEstimator, TransformerMixin):
    """ Applies CountVectorizer and TfidfTransformer to strings

    """

    def __init__(self, tokenize):
        self.tokenize = tokenize
        self.message_transformer = Pipeline([
            ('vect', CountVectorizer(tokenizer=self.tokenize)),
            ('tfidf', TfidfTransformer())
        ])

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        return self.message_transformer.fit_transform(X.squeeze()).toarray()
