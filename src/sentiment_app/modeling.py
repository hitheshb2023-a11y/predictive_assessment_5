from typing import Dict

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .config import Settings


def make_models(settings: Settings) -> Dict[str, object]:
    return {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "MultinomialNB": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98,
                                        sublinear_tf=True, max_features=settings.max_word_features)),
            ("model", MultinomialNB(alpha=0.5)),
        ]),
        "LogisticRegression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98,
                                        sublinear_tf=True, max_features=settings.max_word_features)),
            ("model", LogisticRegression(max_iter=1200, class_weight="balanced", random_state=settings.seed)),
        ]),
        "LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98,
                                        sublinear_tf=True, max_features=settings.max_word_features)),
            ("model", LinearSVC(class_weight="balanced", random_state=settings.seed)),
        ]),
    }
