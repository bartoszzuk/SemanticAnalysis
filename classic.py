import os.path
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

import utils
from utils import Dataset, ClassifierConfig


@dataclass
class RegressionConfig(ClassifierConfig):
    output_directory_path: str = 'resources/classic'


def create_dataset(data: DataFrame, vectorizer: TfidfVectorizer, fit: bool = False) -> Dataset:
    labels = data['label'].to_numpy()
    tweets = data['text-normalized']

    if fit:
        vectorizer.fit(tweets)

    features = vectorizer.transform(tweets)

    return Dataset(features, labels)


def main() -> None:
    seaborn.set_theme(style='darkgrid')
    config = RegressionConfig()
    vectorizer = TfidfVectorizer(max_df=0.9, max_features=10000)

    train = pd.read_csv(config.train_subset_path, index_col=0)
    valid = pd.read_csv(config.valid_subset_path, index_col=0)

    train = create_dataset(train, vectorizer, fit=True)
    valid = create_dataset(valid, vectorizer, fit=True)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=500, random_state=config.random_seed),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5, metric='cosine'),
        'MultinomialNB': MultinomialNB()
    }

    scores = {}

    for name, model in models.items():
        model.fit(train.features, train.labels)
        scores[name] = model.score(valid.features, valid.labels)

    best = max(scores, key=lambda key: scores[key])
    model = models[best]

    test = pd.read_csv(config.test_subset_path, index_col=0)
    test = create_dataset(test, vectorizer)

    predictions = model.predict(test.features)

    names = ['Negative', 'Positive']

    path = os.path.join(config.output_directory_path, 'roc.png')
    utils.plot_roc_curve(test, model, path)

    path = os.path.join(config.output_directory_path, 'validation.png')
    utils.plot_models_comparison(scores, path)

    path = os.path.join(config.output_directory_path, 'classification-report.txt')
    utils.save_classification_report(test.labels, predictions, names, path)

    path = os.path.join(config.output_directory_path, 'confusion-matrix.png')
    utils.plot_confusion_matrix(test.labels, predictions, names, path)


if __name__ == '__main__':
    main()
