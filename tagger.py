import csv
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


@dataclass
class TaggerConfig:
    tweets_preprocessed_path: str = 'resources/preprocessor/tweets-preprocessed.csv'
    sentiment_embeddings_path: str = 'resources/preprocessor/embeddings-sentiment.npy'
    spacy_embeddings_path: str = 'resources/preprocessor/embeddings-spacy.npy'
    train_subset_path: str = 'resources/tagger/train.csv'
    valid_subset_path: str = 'resources/tagger/valid.csv'
    test_subset_path: str = 'resources/tagger/test.csv'
    min_tweets_length: int = 20


def plot_clusters(points: np.ndarray, labels: np.ndarray) -> None:
    x = points[:, 0]
    y = points[:, 1]

    mapper = {0: 'blue', 1: 'red', 2: 'green'}
    colors = [mapper.get(label, 'gray') for label in labels]

    plt.scatter(x, y, c=colors)
    plt.show()


def main() -> None:
    config = TaggerConfig()

    sentiment_embeddings = np.load(config.sentiment_embeddings_path)
    spacy_embeddings = np.load(config.spacy_embeddings_path)

    selector = ~np.all(sentiment_embeddings == 0, axis=1)

    sentiment_embeddings = sentiment_embeddings[selector]
    spacy_embeddings = spacy_embeddings[selector]

    decomposer = PCA(n_components=20)
    decomposed = decomposer.fit_transform(spacy_embeddings)

    embeddings = np.hstack([decomposed, sentiment_embeddings])

    model = KMeans(n_clusters=2, random_state=42)
    model.fit(embeddings)

    tweets = pd.read_csv(config.tweets_preprocessed_path)
    tweets = tweets[selector]

    tweets['label'] = model.labels_

    lengths = tweets['text-normalized'].apply(str) \
        .apply(str.split) \
        .apply(len)

    tweets = tweets[lengths > config.min_tweets_length]

    train, test = train_test_split(tweets, test_size=0.2, stratify=tweets['label'])
    train, valid = train_test_split(train, test_size=0.2, stratify=train['label'])

    train.to_csv(config.train_subset_path, index=True, quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')
    valid.to_csv(config.valid_subset_path, index=True, quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')
    test.to_csv(config.test_subset_path, index=True, quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')


if __name__ == '__main__':
    main()
