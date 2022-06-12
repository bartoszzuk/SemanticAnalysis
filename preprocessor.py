import csv
import functools
import os.path
from dataclasses import dataclass
from typing import List, Callable

import numpy as np
import pandas as pd
import spacy as spacy
from pandas import DataFrame
from spacy.tokens import DocBin, Doc, Token


@dataclass
class PreprocessorConfig:
    tweets_unprocessed_path: str = 'resources/downloader/tweets.csv'
    tweets_preprocessed_path: str = 'resources/preprocessor/tweets-preprocessed.csv'
    spacy_documents_path: str = 'resources/preprocessor/documents.spacy'
    tweets_embeddings_path: str = 'resources/preprocessor/embeddings.numpy'
    encoder_type: str = 'spacy'


nlp = spacy.load('pl_core_news_md', disable=['ner', 'tagger', 'parser'])


def serialize_spacy_documents(texts: List[str], path: str, logging_frequency: int = 1000) -> None:
    documents = DocBin()

    for index, document in enumerate(nlp.pipe(texts), start=1):
        documents.add(document)

        if index % logging_frequency == 0:
            print(f'Processed {index}/{len(texts)} documents ...')

    documents.to_disk(path)


def deserialize_spacy_documents(path: str) -> List[Doc]:
    documents = DocBin().from_disk(path)
    return list(documents.get_docs(nlp.vocab))


def validator(token: Token) -> bool:
    return not token.is_stop \
           and not token.like_num \
           and not token.is_punct \
           and not token.like_email \
           and not token.like_url \
           and not token.text.startswith('@') \
           and not token.text.startswith('#')


def preprocess(document: Doc) -> str:
    return ' '.join(token.lemma_ for token in document if validator(token))


def serialize_preprocessed_tweets(tweets: DataFrame, documents: List[Doc], path: str) -> None:
    texts = [preprocess(document) for document in documents]
    tweets['text-normalized'] = texts
    tweets.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')


def spacy_encoder(text: str) -> np.ndarray:
    return nlp(text).vector


def sentiment_encoder(text: str, sentiments: DataFrame) -> np.ndarray:
    vocabulary = sentiments.index
    tokens = set(text.split())

    found = tokens.intersection(vocabulary)
    embedding = np.array([sentiments.loc[token] for token in found]) if found else np.zeros(shape=(1, 4))

    return embedding.mean(axis=0)


def serialize_tweet_embeddings(texts: List[str], encoder: Callable[[str], np.ndarray], path: str) -> None:

    embeddings = []

    for index, text in enumerate(texts, start=1):
        embeddings.append(encoder(text))

        if index % 1000 == 0:
            print(f'Processed {index}/{len(texts)} embeddings ...')

    embeddings = np.stack(embeddings)
    np.save(path, embeddings)


def load_sentiment_dictionary():
    path = 'resources/preprocessor/sentiment-dictionary.csv'
    columns = ['word', 'unknown', 'sentiment 1', 'sentiment 2', 'sentiment 3', 'PMI score']
    sentiments = pd.read_csv(path, sep='\t', header=None, names=columns)
    sentiments = sentiments.set_index('word')
    sentiments = sentiments.dropout('unknown', axis=1)
    return sentiments


def main():
    config = PreprocessorConfig()
    tweets = pd.read_csv(config.tweets_unprocessed_path)

    needs_tokenization = not os.path.exists(config.spacy_documents_path)
    needs_preprocessing = not os.path.exists(config.tweets_preprocessed_path)
    needs_vectorization = not os.path.exists(config.tweets_embeddings_path)

    if needs_tokenization:
        texts = tweets['text'].tolist()
        serialize_spacy_documents(texts, config.spacy_documents_path)
        print(f'Successfully serialized documents ...')

    if needs_preprocessing:
        documents = deserialize_spacy_documents(config.spacy_documents_path)
        serialize_preprocessed_tweets(tweets, documents, config.tweets_preprocessed_path)
        print(f'Successfully serialized preprocessed tweets ...')

    if needs_vectorization:
        tweets = pd.read_csv(config.tweets_preprocessed_path)
        texts = tweets['text'].tolist()

        directory, filename = os.path.split(config.tweets_embeddings_path)
        filename, extension = os.path.splitext(filename)

        if config.encoder_type in ['spacy',  'both']:
            path = os.path.join(directory, f'{filename}-spacy')
            serialize_tweet_embeddings(texts, encoder=spacy_encoder, path=path)
            print(f'Successfully serialized spacy embeddings ...')

        if config.encoder_type in ['sentiment',  'both']:
            path = os.path.join(directory, f'{filename}-sentiment')
            encoder = functools.partial(sentiment_encoder, sentiments=load_sentiment_dictionary())
            serialize_tweet_embeddings(texts, encoder=encoder, path=path)
            print(f'Successfully serialized sentiment embeddings ...')


if __name__ == '__main__':
    main()
