import functools
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Callable, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss, Module
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchtext.data import get_tokenizer, to_map_style_dataset
from torchtext.vocab import Vocab, build_vocab_from_iterator

import utils
from utils import ClassifierConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Tokenizer = Callable[[str], List[int]]
Preprocessor = Callable[[str], Tensor]


@dataclass
class RecurrentConfig(ClassifierConfig):
    output_directory_path: str = 'resources/recurrent/'


def build_vocabulary(train: DataFrame, tokenizer: Tokenizer) -> Vocab:
    words = train['text-normalized'].apply(tokenizer)

    iterator = iter(words.values)

    vocabulary = build_vocab_from_iterator(iterator, specials=['<unk>'])
    vocabulary.set_default_index(index=vocabulary['<unk>'])

    return vocabulary


def build_preprocessor(tokenizer: Tokenizer, vocabulary: Vocab) -> Preprocessor:
    def preprocess(text: str) -> Tensor:
        tokens = vocabulary(tokenizer(text))
        return torch.tensor(tokens, dtype=torch.int64)

    return preprocess


def collate_batch(batch, preprocessor: Preprocessor) -> Tuple[Tensor, ...]:
    labels, tweets, lengths = [], [], []

    for label, tweet in batch:
        tokens = preprocessor(tweet)

        labels.append(label)
        tweets.append(tokens)
        lengths.append(len(tokens))

    labels = torch.tensor(labels, dtype=torch.int64, device=device)
    tweets = pad_sequence(tweets, batch_first=True).to(device)
    lengths = torch.tensor(lengths, dtype=torch.int64, device=device)

    return labels, tweets, lengths


def create_dataset_iterator(data: DataFrame) -> Dataset[Tuple]:
    labels = data['label'].to_numpy()
    tweets = data['text-normalized']

    for label, tweet in zip(labels, tweets):
        yield label, tweet.lower()


class SentimentModel(nn.Module):

    def __init__(self, vocabulary_size: int, embedding_dim: int, classes: int, hidden_dim: int = 128) -> None:
        super(SentimentModel, self).__init__()

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)

        # Just switch to LSTM
        self.encoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.header = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, classes)
        )

    def forward(self, tweets: Tensor, lengths: Tensor) -> Tensor:
        embedded = self.embedding(tweets)

        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self.encoder(packed)
        logits, _ = pad_packed_sequence(packed, batch_first=True)

        indices = torch.arange(len(logits))
        dimension = self.encoder.hidden_size

        forward_logits = logits[indices, lengths - 1, :dimension]
        reverse_logits = logits[:, 0, dimension:]

        logits = torch.cat([forward_logits, reverse_logits], dim=1)

        return self.header(logits)


@dataclass
class Trainer:
    model: Module
    criterion: CrossEntropyLoss
    optimizer: Optimizer
    scheduler: StepLR
    epochs: int = 100
    epochs_per_evaluation: int = 1
    metrics: Dict[str, Tuple] = field(default_factory=dict)
    best_validation_score: float = 0.0
    best_model_path: str = 'resources/recurrent/model'

    def train(self, train: DataLoader, valid: DataLoader) -> None:
        for epoch in range(1, self.epochs + 1):

            self.model.train()
            metrics = defaultdict(list)

            for labels, tweets, lengths in train:
                self.optimizer.zero_grad()
                logits = self.model(tweets, lengths)

                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()

                predictions = logits.argmax(1)

                metrics['losses'].append(loss.item())
                metrics['predictions'].extend(predictions.tolist())
                metrics['labels'].extend(labels.tolist())

            loss = np.mean(metrics['losses'])
            score = accuracy_score(metrics['labels'], metrics['predictions'])

            message = f'Epoch {epoch} | Loss {loss:.4f} | Accuracy: {score:.2f}'

            self.metrics[f'Epoch {epoch}'] = (loss, score)

            if epoch % self.epochs_per_evaluation == 0:
                loss, score = self.evaluate(valid)

                if score > self.best_validation_score:
                    self.best_validation_score = score
                    torch.save(self.model, self.best_model_path)

                message += f' | Validation Loss {loss:.4f} | Validation Accuracy: {score:.2f}'
                self.metrics[f'Validation {epoch // self.epochs_per_evaluation}'] = (loss, score)

            print(message)

    def evaluate(self, loader: DataLoader, return_metrics: bool = False) -> Tuple:
        self.model.eval()
        metrics = defaultdict(list)

        for labels, tweets, lengths in loader:
            with torch.no_grad():
                logits = self.model(tweets, lengths)
                loss = self.criterion(logits, labels)
                predictions = logits.argmax(1)

                metrics['losses'].append(loss.item())
                metrics['predictions'].extend(predictions.tolist())
                metrics['labels'].extend(labels.tolist())

        loss = np.mean(metrics['losses']).item()
        score = accuracy_score(metrics['labels'], metrics['predictions'])

        return (loss, score, metrics) if return_metrics else (loss, score)


def main():
    config = RecurrentConfig()
    torch.manual_seed(config.random_seed)

    train = pd.read_csv(config.train_subset_path, index_col=0)
    valid = pd.read_csv(config.valid_subset_path, index_col=0)

    tokenizer = get_tokenizer('spacy', language='pl_core_news_md')
    vocabulary = build_vocabulary(train, tokenizer)
    preprocessor = build_preprocessor(tokenizer, vocabulary)
    collate = functools.partial(collate_batch, preprocessor=preprocessor)

    train = to_map_style_dataset(create_dataset_iterator(train))
    valid = to_map_style_dataset(create_dataset_iterator(valid))

    train_loader = DataLoader(train, batch_size=128, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid, batch_size=128, shuffle=True, collate_fn=collate)

    model = SentimentModel(
        vocabulary_size=len(vocabulary),
        embedding_dim=128,
        hidden_dim=128,
        classes=2
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    trainer = Trainer(model, criterion, optimizer, scheduler, epochs=10)
    trainer.train(train_loader, valid_loader)

    names = ['Negative', 'Positive']

    trainer.model = torch.load(trainer.best_model_path).to(device)

    test = pd.read_csv(config.test_subset_path, index_col=0)

    test = to_map_style_dataset(create_dataset_iterator(test))
    test_loader = DataLoader(test, batch_size=128, shuffle=True, collate_fn=collate)

    loss, score, metrics = trainer.evaluate(test_loader, return_metrics=True)
    labels, predictions = metrics['labels'], metrics['predictions']

    path = os.path.join(config.output_directory_path, 'classification-report.txt')
    utils.save_classification_report(labels, predictions, names, path)

    path = os.path.join(config.output_directory_path, 'confusion-matrix.png')
    utils.plot_confusion_matrix(labels, predictions, names, path)

    path = os.path.join(config.output_directory_path, 'validation.png')
    utils.plot_losses(trainer.metrics, path)


if __name__ == '__main__':
    main()
