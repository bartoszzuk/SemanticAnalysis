from dataclasses import dataclass
from typing import List, Union, Dict, Any

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay

seaborn.set_theme(style='darkgrid')


@dataclass
class ClassifierConfig:
    train_subset_path: str = 'resources/tagger/train.csv'
    valid_subset_path: str = 'resources/tagger/valid.csv'
    test_subset_path: str = 'resources/tagger/test.csv'
    random_seed: int = 42


@dataclass
class Dataset:
    features: np.ndarray
    labels: np.ndarray


ArrayLike = Union[List, np.ndarray]


def plot_confusion_matrix(labels: ArrayLike, predictions: ArrayLike, names: ArrayLike, path: str) -> None:
    matrix = confusion_matrix(labels, predictions)
    matrix = pd.DataFrame(matrix, index=names, columns=names)

    plt.figure(figsize=(8, 8))
    heatmap = seaborn.heatmap(matrix, annot=True, fmt="d", cmap="Blues", )
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(path, dpi=300)

    print(f'Successfully saved classification report to {path}')


def save_classification_report(labels: ArrayLike, predictions: ArrayLike, names: ArrayLike, path: str) -> None:
    report = classification_report(labels, predictions, target_names=names)

    with open(path, 'w') as file:
        file.write(report)

    print(f'Successfully saved confusion matrix to {path}')


def plot_losses(metrics: Dict[str, Any], path: str) -> None:
    train = [values[0] for key, values in metrics.items() if key.startswith('Epoch')]
    valid = [values[0] for key, values in metrics.items() if key.startswith('Validation')]

    epochs = np.arange(len(train))

    plt.figure(figsize=(10, 8))
    plt.plot(epochs, train, label='Train')
    plt.plot(epochs, valid, label='Valid')
    plt.legend()
    plt.title('Train and Valid losses')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(path, dpi=300)

    print(f'Successfully saved losses to {path}')


def plot_models_comparison(scores: Dict[str, Any], path: str) -> None:
    names = sorted(scores.keys())
    scores = [scores[name] for name in names]

    plt.figure(figsize=(10, 8))
    seaborn.barplot(x=names, y=scores)
    plt.title(f'Models validation score')
    plt.ylabel('Accuracy')

    bottom = round(min(scores) - 0.1, 1)
    top = round(max(scores) + 0.1, 1)

    plt.ylim(bottom, top)
    plt.savefig(path, dpi=300)

    print(f'Successfully saved validation scores to {path}')


def plot_roc_curve(test: Dataset, model: BaseEstimator, path: str) -> None:
    RocCurveDisplay.from_estimator(model, test.features, test.labels)
    plt.xlabel('FP rate')
    plt.ylabel('TP rate')
    plt.title('ROC curve')
    plt.savefig(path, dpi=300)

    print(f'Successfully saved roc curve to {path}')