import csv
import math
import os
import time
from argparse import Namespace, ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List

import toml
from pandas import DataFrame
from termcolor import colored
from tweepy import Client, Tweet, Paginator


columns = ['user', 'text', 'date', 'language']


@dataclass(frozen=True)
class DownloaderConfig:
    tweets_per_user: int
    users: List[str]
    output_csv_path: str

    token: str = '<Input Your Token Here>'
    tweets_per_page: int = 500
    tweets_language: str = 'pl'
    columns: List[str] = field(default_factory=lambda: columns)
    verbose: bool = True

    @property
    def pages_per_user(self) -> int:
        return math.ceil(self.tweets_per_user / self.tweets_per_page)


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('-t', '--tweets', type=int, help='number of tweets per user to download')
    parser.add_argument('-u', '--users', help='path to .txt file with users (one per line)')
    parser.add_argument('-o', '--output', default='tweets.csv', help='path to output .csv file')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()


def compute_page_limit(config: DownloaderConfig, downloaded: int) -> int:
    return min(config.tweets_per_page, config.tweets_per_user - downloaded)


def handle_page(config: DownloaderConfig, tweets: List[Tweet], user: str) -> None:
    initial = not os.path.exists(config.output_csv_path)

    records = [(user, tweet.text, tweet.created_at, tweet.lang) for tweet in tweets]
    records = DataFrame.from_records(records, columns=config.columns)
    records.to_csv(
        config.output_csv_path,
        index=False,
        header=initial,
        mode='a',
        quoting=csv.QUOTE_NONNUMERIC,
        escapechar='\\'
    )


def handle_user_tweets(config: DownloaderConfig, paginator: Paginator, user: str) -> None:
    downloaded = 0

    for index, page in enumerate(paginator, start=1):
        limit = compute_page_limit(config, downloaded)
        tweets = page.data[:limit]

        downloaded += len(tweets)

        if config.verbose:
            print(f'User {user} | Page {index} | Downloaded {len(tweets)} tweets')

        handle_page(config, tweets, user)
        time.sleep(1)

    if downloaded < config.tweets_per_user:
        date = paginator.kwargs['start_time']
        years = datetime.now().year - date.year
        print(f'WARNING! User {user} has made only {downloaded} in last {years} years, '
              f'{config.tweets_per_user} were requested.')


def download(client: Client, config: DownloaderConfig) -> None:
    date = datetime.now() - timedelta(days=10 * 365)
    limit = compute_page_limit(config, downloaded=0)

    for user in config.users:

        paginator = Paginator(
            method=client.search_all_tweets,
            query=f'from:{user} -is:retweet lang:{config.tweets_language}',
            max_results=limit,
            limit=config.pages_per_user,
            tweet_fields=['created_at', 'lang'],
            start_time=date
        )

        handle_user_tweets(config, paginator, user)


def create_downloader_config(arguments: Namespace) -> DownloaderConfig:
    with open(arguments.config, 'r') as file:
        config = toml.loads(file.read())['downloader']

    with open(arguments.users, 'r') as file:
        users = file.read().split()

    return DownloaderConfig(
        tweets_per_user=arguments.tweets,
        output_csv_path=arguments.output,
        verbose=arguments.verbose,
        users=users,
        **config
    )


def main(arguments: Namespace) -> None:
    config = create_downloader_config(arguments)
    client = Client(config.token)

    if config.verbose:
        tweets_count = len(config.users) * config.tweets_per_user
        print(f'Detected {len(config.users)} users ({config.tweets_per_user} tweets per user)')
        print('Starting download of', colored(f'{tweets_count} tweets', 'magenta', attrs=['bold']), '...')

    path = config.output_csv_path

    if os.path.exists(path):
        decision = input('Output path already exists, overwrite it? [Y/N] ')

        if decision.upper() == 'Y':
            os.remove(path)
    else:
        directory, filename = os.path.split(path)
        os.makedirs(directory, exist_ok=True)

    download(client, config)

    if config.verbose:
        print(f'Saving results to {config.output_csv_path} ...')
        print(colored('Successfully finished download', 'green', attrs=['bold']))


if __name__ == '__main__':
    main(parse_arguments())
