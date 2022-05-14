# Semantic analysis
Simple university project showcasing **sentiment analysis** on collected **tweets**.


## Downloader

Use `downloader.py` script to download tweets to .csv file. Pass `--tweets` argument to specify the number of tweets per user.
The tweets are restricted to 10 years, if user made fewer tweets in that period than tweets are limited to that number.

Below is an exact command used to generate `resources\downloader\tweets.csv` file:

```shell
python downloader.py \
--tweets 10000 \
--users resources/downloader/users.txt \
--output resources/downloader/tweets.csv \
--verbose
```