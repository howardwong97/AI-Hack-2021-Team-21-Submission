import praw
from psaw import PushshiftAPI
import pandas as pd
import sys
import datetime as dt
import ktrain
import numpy as np

start = int(dt.datetime(2020, 8, 3).timestamp())
end = int(dt.datetime(2020, 9, 4).timestamp())


def get_reddit_comments(search_terms: list, subreddits: list):
    reddit = praw.Reddit(client_id="",
                         client_secret="",
                         user_agent="")

    api = PushshiftAPI(reddit)
    body, timestamp, subreddit_name = [], [], []

    for query in search_terms:

        for subreddit in subreddits:
            print('Searching ' + subreddit + ' for:', query)
            gen = api.search_submissions(q=query, subreddit=subreddit, after=start, before=end)
            comment_counter = 0
            submission_counter = 0

            for submission in list(gen):
                submission.comments.replace_more(limit=None)
                submission_counter += 1
                sys.stdout.write("\033[F")  # back to previous line
                sys.stdout.write("\033[K")  # clear line
                print(str(submission_counter) + ' posts found')

                for comment in list(submission.comments):
                    body += [comment.body]
                    timestamp += [pd.to_datetime(int(comment.created_utc), unit='s').tz_localize('UTC')]
                    subreddit_name += [comment.subreddit.display_name]
                    comment_counter += 1
                    sys.stdout.write("\033[F")  # back to previous line
                    sys.stdout.write("\033[K")  # clear line
                    print(str(comment_counter) + ' comments found')
                    # Check that all are same length, otherwise just add a nan
                    if len(body) < len(timestamp) or len(body) < len(subreddit_name):
                        body += [np.nan]
                    elif len(timestamp) < len(body) or len(timestamp) < len(subreddit_name):
                        timestamp += [np.nan]
                    elif len(subreddit_name) < len(body) or len(subreddit_name) < len(timestamp):
                        subreddit_name += [np.nan]

    df = pd.DataFrame({'Timestamp': timestamp, 'Body': body, 'Subreddit': subreddit_name}).dropna()
    df.set_index('Timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df.drop_duplicates()
    return df


class RobertaModel:
    def __init__(self, model_directory):
        print("Initializing Roberta Model from:", model_directory + '...')
        self.model_directory = model_directory
        self.model = ktrain.load_predictor(model_directory).model
        self.preproc = ktrain.load_predictor(model_directory).preproc
        self.predictor = ktrain.get_predictor(self.model, self.preproc)
        print("Initialization complete.")

    def predict(self, text: str, output: int):
        """

        :param text: text that you would like to classify
        :param output: 0 for just prediction, 1 for probability, 2 for both
        :return: prediction in the form stipulated by output parameter
        """
        prediction = self.predictor.predict(text)
        probabilities = self.predictor.predict(text, return_proba=True)
        if output == 0:
            return prediction
        elif output == 1:
            return probabilities
        elif output == 2:
            return prediction, probabilities
        else:
            raise ValueError("Output must be 0, 1 or 2.")




