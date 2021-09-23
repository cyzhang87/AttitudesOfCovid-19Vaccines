#  Copyright (c) 2021.
#  Chunyan Zhang

import pandas as pd
import os
import ast
from datetime import datetime
from config import dir_name, vaccine_file_list

def read_tweets(tweet_file):
    tweets_time_list = []
    with open(tweet_file, "r", encoding='utf-8') as fhIn:
        count = 0
        for line in fhIn:
            if isinstance(line, str):
                line = ast.literal_eval(line)  # to dict
                if 'data' in line:
                    x = datetime.strptime(line['includes']['users'][0]['created_at'][:10], '%Y-%m-%d')
                    y = datetime.strptime(line["data"]['created_at'][:10], '%Y-%m-%d')
                    t = format((y - x).days / 365, '.2f')
                    if line['includes']['users'][0]['verified'] == False:
                        v = 0
                    else:
                        v = 1

                    stats = 'stats'
                    if 'stats' not in line['includes']['users'][0]:
                        stats = 'public_metrics'

                    tweets_time_list.append([t, v,
                                             line['includes']['users'][0][stats]['followers_count'],
                                             line['includes']['users'][0][stats]['following_count'],
                                             line['includes']['users'][0][stats]['tweet_count'],
                                             line['includes']['users'][0][stats]['listed_count'],
                                             line['includes']['users'][0]['id']])
                else:
                    print(line + "error1")
            else:
                print(line + "error2")
                return None
            count += 1

            if count % 5000 == 0:
                print(count)

    print("read end")
    return tweets_time_list


if __name__ == '__main__':
    for file in vaccine_file_list:
        print("anlyzing {} ...".format(file))
        vaccine_file_dir = os.path.join(dir_name, file)
        if not os.path.exists(vaccine_file_dir):
            print('{} not exsits.'.format(vaccine_file_dir))
            continue
        tweet_file = os.path.join(vaccine_file_dir, 'vaccine_tweets.csv')
        virtual_file = os.path.join(vaccine_file_dir, "tweets_user_info.csv")
        if os.path.exists(virtual_file):
            print('{} exsits.'.format(virtual_file))
            continue

        tweets_time_list = read_tweets(tweet_file)
        pred_df = pd.DataFrame(
            columns=['twitter_age', 'verified', 'followers_count', 'following_count', 'tweet_count', 'listed_count', 'user_id'],
            data=tweets_time_list)
        pred_df.to_csv(virtual_file, index=False)
