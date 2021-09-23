import pandas as pd
import os

#dir_name = "/data/twitter_data/vaccine_covid_origin_tweets/"
from config import dir_name, vaccine_file_list

#一年数据结果加上yearlong，否则不加
yearlong = 'yearlong'

def merge_single_result():
    for file in vaccine_file_list:
        print("anlyzing {} ...".format(file))
        vaccine_file_dir = os.path.join(dir_name, file)
        if not os.path.exists(vaccine_file_dir):
            print('{} not exsits.'.format(vaccine_file_dir))
            continue

        tweets_user_file = os.path.join(vaccine_file_dir, 'user_pred.csv')
        tweets_sentiment_file = os.path.join(vaccine_file_dir, 'tweets_sentiment_scores.csv')
        tweets_emotion_file = os.path.join(vaccine_file_dir, 'tweets_emotion_scores.csv')
        tweets_info_file = os.path.join(vaccine_file_dir, 'tweets_user_info.csv')
        result_file = os.path.join(vaccine_file_dir, 'tweets_analysis_result.csv')

        if os.path.exists(result_file):
            print('{} exsits.'.format(result_file))
            continue

        user_df = pd.read_csv(tweets_user_file)
        sentiment_df = pd.read_csv(tweets_sentiment_file)
        emotion_df = pd.read_csv(tweets_emotion_file)
        info_df = pd.read_csv(tweets_info_file)
        result_df = pd.concat([user_df, sentiment_df, emotion_df,
                               info_df[['twitter_age', 'verified', 'followers_count', 'following_count', 'tweet_count', 'listed_count']]], axis=1)
        result_df.to_csv(result_file, index=False)

        print("merge end")

def merge_all_results():
    print("merging all result ...")

    results_df = pd.DataFrame()
    for file in vaccine_file_list:
        print("{}".format(file))
        vaccine_file_dir = os.path.join(dir_name, file)
        if not os.path.exists(vaccine_file_dir):
            print('{} not exsits.'.format(vaccine_file_dir))
            continue

        result_file = os.path.join(vaccine_file_dir, 'tweets_analysis_result.csv')
        results_df = results_df.append(pd.read_csv(result_file))

    geo_df = pd.read_csv(os.path.join(dir_name, yearlong, 'vaccine_location_country_state_2.csv'), keep_default_na=False,
                         na_values=['_'], engine='python')
    job_df = pd.read_csv(os.path.join(dir_name, yearlong, 'job_pred.csv'))
    results_df['location'] = geo_df['location'].values
    results_df['state'] = geo_df['state'].values
    results_df['country'] = geo_df['country'].values
    results_df['job_type'] = job_df['job_type'].values
    results_df.to_csv(os.path.join(dir_name, yearlong, "tweets_analysis_country_state_result.csv"), index=False)

    print("merge end")

def merge_geo_results():
    result_df = pd.read_csv(os.path.join(dir_name, 'tweets_analysis_result.csv'))
    geo_df = pd.read_csv(os.path.join(dir_name, 'vaccine_location_country_state_2.csv'), keep_default_na=False, na_values=['_'], engine='python')
    job_df = pd.read_csv(os.path.join(dir_name, 'job_pred.csv'))

    result_df['location'] = geo_df['location']
    result_df['state'] = geo_df['state']
    result_df['country'] = geo_df['country']
    result_df['job_type'] = job_df['job_type']
    result_df.to_csv(os.path.join(dir_name, "tweets_analysis_country_state_result.csv"), index=False)

if __name__ == '__main__':
    #step 1: merge single file
    merge_single_result()

    #step 2: merge all results
    merge_all_results()

    #merge_geo_results()








