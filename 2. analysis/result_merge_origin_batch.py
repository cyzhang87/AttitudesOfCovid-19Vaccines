#  Copyright (c) 2021.
#  Chunyan Zhang

import pandas as pd
import os
import pycountry_convert as pc
from collections import Counter
import numpy as np

dir_name = "D:/twitter_data/origin_tweets/"
#dir_name = "./origin_tweets/"

origin_file_list = ['Sampled_Stream_detail_20200715_0720_origin',
                    'Sampled_Stream_detail_20200811_0815_origin',
                    'Sampled_Stream_detail_20200914_0917_origin',
                    'Sampled_Stream_detail_20201105_1110_origin',
                    'Sampled_Stream_detail_20201210_1214_origin',
                    'Sampled_Stream_detail_20210410_0416_origin']

def merge_single_result():
    for file in origin_file_list:
        print("anlyzing {} ...".format(file))
        file_dir = os.path.join(dir_name, file)
        if not os.path.exists(file_dir):
            print('{} not exsits.'.format(file_dir))
            continue

        user_df = pd.read_csv(os.path.join(file_dir, 'user_pred.csv'))
        sentiment_df = pd.read_csv(os.path.join(file_dir, 'tweets_sentiment_scores.csv'))
        emotion_df = pd.read_csv(os.path.join(file_dir, 'tweets_emotion_scores.csv'))
        info_df = pd.read_csv(os.path.join(file_dir, 'tweets_user_info.csv'))
        geo_df = pd.read_csv(os.path.join(file_dir, 'vaccine_location_country_state_2.csv'), keep_default_na=False,
                             na_values=['_'], engine='python')
        job_df = pd.read_csv(os.path.join(file_dir, 'occ_pred.csv'))

        result_df = pd.concat([user_df, sentiment_df, emotion_df,
                               info_df[['twitter_age', 'verified', 'followers_count', 'following_count', 'tweet_count', 'listed_count']],
                               geo_df[['location', 'state', 'country']], job_df], axis=1)
        result_df.to_csv(os.path.join(file_dir, 'tweets_analysis_result.csv'), index=False)

        print("merge end")

def merge_all_results():
    print("merging all result ...")

    results_df = pd.DataFrame()
    for file in origin_file_list:
        print("{}".format(file))
        vaccine_file_dir = os.path.join(dir_name, file)
        if not os.path.exists(vaccine_file_dir):
            print('{} not exsits.'.format(vaccine_file_dir))
            continue

        result_file = os.path.join(vaccine_file_dir, 'tweets_analysis_result.csv')

        if not os.path.exists(result_file):
            print('{} not exsits.'.format(result_file))
            continue

        results_df = results_df.append(pd.read_csv(result_file, keep_default_na=False, na_values=['_'], engine='python'))

    results_df.to_csv(os.path.join(dir_name, "tweets_analysis_result.csv"), index=False)
    print("merge end")

def convert_continents(file_name):
    continents = {
        'AF': 'Africa',
        'AN': 'Antarctica',
        'AS': 'Asia',
        'EU': 'Europe',
        'OC': 'Oceania',
        'NA': 'North America',
        'SA': 'South America',
    }

    results_df = pd.read_csv(file_name, keep_default_na=False, na_values=['_'], engine='python')

    continents_list = []
    for index, item in results_df.iterrows():
        if index % 1000 == 0:
            print(index)
        if item['country'] == 'null loc' or str(item['country']) == 'nan' or str(item['country']) == '':
            continents_list.append('null loc')
        else:
            continents_list.append(continents[pc.country_alpha2_to_continent_code(item['country'])])

    results_df['continent'] = continents_list
    results_df.to_csv(file_name, index=False)

def add_twitter_age_following_count(file_name):
    results_df = pd.read_csv(file_name, keep_default_na=False, na_values=['_'], engine='python')
    twi_age_th = [0, 5, 10]
    twi_follow_th = [0, 500, 5000]
    twi_age_list = []
    twi_follow_list = []

    for index, item in results_df.iterrows():
        if index % 1000 == 0:
            print(index)
        if item['twitter_age'] < twi_age_th[1]:
            twi_age_list.append(0)
        elif item['twitter_age'] < twi_age_th[2]:
            twi_age_list.append(1)
        else:
            twi_age_list.append(2)

        if item['followers_count'] < twi_follow_th[1]:
            twi_follow_list.append(0)
        elif item['followers_count'] < twi_follow_th[2]:
            twi_follow_list.append(1)
        else:
            twi_follow_list.append(2)

    results_df['twitter_age_class'] = twi_age_list
    results_df['followers_count_class'] = twi_follow_list
    results_df.to_csv(file_name, index=False)

if __name__ == '__main__':
    #step 1: merge single file，当前只需要这一步
    #merge_single_result()

    #step 2: merge all results
    #merge_all_results()

    #转换continent
    origin_file = os.path.join(dir_name, "tweets_analysis_result.csv")
    vaccine_file = "D:/twitter_data/vaccine_covid_origin_tweets/yearlong/tweets_analysis_country_state_result_order_new.csv"
    #convert_continents(origin_file)
    #add_twitter_age_following_count(origin_file)
    results_df = pd.read_csv(vaccine_file, keep_default_na=False, na_values=['_'], engine='python')
    continent_count = pd.DataFrame(Counter(results_df['continent']).most_common(), columns=["continent", "count"])
    continent_count = continent_count.drop(continent_count[continent_count['continent']=='null loc'].index)
    continent_count.reset_index(drop=True, inplace=True)
    location_sum = np.sum(continent_count['count'])
    for index, row in continent_count.iterrows():
        print('{}: {}, {:.2f} '.format(row['continent'], row['count'], row['count'] / location_sum * 100))
    print('mean: {:.4f} sd: {:.4f}'.format(np.mean(results_df['senti-score']), np.std(results_df['senti-score'])))
    county_count = pd.DataFrame(Counter(results_df['country']).most_common(), columns=["country", "count"])
    county_count = county_count.drop(county_count[county_count['country'] == 'null loc'].index)
    county_count.reset_index(drop=True, inplace=True)
    location_sum = np.sum(county_count['count'])
    for index, row in county_count.iterrows():
        print('{}: {}, {:.2f} '.format(row['country'], row['count'], row['count'] / location_sum * 100))
        if index >= 5:
            break

    print('end')

