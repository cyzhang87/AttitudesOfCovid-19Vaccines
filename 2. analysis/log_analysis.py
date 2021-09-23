#  Copyright (c) 2021.
#  Chunyan Zhang

import numpy as np
import scipy.stats as st
import pandas as pd
import os

######################################
## covid tweets
######################################

log_filename = '../tweet_preprocess/log/log-filter_covid_batch.txt'

percent_list = []
for line in open(log_filename, 'r', encoding='utf-8'):
    if 'percent: ' in line:
        percent = line[line.find('percent: ') + len('percent: '):-2]
        try:
            percent = float(percent)
            if percent < 10:
                percent_list.append(percent)
            else:
                print("{} too big.".format(percent))
        except ValueError:
            print("{} not a number".format(percent))

data_mean = np.mean(percent_list)
data_std = np.std(percent_list)
print('COVID-19: {:.4f} + {:.4f}'.format(data_mean, data_std))

ci = st.t.interval(0.95, len(percent_list)-1, loc=np.mean(percent_list), scale=st.sem(percent_list))
print('COVID-19: {:.2f}% (95% CI {:.2f}%-{:.2f}%)'.format(data_mean, ci[0], ci[1]))
print('[{:.2f}, {:.2f}]'.format(min(percent_list), max(percent_list)))

######################################
## vaccine tweets
######################################
print('')
log_filename = '../tweet_preprocess/log/log-filter_vaccine_batch.txt'

percent_list = []
for line in open(log_filename, 'r', encoding='utf-8'):
    if 'percent: ' in line:
        percent = line[line.find('percent: ') + len('percent: '):-2]
        try:
            percent = float(percent)
            percent_list.append(percent)
        except ValueError:
            print("{} not a number".format(percent))

data_mean = np.mean(percent_list)
data_std = np.std(percent_list)
print('Vaccine: {:.4f} + {:.4f}'.format(data_mean, data_std))

ci = st.t.interval(0.95, len(percent_list)-1, loc=np.mean(percent_list), scale=st.sem(percent_list))
print('Vaccine: {:.2f}% (95% CI {:.2f}%-{:.2f}%)'.format(data_mean, ci[0], ci[1]))
print('[{:.2f}, {:.2f}]'.format(min(percent_list), max(percent_list)))

######################################
## vaccine tweets
######################################
print('')
dir_name = "D:/twitter_data/vaccine_covid_origin_tweets/"
origin_file = 'tweets_analysis_country_state_result.csv'
date_order_file = "tweets_analysis_country_state_result_order.csv"
covid_count_file = "D:/twitter_data/covid_origin_tweets/covid_date_count.csv"

covid_count_df = pd.read_csv(covid_count_file)
covid_count = covid_count_df.set_index('date').T.to_dict()

df = pd.read_csv(os.path.join(dir_name, date_order_file))
date_list = sorted(set(df['date']))
date_list = date_list[:len(date_list)-1]
total_list = []
for i in range(len(date_list)):
    total_df = df[df['date'] == date_list[i]]
    total_list.append(total_df.shape[0] / covid_count[date_list[i]]['count'] * 100)
percent_list = total_list

data_mean = np.mean(percent_list)
data_std = np.std(percent_list)
print('Vaccine: {:.4f} + {:.4f}'.format(data_mean, data_std))

ci = st.t.interval(0.95, len(percent_list)-1, loc=np.mean(percent_list), scale=st.sem(percent_list))
print('Vaccine: {:.2f}% (95% CI {:.2f}%-{:.2f}%)'.format(data_mean, ci[0], ci[1]))
print('[{:.2f}, {:.2f}]'.format(min(percent_list), max(percent_list)))
print('end')

