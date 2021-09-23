#  Copyright (c) 2021.
#  Chunyan Zhang
import ast
import os
import pandas as pd
from collections import Counter

#dir_name = "D:/twitter_data/vaccine_covid_origin_tweets/"
dir_name = "/data/twitter_data/covid_origin_tweets/"
covid_file_list = [#'tweet_202001_origin',
                    #'tweet_202002_origin',
                    #'Sampled_Stream_20200602_0603_origin_covid',
                    #'Sampled_Stream_20200604_0608_origin_covid',
                    'Filterd_Stream_20200629_origin_covid',
                    'Sampled_Stream_detail_20200608_0614_origin_covid',
                    'Sampled_Stream_detail_20200614_0619_origin_covid',
                    'Sampled_Stream_detail_20200619_0630_origin_covid',
                    'Sampled_Stream_detail_20200715_0720_origin_covid',
                    'Sampled_Stream_detail_20200720_0726_origin_covid',
                    'Sampled_Stream_detail_20200726_0731_origin_covid',
                    'Sampled_Stream_detail_20200731_0804_origin_covid',
                    'Sampled_Stream_detail_20200804_0807_origin_covid',
                    'Sampled_Stream_detail_20200807_0811_origin_covid',
                    'Sampled_Stream_detail_20200811_0815_origin_covid',
                    'Sampled_Stream_detail_20200816_0821_origin_covid',
                    'Sampled_Stream_detail_20200821_0824_origin_covid',
                    'Sampled_Stream_detail_20200825_0828_origin_covid',
                    'Sampled_Stream_detail_20200828_0830_origin_covid',
                    'Sampled_Stream_detail_20200830_0904_origin_covid',
                    'Sampled_Stream_detail_20200904_0908_origin_covid',
                    'Sampled_Stream_detail_20200910_0914_origin_covid',
                    'Sampled_Stream_detail_20200914_0917_origin_covid',
                    'Sampled_Stream_detail_20200917_0921_origin_covid',
                    'Sampled_Stream_detail_20200921_0924_origin_covid',
                    'Sampled_Stream_detail_20200924_0928_origin_covid',
                    'Sampled_Stream_detail_20200928_1002_origin_covid',
                    'Sampled_Stream_detail_20201002_1006_origin_covid',
                    'Sampled_Stream_detail_20201006_1009_origin_covid',
                    'Sampled_Stream_detail_20201009_1012_origin_covid',
                    'Sampled_Stream_detail_20201017_1020_origin_covid',
                    'Sampled_Stream_detail_20201020_1023_origin_covid',
                    'Sampled_Stream_detail_20201023_1031_origin_covid',
                    'Sampled_Stream_detail_20201031_1104_origin_covid',
                    'Sampled_Stream_detail_20201105_1110_origin_covid',
                    'Sampled_Stream_detail_20201110_1119_origin_covid',
                    'Sampled_Stream_detail_20201119_1124_origin_covid',
                    'Sampled_Stream_detail_20201124_1129_origin_covid',
                    'Sampled_Stream_detail_20201129_1205_origin_covid',
                    'Sampled_Stream_detail_20201205_1209_origin_covid',
                    'Sampled_Stream_detail_20201210_1214_origin_covid',
                    'Sampled_Stream_detail_20201214_1218_origin_covid',
                    'Sampled_Stream_detail_20201218_1224_origin_covid',
                    'Sampled_Stream_detail_20201224_1229_origin_covid',
                    'Sampled_Stream_detail_20201229_0103_origin_covid',
                    'Sampled_Stream_detail_20210103_0108_origin_covid',
                    'Sampled_Stream_detail_20210108_0112_origin_covid',
                    'Sampled_Stream_detail_20210112_0122_origin_covid',
                    'Sampled_Stream_detail_20210122_0128_origin_covid',
                    'Sampled_Stream_detail_20210128_0202_origin_covid',
                    'Sampled_Stream_detail_20210202_0205_origin_covid',
                    'Sampled_Stream_detail_20210206_0208_origin_covid',
                    'Sampled_Stream_detail_20210209_0213_origin_covid',
                    'Sampled_Stream_detail_20210213_0216_origin_covid',
                    'Sampled_Stream_detail_20210216_0219_origin_covid',
                    'Sampled_Stream_detail_20210219_0222_origin_covid',
                    'Sampled_Stream_detail_20210222_0225_origin_covid',
                    'Sampled_Stream_detail_20210225_0228_origin_covid',
                    'Sampled_Stream_detail_20210228_0304_origin_covid',
                    # update new data as of 2021.6
                    'Sampled_Stream_detail_20210304_0308_origin_covid',
                    'Sampled_Stream_detail_20210308_0312_origin_covid',
                    'Sampled_Stream_detail_20210312_0319_origin_covid',
                    'Sampled_Stream_detail_20210319_0323_origin_covid',
                    'Sampled_Stream_detail_20210323_0326_origin_covid',
                    'Sampled_Stream_detail_20210326_0329_origin_covid',
                    'Sampled_Stream_detail_20210329_0402_origin_covid',
                    'Sampled_Stream_detail_20210402_0406_origin_covid',
                    'Sampled_Stream_detail_20210406_0410_origin_covid',
                    'Sampled_Stream_detail_20210410_0416_origin_covid',
                    'Sampled_Stream_detail_20210416_0420_origin_covid',
                    'Sampled_Stream_detail_20210420_0423_origin_covid',
                    'Sampled_Stream_detail_20210423_0427_origin_covid',
                    'Sampled_Stream_detail_20210427_0501_origin_covid',
                    'Sampled_Stream_detail_20210501_0506_origin_covid',
                    'Sampled_Stream_detail_20210506_0512_origin_covid',
                    'Sampled_Stream_detail_20210512_0517_origin_covid',
                    'Sampled_Stream_detail_20210517_0522_origin_covid',
                    'Sampled_Stream_detail_20210522_0527_origin_covid',
                    'Sampled_Stream_detail_20210527_0530_origin_covid',
                    'Sampled_Stream_detail_20210530_0603_origin_covid',
                    'Sampled_Stream_detail_20210607_0616_origin_covid',
                    'Sampled_Stream_detail_20210616_0620_origin_covid',
                    'Sampled_Stream_detail_20210620_0624_origin_covid',
                    'Sampled_Stream_detail_20210624_0629_origin_covid',
                    'Sampled_Stream_detail_20210629_0703_origin_covid',]


def read_tweets(tweet_file):
    count = 0
    tweet_dates_list = []
    with open(tweet_file, "r", encoding='utf-8') as fhIn:
        for line in fhIn:
            if isinstance(line, str):
                line = ast.literal_eval(line)  # to dict
                if 'data' in line:
                    tweet_dates_list.append(line["data"]['created_at'][:10])
                else:
                    tweet_dates_list.append(line['created_at'][:10])
            else:
                print(line + "error")
                return None
            count += 1

            if count % 5000 == 0:
                print(count)

    print("{} read end".format(tweet_file))
    return tweet_dates_list

if __name__ == '__main__':
    total_date_list = []
    for file in covid_file_list:
        print("{}".format(file))
        covid_file_dir = os.path.join(dir_name, file)
        if not os.path.exists(covid_file_dir):
            print('{} not exsits.'.format(covid_file_dir))
            continue

        tweet_file = os.path.join(dir_name, file, 'covid_tweets.csv')

        if not os.path.exists(tweet_file):
            print('{} not exsits.'.format(tweet_file))
            continue
        total_date_list += read_tweets(tweet_file)

    total_date_df = pd.DataFrame(data=total_date_list, columns=['date'])
    date_count = Counter(total_date_df['date']).most_common()
    date_count_df = pd.DataFrame(data=date_count, columns=['date', 'count'])
    date_count_df = date_count_df.sort_values(by='date')
    date_count_df.to_csv(os.path.join(dir_name, "covid_date_count.csv"), index=False)