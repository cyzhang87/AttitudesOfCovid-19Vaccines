#  Copyright (c) 2021.
#  Chunyan Zhang
import pandas as pd
import numpy as np
import os
import math
import statsmodels.api as sm

dir_name = "D:/twitter_data/origin_tweets/"
origin_file_list = ['Sampled_Stream_detail_20200715_0720_origin',
                    'Sampled_Stream_detail_20200811_0815_origin',
                    'Sampled_Stream_detail_20201105_1110_origin',
                    'Sampled_Stream_detail_20201210_1214_origin',
                    'Sampled_Stream_detail_20210410_0416_origin']
vaccine_file = "D:/twitter_data/vaccine_covid_origin_tweets/yearlong/tweets_analysis_country_state_result_order_new.csv"
date_list = ['2020-07-15', '2020-08-12', '2020-11-09', '2020-12-15', '2021-04-10']
vaccine_df = pd.read_csv(vaccine_file)

origin_df_list = []
for file in origin_file_list:
    origin_df_list.append(pd.read_csv(os.path.join(dir_name, file, "tweets_analysis_result.csv")))

vaccine_date_list = sorted(set(vaccine_df['date']))
vaccine_date_list = vaccine_date_list[:len(vaccine_date_list) - 1]

######################################
##     public functions
######################################
def refine_timeline(date_list, value_list, start_time, end_time, columns):
    index = 0
    last_value = value_list[0]
    total_list = []
    for date in pd.date_range(start=start_time, end=end_time):
        date_str = date.strftime("%Y-%m-%d")
        if date_str != date_list[index]:
            value_tmp = last_value
        else:
            value_tmp = value_list[index]
            last_value = value_tmp
            index += 1
        if date_str == "2020-06-11":
            value_tmp = total_list[len(total_list) - 1][1]
        total_list.append([date_str, value_tmp])

    return pd.DataFrame(data=total_list, columns=columns)

def Smooth(vector, smooth_size=2):
    # vector_out = vector.copy()
    vector_out = vector
    if len(vector) < 2 * smooth_size + 1:
        return vector_out
    if smooth_size == 0:
        return  vector_out

    for i in range(len(vector)):
        iStart = i - smooth_size
        iEnd = i + smooth_size

        if iStart < 0:
            iStart = 0
            iEnd = iStart + (2 * smooth_size + 1)
        if iEnd > len(vector):
            iEnd = len(vector)
            iStart = iEnd - (2 * smooth_size + 1)
        if iStart < 0:
            iStart = 0
        vector_part = vector[iStart : iEnd]
        # print(len(vector_part))
        vector_out[i] = np.sum(vector_part)/len(vector_part)

    return vector_out

######################################
##     chi-square
######################################
from  scipy.stats import chi2_contingency
# continent
kf_data = np.array([[2088, 7649, 7333, 27535, 206, 5, 1155],
                    [2933, 5356, 5453, 15470, 366, 6, 610]])
kf = chi2_contingency(kf_data)
print('chisq-statistic=%.4f, p-value=%.4f, df=%i expected_frep=%s'%kf)

# Country
kf_data = np.array([[26447, 4886, 4617, 788, 758],
                    [14497, 2630, 754, 215, 202]])
kf = chi2_contingency(kf_data)
print('chisq-statistic=%.4f, p-value=%.4f, df=%i expected_frep=%s'%kf)

# Twitter age
kf_data = np.array([[18469, 16843, 13657],
                    [25333, 13640, 5288]])
kf = chi2_contingency(kf_data)
print('chisq-statistic=%.4f, p-value=%.4f, df=%i expected_frep=%s'%kf)

# follower count
kf_data = np.array([[24472, 17456, 7041],
                    [25960, 15481, 2820]])
kf = chi2_contingency(kf_data)
print('chisq-statistic=%.4f, p-value=%.4f, df=%i expected_frep=%s'%kf)

# sentiment
kf_data = np.array([[29219, 20149, 20246],
                    [21547, 15666, 12787]])
kf = chi2_contingency(kf_data)
print('chisq-statistic=%.4f, p-value=%.4f, df=%i expected_frep=%s'%kf)

# emotion
kf_data = np.array([[23348, 11743, 29563, 2001, 1765, 1194],
                    [20715, 13649, 6866, 5869, 2433, 468]])
kf = chi2_contingency(kf_data)
print('chisq-statistic=%.4f, p-value=%.4f, df=%i expected_frep=%s'%kf)

######################################
##     user type
######################################
OR_list = []
type_list = [[], []]
for i in range(len(vaccine_date_list)):
    total_df = vaccine_df[vaccine_df['date'] == vaccine_date_list[i]]

    for j in range(len(type_list)):
        type_df = total_df[total_df['org'] == j]
        type_list[j].append(type_df.shape[0])

type_df_list = []
for j in range(len(type_list)):
    type_df = refine_timeline(vaccine_date_list, type_list[j], vaccine_date_list[0], vaccine_date_list[-1], ['date', 'type'])
    type_df['type'] = Smooth(Smooth(type_df['type'], 3), 3)
    type_df_list.append(type_df)

print("User Type:")
for i in range(len(date_list)):
    tmp_vaccine_df = vaccine_df[vaccine_df['date'] == date_list[i]]

    OR_list.append(sum(tmp_vaccine_df['org'] == 1) / sum(tmp_vaccine_df['org'] == 0) * sum(origin_df_list[i]['org'] == 0) / sum(origin_df_list[i]['org'] == 1))
    #tabel_or = [[sum(tmp_vaccine_df['org'] == 1), sum(origin_df_list[i]['org'] == 1)],
    #            [sum(tmp_vaccine_df['org'] == 0), sum(origin_df_list[i]['org'] == 0)]]
    # print(result.summary(method='normal'))
    # delt_ci = math.exp(1.96 * math.pow(1 / sum(tmp_vaccine_df['org'] == 1) + 1 / sum(tmp_vaccine_df['org'] == 0), 0.5))

    tabel_or = [[type_df_list[1][type_df_list[1]['date'] == date_list[i]]['type'].values[0], sum(origin_df_list[i]['org'] == 1) * 100],
                [type_df_list[0][type_df_list[0]['date'] == date_list[i]]['type'].values[0], sum(origin_df_list[i]['org'] == 0) * 100]]
    result = sm.stats.Table2x2(tabel_or)
    oddsratio = result.oddsratio
    lcb = result.oddsratio_confint()[0]
    ucb = result.oddsratio_confint()[1]
    pvalue = result.oddsratio_pvalue()
    print("{}: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(i, oddsratio, lcb, ucb, pvalue))
    senti_str = 'vaccine: '
    for j in range(len(type_list)):
        senti_str += '  {}: {:.4f}'.format(j, np.mean(tmp_vaccine_df[tmp_vaccine_df['org'] == j]['senti-score']))
    print(senti_str)
    senti_str = 'origin: '
    for j in range(len(type_list)):
        senti_str += '  {}: {:.4f}'.format(j, np.mean(origin_df_list[i][origin_df_list[i]['org'] == j]['senti-score']))
    print(senti_str)

######################################
##     gender
######################################
OR_list = []
type_list = [[], []]
for i in range(len(vaccine_date_list)):
    total_df = vaccine_df[vaccine_df['date'] == vaccine_date_list[i]]
    person_df = total_df[total_df['org'] == 0]

    for j in range(len(type_list)):
        type_df = person_df[person_df['gender'] == j]
        type_list[j].append(type_df.shape[0])

type_df_list = []
for j in range(len(type_list)):
    type_df = refine_timeline(vaccine_date_list, type_list[j], vaccine_date_list[0], vaccine_date_list[-1], ['date', 'type'])
    type_df['type'] = Smooth(Smooth(type_df['type'], 3), 3)
    type_df_list.append(type_df)

print("Gender:")
for i in range(len(date_list)):
    tmp_vaccine_df = vaccine_df[vaccine_df['date'] == date_list[i]]
    tmp_vaccine_df = tmp_vaccine_df[tmp_vaccine_df['org'] == 0]
    tmp_origin_df = origin_df_list[i][origin_df_list[i]['org'] == 0]
    OR_list.append(sum(tmp_vaccine_df['gender'] == 1) / sum(tmp_vaccine_df['gender'] == 0) * sum(tmp_origin_df['gender'] == 0) / sum(tmp_origin_df['gender'] == 1))

    tabel_or = [[type_df_list[1][type_df_list[1]['date'] == date_list[i]]['type'].values[0], sum(tmp_origin_df['gender'] == 1) * 100],
                [type_df_list[0][type_df_list[0]['date'] == date_list[i]]['type'].values[0], sum(tmp_origin_df['gender'] == 0) * 100]]
    result = sm.stats.Table2x2(tabel_or)
    oddsratio = result.oddsratio
    lcb = result.oddsratio_confint()[0]
    ucb = result.oddsratio_confint()[1]
    pvalue = result.oddsratio_pvalue()
    print("{}: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(i, oddsratio, lcb, ucb, pvalue))

    type = 'gender'
    senti_str = 'vaccine: '
    for j in range(len(type_list)):
        senti_str += '  {}: {:.4f}'.format(j, np.mean(tmp_vaccine_df[tmp_vaccine_df[type] == j]['senti-score']))
    print(senti_str)
    senti_str = 'origin: '
    for j in range(len(type_list)):
        senti_str += '  {}: {:.4f}'.format(j, np.mean(tmp_origin_df[tmp_origin_df[type] == j]['senti-score']))
    print(senti_str)

######################################
##     age
######################################
OR_list = []
type_list = [[], [], [], []]
for i in range(len(vaccine_date_list)):
    total_df = vaccine_df[vaccine_df['date'] == vaccine_date_list[i]]
    person_df = total_df[total_df['org'] == 0]

    for j in range(len(type_list)):
        type_df = person_df[person_df['age'] == j]
        type_list[j].append(type_df.shape[0])

type_df_list = []
for j in range(len(type_list)):
    type_df = refine_timeline(vaccine_date_list, type_list[j], vaccine_date_list[0], vaccine_date_list[-1], ['date', 'type'])
    type_df['type'] = Smooth(type_df['type'], 2)
    type_df_list.append(type_df)

print("Age:")
for i in range(len(date_list)):
    tmp_vaccine_df = vaccine_df[vaccine_df['date'] == date_list[i]]
    tmp_vaccine_df = tmp_vaccine_df[tmp_vaccine_df['org'] == 0]
    tmp_origin_df = origin_df_list[i][origin_df_list[i]['org'] == 0]
    OR_age_list = []
    for j in range(1, len(type_list)):
        OR_age_list.append(sum(tmp_vaccine_df['age'] == j) / sum(tmp_vaccine_df['age'] == 0) *
                       sum(tmp_origin_df['age'] == 0) / sum(tmp_origin_df['age'] == j))

        tabel_or = [[type_df_list[j][type_df_list[j]['date'] == date_list[i]]['type'].values[0] * 10,
                     sum(tmp_origin_df['age'] == j) * 1000],
                    [type_df_list[0][type_df_list[0]['date'] == date_list[i]]['type'].values[0] * 10,
                     sum(tmp_origin_df['age'] == 0) * 1000]]
        result = sm.stats.Table2x2(tabel_or)
        oddsratio = result.oddsratio
        lcb = result.oddsratio_confint()[0]
        ucb = result.oddsratio_confint()[1]
        pvalue = result.oddsratio_pvalue()
        print("age {}: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(j, oddsratio, lcb, ucb, pvalue))
        senti_str = 'vaccine: '
        type = 'age'
        for j in range(len(type_list)):
            senti_str += '  {}: {:.4f}'.format(j, np.mean(tmp_vaccine_df[tmp_vaccine_df[type] == j]['senti-score']))
        print(senti_str)
        senti_str = 'origin: '
        for j in range(len(type_list)):
            senti_str += '  {}: {:.4f}'.format(j, np.mean(tmp_origin_df[tmp_origin_df[type] == j]['senti-score']))
        print(senti_str)

    OR_list.append(OR_age_list)

print(OR_list)
######################################
##     occupation
######################################
OR_list = []
type_list = [[], [], []]
for i in range(len(vaccine_date_list)):
    total_df = vaccine_df[vaccine_df['date'] == vaccine_date_list[i]]
    person_df = total_df[total_df['org'] == 0]

    for j in range(len(type_list)):
        type_df = person_df[person_df['job_type'] == j]
        type_list[j].append(type_df.shape[0])

type_df_list = []
for j in range(len(type_list)):
    type_df = refine_timeline(vaccine_date_list, type_list[j], vaccine_date_list[0], vaccine_date_list[-1], ['date', 'type'])
    type_df['type'] = Smooth(type_df['type'], 2)
    type_df_list.append(type_df)

print("Occupation:")
for i in range(len(date_list)):
    tmp_vaccine_df = vaccine_df[vaccine_df['date'] == date_list[i]]
    tmp_vaccine_df = tmp_vaccine_df[tmp_vaccine_df['org'] == 0]
    tmp_origin_df = origin_df_list[i][origin_df_list[i]['org'] == 0]
    OR_job_list = []
    for j in range(1, len(type_list)):
        OR_job_list.append(sum(tmp_vaccine_df['job_type'] == j) / sum(tmp_vaccine_df['job_type'] == 0) *
                       sum(tmp_origin_df['job_type'] == 0) / sum(tmp_origin_df['job_type'] == j))

        tabel_or = [[type_df_list[j][type_df_list[j]['date'] == date_list[i]]['type'].values[0],
                     sum(tmp_origin_df['job_type'] == j) * 100],
                    [type_df_list[0][type_df_list[0]['date'] == date_list[i]]['type'].values[0],
                     sum(tmp_origin_df['job_type'] == 0) * 100]]

        result = sm.stats.Table2x2(tabel_or)
        oddsratio = result.oddsratio
        lcb = result.oddsratio_confint()[0]
        ucb = result.oddsratio_confint()[1]
        pvalue = result.oddsratio_pvalue()
        print("job {}: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(j, oddsratio, lcb, ucb, pvalue))
    OR_list.append(OR_job_list)

print(OR_list)

######################################
##     sentiment
######################################
senti_3class = []
for index, item in vaccine_df.iterrows():
    if item['senti-degree'] > 0:
        senti_3class.append(0)
    elif item['senti-degree'] == 0:
        senti_3class.append(1)
    else:
        senti_3class.append(2)
vaccine_df['senti_3class'] = senti_3class

for i in range(len(date_list)):
    senti_3class = []
    for index, item in origin_df_list[i].iterrows():
        if item['senti-degree'] > 0:
            senti_3class.append(0)
        elif item['senti-degree'] == 0:
            senti_3class.append(1)
        else:
            senti_3class.append(2)
    origin_df_list[i]['senti_3class'] = senti_3class

OR_list = []
type_list = [[], [], []]
for i in range(len(vaccine_date_list)):
    total_df = vaccine_df[vaccine_df['date'] == vaccine_date_list[i]]
    person_df = total_df[total_df['org'] == 0]
    for j in range(len(type_list)):
        type_df = person_df[person_df['senti_3class'] == j]
        type_list[j].append(type_df.shape[0])

type_df_list = []
for j in range(len(type_list)):
    type_df = refine_timeline(vaccine_date_list, type_list[j], vaccine_date_list[0], vaccine_date_list[-1], ['date', 'type'])
    type_df['type'] = Smooth(type_df['type'], 3)
    type_df_list.append(type_df)

print("Sentiment 3class:")
for i in range(len(date_list)):
    tmp_vaccine_df = vaccine_df[vaccine_df['date'] == date_list[i]]
    tmp_origin_df = origin_df_list[i][origin_df_list[i]['org'] == 0]
    OR_senti_list = []
    for j in range(1, len(type_list)):
        OR_senti_list.append(sum(tmp_vaccine_df['senti_3class'] == j) / sum(tmp_vaccine_df['senti_3class'] == 0) *
                       sum(tmp_origin_df['senti_3class'] == 0) / sum(tmp_origin_df['senti_3class'] == j))

        tabel_or = [[type_df_list[j][type_df_list[j]['date'] == date_list[i]]['type'].values[0],
                     sum(tmp_origin_df['senti_3class'] == j) * 100],
                    [type_df_list[0][type_df_list[0]['date'] == date_list[i]]['type'].values[0],
                     sum(tmp_origin_df['senti_3class'] == 0) * 100]]

        result = sm.stats.Table2x2(tabel_or)
        oddsratio = result.oddsratio
        lcb = result.oddsratio_confint()[0]
        ucb = result.oddsratio_confint()[1]
        pvalue = result.oddsratio_pvalue()
        print("sentiment {}: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(j, oddsratio, lcb, ucb, pvalue))
    OR_list.append(OR_senti_list)

print(OR_list)


######################################
##     twitter age
######################################
OR_list = []
type_list = [[], [], []]
twi_age_th = [0, 5, 10]
type_name = 'twitter_age_3class'
twi_age_list = []
for index, item in vaccine_df.iterrows():
    if item['twitter_age'] < twi_age_th[1]:
        twi_age_list.append(0)
    elif item['twitter_age'] < twi_age_th[2]:
        twi_age_list.append(1)
    else:
        twi_age_list.append(2)
vaccine_df[type_name] = twi_age_list

for i in range(len(date_list)):
    twi_age_list = []
    for index, item in origin_df_list[i].iterrows():
        if item['twitter_age'] < twi_age_th[1]:
            twi_age_list.append(0)
        elif item['twitter_age'] < twi_age_th[2]:
            twi_age_list.append(1)
        else:
            twi_age_list.append(2)
    origin_df_list[i][type_name] = twi_age_list

for i in range(len(vaccine_date_list)):
    total_df = vaccine_df[vaccine_df['date'] == vaccine_date_list[i]]
    person_df = total_df[total_df['org'] == 0]

    for j in range(len(type_list)):
        type_df = person_df[person_df[type_name] == j]
        type_list[j].append(type_df.shape[0])

type_df_list = []
for j in range(len(type_list)):
    type_df = refine_timeline(vaccine_date_list, type_list[j], vaccine_date_list[0], vaccine_date_list[-1], ['date', 'type'])
    type_df['type'] = Smooth(type_df['type'], 2)
    type_df_list.append(type_df)

print("Twitter age:")
for i in range(len(date_list)):
    tmp_vaccine_df = vaccine_df[vaccine_df['date'] == date_list[i]]
    tmp_vaccine_df = tmp_vaccine_df[tmp_vaccine_df['org'] == 0]
    tmp_origin_df = origin_df_list[i][origin_df_list[i]['org'] == 0]
    OR_job_list = []
    for j in range(1, len(type_list)):
        OR_job_list.append(sum(tmp_vaccine_df[type_name] == j) / sum(tmp_vaccine_df[type_name] == 0) *
                       sum(tmp_origin_df[type_name] == 0) / sum(tmp_origin_df[type_name] == j))

        tabel_or = [[type_df_list[j][type_df_list[j]['date'] == date_list[i]]['type'].values[0],
                     sum(tmp_origin_df[type_name] == j) * 100],
                    [type_df_list[0][type_df_list[0]['date'] == date_list[i]]['type'].values[0],
                     sum(tmp_origin_df[type_name] == 0) * 100]]

        result = sm.stats.Table2x2(tabel_or)
        oddsratio = result.oddsratio
        lcb = result.oddsratio_confint()[0]
        ucb = result.oddsratio_confint()[1]
        pvalue = result.oddsratio_pvalue()
        print("{}: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(j, oddsratio, lcb, ucb, pvalue))
    OR_list.append(OR_job_list)

#print(OR_list)

######################################
##     tweet count
######################################
OR_list = []
type_list = [[], [], []]
twi_cnt_th = [0, 1000, 2000]
type_name = 'twitter_count_3class'
twi_age_list = []
for index, item in vaccine_df.iterrows():
    if item['twitter_age'] == 0:
        item['twitter_age'] = 0.01
    tmp_cnt = item['tweet_count'] / item['twitter_age']
    if tmp_cnt < twi_cnt_th[1]:
        twi_age_list.append(0)
    elif tmp_cnt < twi_cnt_th[2]:
        twi_age_list.append(1)
    else:
        twi_age_list.append(2)
vaccine_df[type_name] = twi_age_list

for i in range(len(date_list)):
    twi_age_list = []
    for index, item in origin_df_list[i].iterrows():
        if item['twitter_age'] == 0:
            item['twitter_age'] = 0.01
        tmp_cnt = item['tweet_count'] / item['twitter_age']
        if tmp_cnt < twi_cnt_th[1]:
            twi_age_list.append(0)
        elif tmp_cnt < twi_cnt_th[2]:
            twi_age_list.append(1)
        else:
            twi_age_list.append(2)
    origin_df_list[i][type_name] = twi_age_list

for i in range(len(vaccine_date_list)):
    total_df = vaccine_df[vaccine_df['date'] == vaccine_date_list[i]]
    person_df = total_df[total_df['org'] == 0]

    for j in range(len(type_list)):
        type_df = person_df[person_df[type_name] == j]
        type_list[j].append(type_df.shape[0])

type_df_list = []
for j in range(len(type_list)):
    type_df = refine_timeline(vaccine_date_list, type_list[j], vaccine_date_list[0], vaccine_date_list[-1], ['date', 'type'])
    type_df['type'] = Smooth(type_df['type'], 2)
    type_df_list.append(type_df)

print("Tweet count:")
for i in range(len(date_list)):
    tmp_vaccine_df = vaccine_df[vaccine_df['date'] == date_list[i]]
    tmp_vaccine_df = tmp_vaccine_df[tmp_vaccine_df['org'] == 0]
    tmp_origin_df = origin_df_list[i][origin_df_list[i]['org'] == 0]
    OR_job_list = []
    for j in range(1, len(type_list)):
        OR_job_list.append(sum(tmp_vaccine_df[type_name] == j) / sum(tmp_vaccine_df[type_name] == 0) *
                       sum(tmp_origin_df[type_name] == 0) / sum(tmp_origin_df[type_name] == j))

        tabel_or = [[type_df_list[j][type_df_list[j]['date'] == date_list[i]]['type'].values[0],
                     sum(tmp_origin_df[type_name] == j) * 100],
                    [type_df_list[0][type_df_list[0]['date'] == date_list[i]]['type'].values[0],
                     sum(tmp_origin_df[type_name] == 0) * 100]]

        result = sm.stats.Table2x2(tabel_or)
        oddsratio = result.oddsratio
        lcb = result.oddsratio_confint()[0]
        ucb = result.oddsratio_confint()[1]
        pvalue = result.oddsratio_pvalue()
        print("{}: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(j, oddsratio, lcb, ucb, pvalue))
    OR_list.append(OR_job_list)

print(OR_list)


######################################
##     twitter follower
######################################
OR_list = []
type_list = [[], [], []]
twi_follow_th = [0, 500, 5000]
type_name = 'twitter_follower_3class'
twi_age_list = []
for index, item in vaccine_df.iterrows():
    if item['followers_count'] < twi_follow_th[1]:
        twi_age_list.append(0)
    elif item['followers_count'] < twi_follow_th[2]:
        twi_age_list.append(1)
    else:
        twi_age_list.append(2)
vaccine_df[type_name] = twi_age_list

for i in range(len(date_list)):
    twi_age_list = []
    for index, item in origin_df_list[i].iterrows():
        if item['followers_count'] < twi_follow_th[1]:
            twi_age_list.append(0)
        elif item['followers_count'] < twi_follow_th[2]:
            twi_age_list.append(1)
        else:
            twi_age_list.append(2)
    origin_df_list[i][type_name] = twi_age_list

for i in range(len(vaccine_date_list)):
    total_df = vaccine_df[vaccine_df['date'] == vaccine_date_list[i]]
    person_df = total_df[total_df['org'] == 0]

    for j in range(len(type_list)):
        type_df = person_df[person_df[type_name] == j]
        type_list[j].append(type_df.shape[0])

type_df_list = []
for j in range(len(type_list)):
    type_df = refine_timeline(vaccine_date_list, type_list[j], vaccine_date_list[0], vaccine_date_list[-1], ['date', 'type'])
    type_df['type'] = Smooth(type_df['type'], 2)
    type_df_list.append(type_df)

print("Twitter follower:")
for i in range(len(date_list)):
    tmp_vaccine_df = vaccine_df[vaccine_df['date'] == date_list[i]]
    tmp_vaccine_df = tmp_vaccine_df[tmp_vaccine_df['org'] == 0]
    tmp_origin_df = origin_df_list[i][origin_df_list[i]['org'] == 0]
    OR_job_list = []
    for j in range(1, len(type_list)):
        OR_job_list.append(sum(tmp_vaccine_df[type_name] == j) / sum(tmp_vaccine_df[type_name] == 0) *
                       sum(tmp_origin_df[type_name] == 0) / sum(tmp_origin_df[type_name] == j))

        tabel_or = [[type_df_list[j][type_df_list[j]['date'] == date_list[i]]['type'].values[0],
                     sum(tmp_origin_df[type_name] == j) * 100],
                    [type_df_list[0][type_df_list[0]['date'] == date_list[i]]['type'].values[0],
                     sum(tmp_origin_df[type_name] == 0) * 100]]

        result = sm.stats.Table2x2(tabel_or)
        oddsratio = result.oddsratio
        lcb = result.oddsratio_confint()[0]
        ucb = result.oddsratio_confint()[1]
        pvalue = result.oddsratio_pvalue()
        print("{}: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(j, oddsratio, lcb, ucb, pvalue))
    OR_list.append(OR_job_list)

print(OR_list)

######################################
##     emotion 12.15 odd ratio, 不可行，放弃
######################################
OR_list = []
emotion_labels = ['Fear', 'Joy', 'Surprise', 'Anger', 'Disgust', 'Sadness']

print("emotion:")
i = 4
total_vaccine_df = vaccine_df[vaccine_df['date'] == date_list[i]]
person_vaccine_df = total_vaccine_df[total_vaccine_df['org'] == 0]
org_vaccine_df = total_vaccine_df[total_vaccine_df['org'] == 1]

total_origin_df = origin_df_list[i]
person_origin_df = total_origin_df[total_origin_df['org'] == 0]
org_origin_df = total_origin_df[total_origin_df['org'] == 1]

for j in range(len(emotion_labels)):
    print(emotion_labels[j])
    tabel_or = [[sum(org_vaccine_df['ekman'] == emotion_labels[j]), sum(org_origin_df['ekman'] == emotion_labels[j]) * 100],
                [sum(person_vaccine_df['ekman'] == emotion_labels[j]), sum(person_origin_df['ekman'] == emotion_labels[j]) * 100]]

    result = sm.stats.Table2x2(tabel_or)
    oddsratio = result.oddsratio
    lcb = result.oddsratio_confint()[0]
    ucb = result.oddsratio_confint()[1]
    pvalue = result.oddsratio_pvalue()
    print("user type: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(oddsratio, lcb, ucb, pvalue))


######################################
##     emotion 12.15 ratio
######################################
OR_list = []
emotion_labels = ['Fear', 'Joy', 'Surprise']

print("emotion:")
date_i = 3
emotion_date =  '2020-12-15'
total_vaccine_df = vaccine_df[vaccine_df['date'] == emotion_date]
person_vaccine_df = total_vaccine_df[total_vaccine_df['org'] == 0]
org_vaccine_df = total_vaccine_df[total_vaccine_df['org'] == 1]

gender_df_list = []
for i in range(2):
    gender_df_list.append(person_vaccine_df[person_vaccine_df['gender'] == i])

age_df_list = []
for i in range(4):
    age_df_list.append(person_vaccine_df[person_vaccine_df['age'] == i])

occ_df_list = []
for i in range(3):
    occ_df_list.append(person_vaccine_df[person_vaccine_df['job_type'] == i])

twi_age_df_list = []
for i in range(3):
    twi_age_df_list.append(person_vaccine_df[person_vaccine_df['twitter_age_class'] == i])

follower_df_list = []
for i in range(3):
    follower_df_list.append(person_vaccine_df[person_vaccine_df['followers_count_class'] == i])

result_list = []
for j in range(len(emotion_labels)):
    print(emotion_labels[j])
    # user type
    a = sum(org_vaccine_df['ekman'] == emotion_labels[j])
    b = org_vaccine_df.shape[0] - a
    c = sum(person_vaccine_df['ekman'] == emotion_labels[j])
    d = person_vaccine_df.shape[0] - c
    tabel_or = [[a, b],
                [c, d]]
    result = sm.stats.Table2x2(tabel_or)
    oddsratio = result.oddsratio
    lcb = result.oddsratio_confint()[0]
    ucb = result.oddsratio_confint()[1]
    pvalue = result.oddsratio_pvalue()
    print("user type: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(oddsratio, lcb, ucb, pvalue))
    print('\t ratio: {}, {}'.format(round(a/(a+b) * 100, 2), round(c/(c+d) * 100, 2)))
    result_list.append([round(c/(c+d) * 100, 2), '1 (ref)', '', '', ''])
    result_list.append([round(a/(a+b) * 100, 2), "{:.2f} ({:.2f}-{:.2f})".format(oddsratio, lcb, ucb),
                       "{:.2f}".format(oddsratio), "{:.2f}".format(lcb), "{:.2f}".format(ucb)])
    result_list.append(['', '', '', '', '' ])

    # gender
    a = sum(gender_df_list[1]['ekman'] == emotion_labels[j])
    b = gender_df_list[1].shape[0] - a
    c = sum(gender_df_list[0]['ekman'] == emotion_labels[j])
    d = gender_df_list[0].shape[0] - c
    tabel_or = [[a, b],
                [c, d]]
    result = sm.stats.Table2x2(tabel_or)
    oddsratio = result.oddsratio
    lcb = result.oddsratio_confint()[0]
    ucb = result.oddsratio_confint()[1]
    pvalue = result.oddsratio_pvalue()
    print("gender: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(oddsratio, lcb, ucb, pvalue))
    print('\t ratio: {}, {}'.format(round(a / (a + b) * 100, 2), round(c / (c + d) * 100, 2)))
    result_list.append([round(c / (c + d) * 100, 2), '1 (ref)', '', '', ''])
    result_list.append([round(a / (a + b) * 100, 2), "{:.2f} ({:.2f}-{:.2f})".format(oddsratio, lcb, ucb),
                        "{:.2f}".format(oddsratio), "{:.2f}".format(lcb), "{:.2f}".format(ucb)])
    result_list.append(['', '', '', '', ''])

    # age
    for i in range(1, 4):
        a = sum(age_df_list[i]['ekman'] == emotion_labels[j])
        b = age_df_list[i].shape[0] - a
        c = sum(age_df_list[0]['ekman'] == emotion_labels[j])
        d = age_df_list[0].shape[0] - c
        tabel_or = [[a, b],
                    [c, d]]
        result = sm.stats.Table2x2(tabel_or)
        oddsratio = result.oddsratio
        lcb = result.oddsratio_confint()[0]
        ucb = result.oddsratio_confint()[1]
        pvalue = result.oddsratio_pvalue()
        print("age {}: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(i, oddsratio, lcb, ucb, pvalue))
        print('\t ratio: {}, {}'.format(round(a / (a + b) * 100, 2), round(c / (c + d) * 100, 2)))
        if i == 1:
            result_list.append([round(c / (c + d) * 100, 2), '1 (ref)', '', '', ''])
        result_list.append([round(a / (a + b) * 100, 2), "{:.2f} ({:.2f}-{:.2f})".format(oddsratio, lcb, ucb),
                            "{:.2f}".format(oddsratio), "{:.2f}".format(lcb), "{:.2f}".format(ucb)])
    result_list.append(['', '', '', '', ''])

    # occupation
    for i in range(1, 3):
        a = sum(occ_df_list[i]['ekman'] == emotion_labels[j])
        b = occ_df_list[i].shape[0] - a
        c = sum(occ_df_list[0]['ekman'] == emotion_labels[j])
        d = occ_df_list[0].shape[0] - c
        tabel_or = [[a, b],
                    [c, d]]
        result = sm.stats.Table2x2(tabel_or)
        oddsratio = result.oddsratio
        lcb = result.oddsratio_confint()[0]
        ucb = result.oddsratio_confint()[1]
        pvalue = result.oddsratio_pvalue()
        print("occupation {}: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(i, oddsratio, lcb, ucb, pvalue))
        print('\t ratio: {}, {}'.format(round(a / (a + b) * 100, 2), round(c / (c + d) * 100, 2)))
        if i == 1:
            result_list.append([round(c / (c + d) * 100, 2), '1 (ref)', '', '', ''])
        result_list.append([round(a / (a + b) * 100, 2), "{:.2f} ({:.2f}-{:.2f})".format(oddsratio, lcb, ucb),
                            "{:.2f}".format(oddsratio), "{:.2f}".format(lcb), "{:.2f}".format(ucb)])
    result_list.append(['', '', '', '', ''])

    # twitter age
    for i in range(1, 3):
        a = sum(twi_age_df_list[i]['ekman'] == emotion_labels[j])
        b = twi_age_df_list[i].shape[0] - a
        c = sum(twi_age_df_list[0]['ekman'] == emotion_labels[j])
        d = twi_age_df_list[0].shape[0] - c
        tabel_or = [[a, b],
                    [c, d]]
        result = sm.stats.Table2x2(tabel_or)
        oddsratio = result.oddsratio
        lcb = result.oddsratio_confint()[0]
        ucb = result.oddsratio_confint()[1]
        pvalue = result.oddsratio_pvalue()
        print("Twitter age {}: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(i, oddsratio, lcb, ucb, pvalue))
        print('\t ratio: {}, {}'.format(round(a / (a + b) * 100, 2), round(c / (c + d) * 100, 2)))
        if i == 1:
            result_list.append([round(c / (c + d) * 100, 2), '1 (ref)', '', '', ''])
        result_list.append([round(a / (a + b) * 100, 2), "{:.2f} ({:.2f}-{:.2f})".format(oddsratio, lcb, ucb),
                            "{:.2f}".format(oddsratio), "{:.2f}".format(lcb), "{:.2f}".format(ucb)])
    result_list.append(['', '', '', '', ''])

    # follower number
    for i in range(1, 3):
        a = sum(follower_df_list[i]['ekman'] == emotion_labels[j])
        b = follower_df_list[i].shape[0] - a
        c = sum(follower_df_list[0]['ekman'] == emotion_labels[j])
        d = follower_df_list[0].shape[0] - c
        tabel_or = [[a, b],
                    [c, d]]
        result = sm.stats.Table2x2(tabel_or)
        oddsratio = result.oddsratio
        lcb = result.oddsratio_confint()[0]
        ucb = result.oddsratio_confint()[1]
        pvalue = result.oddsratio_pvalue()
        print("follower {}: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(i, oddsratio, lcb, ucb, pvalue))
        print('\t ratio: {}, {}'.format(round(a / (a + b) * 100, 2), round(c / (c + d) * 100, 2)))
        if i == 1:
            result_list.append([round(c / (c + d) * 100, 2), '1 (ref)', '', '', ''])
        result_list.append([round(a / (a + b) * 100, 2), "{:.2f} ({:.2f}-{:.2f})".format(oddsratio, lcb, ucb),
                            "{:.2f}".format(oddsratio), "{:.2f}".format(lcb), "{:.2f}".format(ucb)])
    result_list.append(['', '', '', '', ''])
    result_df = pd.DataFrame(data=result_list, columns=['ratio', 'odd ratio', 'oddratio', 'lcb', 'ucb'])
    result_df.to_csv(emotion_date + '_ratio.csv', index=False)

######################################
##     emotion
######################################
OR_list = []
type_list = [[], [], [], [], [], []]
emotion_labels = ['Joy', 'Surprise', 'Fear', 'Anger', 'Disgust', 'Sadness']
for i in range(len(vaccine_date_list)):
    total_df = vaccine_df[vaccine_df['date'] == vaccine_date_list[i]]
    person_df = total_df[total_df['org'] == 0]
    for j in range(len(type_list)):
        type_df = person_df[person_df['ekman'] == emotion_labels[j]]
        type_list[j].append(type_df.shape[0])

type_df_list = []
for j in range(len(type_list)):
    type_df = refine_timeline(vaccine_date_list, type_list[j], vaccine_date_list[0], vaccine_date_list[-1], ['date', 'type'])
    type_df['type'] = Smooth(type_df['type'], 3)
    type_df_list.append(type_df)

print("emotion:")
for i in range(len(date_list)):
    tmp_vaccine_df = vaccine_df[vaccine_df['date'] == date_list[i]]
    tmp_origin_df = origin_df_list[i][origin_df_list[i]['org'] == 0]
    OR_senti_list = []
    for j in range(1, len(type_list)):
        #OR_senti_list.append(sum(tmp_vaccine_df['ekman'] == emotion_labels[j]) / sum(tmp_vaccine_df['ekman'] == emotion_labels[0]) *
        #               sum(tmp_origin_df['ekman'] == emotion_labels[0]) / sum(tmp_origin_df['ekman'] == emotion_labels[j]))

        tabel_or = [[type_df_list[j][type_df_list[j]['date'] == date_list[i]]['type'].values[0],
                     sum(tmp_origin_df['ekman'] == emotion_labels[j]) * 100],
                    [type_df_list[0][type_df_list[0]['date'] == date_list[i]]['type'].values[0],
                     sum(tmp_origin_df['ekman'] == emotion_labels[0]) * 100]]

        result = sm.stats.Table2x2(tabel_or)
        oddsratio = result.oddsratio
        lcb = result.oddsratio_confint()[0]
        ucb = result.oddsratio_confint()[1]
        pvalue = result.oddsratio_pvalue()
        print("{}: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(emotion_labels[j], oddsratio, lcb, ucb, pvalue))
    #OR_list.append(OR_senti_list)

#print(OR_list)

######################################
##     location
######################################
OR_list = []
type_list = [[], [], [], [], []]
country_labels = ['US', 'GB', 'IN', 'CA', 'AU']
for i in range(len(vaccine_date_list)):
    total_df = vaccine_df[vaccine_df['date'] == vaccine_date_list[i]]
    total_df = total_df[total_df['country'] != 'null loc']
    for j in range(len(type_list)):
        type_df = total_df[total_df['country'] == country_labels[j]]
        type_list[j].append(type_df.shape[0])

type_df_list = []
for j in range(len(type_list)):
    type_df = refine_timeline(vaccine_date_list, type_list[j], vaccine_date_list[0], vaccine_date_list[-1], ['date', 'type'])
    type_df['type'] = Smooth(type_df['type'], 3)
    type_df_list.append(type_df)

print("location:")
for i in range(len(date_list)):
    tmp_vaccine_df = vaccine_df[vaccine_df['date'] == date_list[i]]
    tmp_origin_df = origin_df_list[i][origin_df_list[i]['country'] != 'null loc']
    OR_senti_list = []
    for j in range(1, len(type_list)):
        #OR_senti_list.append(sum(tmp_vaccine_df['ekman'] == emotion_labels[j]) / sum(tmp_vaccine_df['ekman'] == emotion_labels[0]) *
        #               sum(tmp_origin_df['ekman'] == emotion_labels[0]) / sum(tmp_origin_df['ekman'] == emotion_labels[j]))

        tabel_or = [[type_df_list[j][type_df_list[j]['date'] == date_list[i]]['type'].values[0],
                     sum(tmp_origin_df['country'] == country_labels[j]) * 100],
                    [type_df_list[0][type_df_list[0]['date'] == date_list[i]]['type'].values[0],
                     sum(tmp_origin_df['country'] == country_labels[0]) * 100]]

        result = sm.stats.Table2x2(tabel_or)
        oddsratio = result.oddsratio
        lcb = result.oddsratio_confint()[0]
        ucb = result.oddsratio_confint()[1]
        pvalue = result.oddsratio_pvalue()
        print("{}: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(country_labels[j], oddsratio, lcb, ucb, pvalue))
    #OR_list.append(OR_senti_list)

#print(OR_list)

######################################
##     verified or not
######################################
OR_list = []
type_list = [[], []]
for i in range(len(vaccine_date_list)):
    total_df = vaccine_df[vaccine_df['date'] == vaccine_date_list[i]]
    person_df = total_df[total_df['org'] == 0]

    for j in range(len(type_list)):
        type_df = person_df[person_df['verified'] == j]
        type_list[j].append(type_df.shape[0])

type_df_list = []
for j in range(len(type_list)):
    type_df = refine_timeline(vaccine_date_list, type_list[j], vaccine_date_list[0], vaccine_date_list[-1], ['date', 'type'])
    type_df['type'] = Smooth(type_df['type'], 2)
    type_df_list.append(type_df)

print("Verified:")
for i in range(len(date_list)):
    tmp_vaccine_df = vaccine_df[vaccine_df['date'] == date_list[i]]
    tmp_vaccine_df = tmp_vaccine_df[tmp_vaccine_df['org'] == 0]
    tmp_origin_df = origin_df_list[i][origin_df_list[i]['org'] == 0]
    OR_job_list = []
    for j in range(1, len(type_list)):
        OR_job_list.append(sum(tmp_vaccine_df['verified'] == j) / sum(tmp_vaccine_df['verified'] == 0) *
                       sum(tmp_origin_df['verified'] == 0) / sum(tmp_origin_df['verified'] == j))

        tabel_or = [[type_df_list[j][type_df_list[j]['date'] == date_list[i]]['type'].values[0],
                     sum(tmp_origin_df['verified'] == j) * 100],
                    [type_df_list[0][type_df_list[0]['date'] == date_list[i]]['type'].values[0],
                     sum(tmp_origin_df['verified'] == 0) * 100]]

        result = sm.stats.Table2x2(tabel_or)
        oddsratio = result.oddsratio
        lcb = result.oddsratio_confint()[0]
        ucb = result.oddsratio_confint()[1]
        pvalue = result.oddsratio_pvalue()
        print("{}: {:.2f} ({:.2f}-{:.2f})  pvalue: {:.3f}".format(j, oddsratio, lcb, ucb, pvalue))
    OR_list.append(OR_job_list)

print(OR_list)

