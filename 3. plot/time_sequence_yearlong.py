#  Copyright (c) 2021.
#  Chunyan Zhang
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from collections import Counter
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import math
from datetime import datetime

import matplotlib.pyplot as plt

#plt.plot(range(5), range(5))
#plt.title('text$ \it{text}$')
#plt.show()


plt.rcParams['font.sans-serif'] = 'Times New Roman'
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size'  : 24}

yearlong = 'yearlong'
dir_name = "D:/twitter_data/vaccine_covid_origin_tweets/"
covid_count_file = "D:/twitter_data/covid_origin_tweets/covid_date_count.csv"
covid_count_df = pd.read_csv(covid_count_file, keep_default_na=False, na_values=['_'], engine='python')
covid_count = covid_count_df.set_index('date').T.to_dict()

ekman_labels = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise']
ekman_labels = ['Joy', 'Surprise', 'Fear', 'Sadness', 'Anger', 'Disgust']
plut_labels = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust', 'Anticipation']
poms_labels = ['Anger', 'Depression', 'Fatigue', 'Vigor', 'Tension', 'Confusion']

date_l = ['2020-07-14', '2020-08-12', '2020-11-09', '2020-12-15', '2021-04-10']

origin_file = 'tweets_analysis_country_state_result.csv'
date_order_file = "tweets_analysis_country_state_result_order_new.csv"

def sort_result():
    df = pd.read_csv(os.path.join(dir_name, yearlong, origin_file), keep_default_na=False, na_values=['_'], engine='python')
    date_list = sorted(set(df['date']))
    new_df = pd.DataFrame()
    for i in range(len(date_list)):
        tmp_df = df[df['date'] == date_list[i]]
        new_df = new_df.append(tmp_df)
    new_df.to_csv(os.path.join(dir_name, yearlong, date_order_file), index=False)

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

def expand_timeline(date_list, value_list, start_time, end_time):
    original_count = len(date_list)
    for date in pd.date_range(start=start_time, end=end_time):
        date_str = date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        for i in range(len(value_list)):
            last_value = value_list[i][original_count - 1]
            value_list[i].append(last_value)

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

def plot_vlines(ymin, ymax):
    plt.vlines(date_l[0], ymin, ymax, colors="#31859c", linestyles="dashed", linewidth=3)
    plt.vlines(date_l[1], ymin, ymax, colors="#7F659F", linestyles="dashed", linewidth=3)
    plt.vlines(date_l[2], ymin, ymax, colors="#799540", linestyles="dashed", linewidth=3)
    plt.vlines(date_l[3], ymin, ymax, colors="#BF9000", linestyles="dashed", linewidth=3)
    plt.vlines(date_l[4], ymin, ymax, colors="#AF3116", linestyles="dashed", linewidth=3)
    plt.ylim((ymin, ymax))

def plot_time_sequence():
    df = pd.read_csv(os.path.join(dir_name, 'tweets_analysis_result_0316.csv'))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list)-1]
    total_list = []
    org_list = []
    male_list = []
    female_list = []
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        total_list.append(total_df.shape[0])

        org_df = total_df[total_df['org'] == 1]
        org_list.append(org_df.shape[0])

        person_df = total_df[total_df['org'] == 0]
        male_df = person_df[person_df['gender'] == 0]
        male_list.append(male_df.shape[0])

        female_df = person_df[person_df['gender'] == 1]
        female_list.append(female_df.shape[0])

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    ax.plot(date_list, Smooth(total_list, smooth_size=3), label='Total')
    ax.plot(date_list, Smooth(org_list, smooth_size=3), label='Organizations')
    ax.plot(date_list, Smooth(male_list, smooth_size=3), label='Males')
    ax.plot(date_list, Smooth(female_list, smooth_size=3), label='Females')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.ylabel('Count', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=20)
    plt.legend(fancybox=True, shadow=False, prop=font2)
    plt.show()


def vif(df, col_i):
    from statsmodels.formula.api import ols
    cols = list(df.columns)
    cols.remove(col_i)
    formula = col_i + '~' + '+'.join(cols)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2)

def plot_time_sequence_total():
    df = pd.read_csv(os.path.join(dir_name, yearlong, date_order_file))
    cases_df = pd.read_csv("owid-covid-data.csv")
    cases_df = cases_df[cases_df["location"] == "World"]

    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list)-1]
    total_list = []
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        total_list.append(total_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

    percent_tmp = total_list[i]
    for date in pd.date_range(start="20210704", end="20210705"):
        date_str = date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        total_list.append(percent_tmp)

    total_list2 = []
    index = 0
    last_percent = total_list[0]
    for date in pd.date_range(start="20200609", end="20210705"):
        date_str = date.strftime("%Y-%m-%d")
        cases = cases_df[cases_df['date'] == date_str]['new_cases_smoothed'].values[0]
        deaths = cases_df[cases_df['date'] == date_str]['new_deaths_smoothed'].values[0]
        vaccined = cases_df[cases_df['date'] == date_str]['new_vaccinations_smoothed'].values[0]
        total_cases = cases_df[cases_df['date'] == date_str]['total_cases'].values[0]
        total_deaths = cases_df[cases_df['date'] == date_str]['total_deaths'].values[0]
        total_vaccinations = cases_df[cases_df['date'] == date_str]['total_vaccinations'].values[0]
        reproduction_rate = cases_df[cases_df['date'] == date_str]['reproduction_rate'].values[0]
        people_vaccinated = cases_df[cases_df['date'] == date_str]['people_vaccinated'].values[0]
        people_fully_vaccinated = cases_df[cases_df['date'] == date_str]['people_fully_vaccinated'].values[0]
        if str(vaccined) == 'nan':
            vaccined = 0
        if str(total_vaccinations) == 'nan':
            total_vaccinations = 0
        if str(people_vaccinated) == 'nan':
            people_vaccinated = 0
        if str(people_fully_vaccinated) == 'nan':
            people_fully_vaccinated = 0
        if str(reproduction_rate) == 'nan':
            reproduction_rate = 1.02

        if date_str != date_list[index]:
            percent_tmp = last_percent

        else:
            percent_tmp = total_list[index]
            last_percent = total_list[index]
            index += 1
        total_list2.append([date_str, percent_tmp, cases, total_cases, deaths, total_deaths, vaccined,
                            total_vaccinations, reproduction_rate, people_vaccinated, people_fully_vaccinated])

    total2_df = pd.DataFrame(data=total_list2, columns=["date", "vaccine_percent", "new_cases", "total_cases",
                                                        "new_deaths", "total_deaths", "new_vaccinations",
                                                        "total_vaccinations", "reproduction_rate", "people_vaccinated",
                                                        "people_fully_vaccinated"])

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    ax.bar(total2_df["date"], Smooth(total2_df["vaccine_percent"], smooth_size=2), color='cornflowerblue', label='vaccine tweets')
    #ax.fill_between(total2_df["date"], y1=Smooth(total2_df["vaccine_percent"], smooth_size=2), y2=0, label='vaccine tweets', alpha=0.5, color='cornflowerblue', linewidth=2)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.set_ylabel('Vaccine Attention (%)', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    #plt.grid(True)
    ax.set_ylim([0, 20])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=90)

    total3_df = total2_df.set_index("date")
    total3_df.pct_change()

    ax2 = ax.twinx()
    ax2.plot(total2_df["date"], total2_df["new_cases"], 'g', label='New Confirmed Cases')
    ax2.plot(total2_df["date"], total2_df["new_deaths"] * 50, 'k', label='New Deaths (/50)')
    ax2.plot(total2_df["date"], total2_df["new_vaccinations"] / 50, 'r', label='New Vaccinations (*50)')
    #ax2.plot(total2_df["date"], (total2_df["new_deaths"] + 0.0012 * total2_df["new_cases"]+ 0.0008 * total2_df["new_vaccinations"])*50, 'c', label='combination (*50)')
    #ax2.plot(total2_df["date"], total2_df["total_cases"] / 200, 'c', label='Total cases (*200)')
    #ax2.plot(total2_df["date"], total2_df["total_deaths"] / 5, 'y', label='Total deaths (*5)')
    #ax2.plot(total2_df["date"], total2_df["total_vaccinations"] / 500, 'm', label='Total vaccinations (*500)')

    ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax2.set_ylabel('COVID-19 Statistics', fontsize=22)
    ax2.set_ylim([0, 1000000])
    #ax2.set_xlabel('Time', fontsize=20)
    plt.setp(ax2.get_yticklabels(), fontsize=20)
    plt.legend(fancybox=True, shadow=False, prop=font2)
    plt.show()
    total2_df[["vaccine_percent", "new_deaths", "new_cases", "new_vaccinations"]].corr()

    lm = ols('vaccine_percent ~ new_deaths + new_cases + new_vaccinations', data=total2_df).fit()
    print(lm.summary())
    print(total2_df["vaccine_percent"].corr(lm.fittedvalues))
    #print(total2_df["vaccine_percent"].corr((lm.params[1] * total2_df["new_deaths"] + lm.params[2] * total2_df["new_cases"] + lm.params[3] * total2_df["new_vaccinations"])))

    y = total2_df['vaccine_percent']
    x = total2_df[["new_cases", "new_deaths", "new_vaccinations", "total_cases", "total_deaths", "total_vaccinations",
                   "reproduction_rate"]]
    x = sm.add_constant(x)  # 若模型中有截距，必须有这一步
    lm = sm.OLS(y, x).fit()  # 构建最小二乘模型并拟合
    print(lm.summary())
    print(total2_df["vaccine_percent"].corr(lm.fittedvalues))

    #total2_df["new_deaths"] = total2_df["new_deaths"].shift(-1, fill_value=total2_df["new_deaths"][len(total2_df["new_deaths"])-1])
    total2_df["new_vaccinations"] = total2_df["new_vaccinations"].shift(-4, fill_value=total2_df["new_vaccinations"][len(total2_df["new_vaccinations"])-1])
    total2_df["reproduction_rate"] = total2_df["reproduction_rate"].shift(-10, fill_value=total2_df["reproduction_rate"][
        len(total2_df["reproduction_rate"]) - 1])

    x_origin = total2_df[["new_cases", "new_deaths", "new_vaccinations", "total_cases", "total_deaths",
                          "total_vaccinations", "people_vaccinated", "reproduction_rate"]]
    x = sm.add_constant(x_origin)  # 若模型中有截距，必须有这一步
    lm = sm.OLS(y, x).fit()
    print(lm.summary())
    print(total2_df["vaccine_percent"].corr(lm.fittedvalues))
    print("VIF:")
    for i in x_origin.columns:
        print(i, '\t', vif(df=x_origin, col_i=i))

    print("多重共线性by combination")
    x_origin = total2_df[["new_cases", "new_deaths", "new_vaccinations", "total_cases", "total_deaths",
                          "total_vaccinations", "people_vaccinated", "reproduction_rate"]]
    x_origin['vaccinate_combination'] = -1.105e-06 * x_origin["new_vaccinations"] \
                                        + 3.856e-08 * x_origin["total_vaccinations"] \
                                        + 6.878e-09 * x_origin["people_vaccinated"]
    """
    x_origin['case_combination'] = -2.138e-06 * x_origin["new_cases"] \
                                   + 4.933e-07 * x_origin["total_cases"]
    x_origin['death_combination'] = 0.0001 * x_origin["new_deaths"] \
                                   -2.18e-05 * x_origin["total_deaths"]
    """
    x_origin = x_origin[["new_cases", "new_deaths", "total_cases", "vaccinate_combination", "reproduction_rate"]]
    #x_origin = x_origin[["death_combination", "case_combination", "vaccinate_combination", "reproduction_rate"]]
    x = sm.add_constant(x_origin)  # 若模型中有截距，必须有这一步
    lm = sm.OLS(y, x).fit()
    print(lm.summary())
    print(total2_df["vaccine_percent"].corr(lm.fittedvalues))
    print("VIF:")
    for i in x_origin.columns:
        print(i, '\t', vif(df=x_origin, col_i=i))

    #lm = ols('vaccine_percent ~ new_deaths + new_cases + new_vaccinations + total_cases + total_deaths + total_vaccinations + reproduction_rate + people_vaccinated', data=total2_df).fit()
    x_origin = total2_df[
        ["new_cases", "new_deaths", "new_vaccinations", "total_cases", "total_vaccinations", "reproduction_rate"]]
    x = sm.add_constant(x_origin)  # 若模型中有截距，必须有这一步
    lm = sm.OLS(y, x).fit()  # 构建最小二乘模型并拟合
    print(lm.summary())
    print('adj r-squared: {}, r-squared: {}'.format(lm.rsquared_adj, lm.rsquared))
    print(total2_df["vaccine_percent"].corr(lm.fittedvalues))
    for i in x_origin.columns:
        print(i, '\t', vif(df=x_origin, col_i=i))

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    result = pd.concat([y, x_origin], axis=1).corr(method='pearson')
    result = x_origin.corr(method='pearson')
    print(result)
    """
    #自变量和因变量两边加ln，仍然没有去除共线性
    covid_type = ["new_cases", "new_deaths", "new_vaccinations", "total_cases", "total_deaths", "total_vaccinations",
                  "people_vaccinated", "reproduction_rate", "vaccine_percent"]

    for type in covid_type:
        total2_df[type + "_ln"] = np.log(total2_df[type])
        for i in range(len(total2_df[type + "_ln"])):
            if total2_df[type + "_ln"][i] < 0:
                total2_df[type + "_ln"][i] = 0

    x_origin = total2_df[["new_cases_ln", "new_deaths_ln", "new_vaccinations_ln", "total_cases_ln", "total_deaths_ln",
                          "total_vaccinations_ln", "people_vaccinated_ln", "reproduction_rate_ln"]]
    y = total2_df['vaccine_percent_ln']
    x = sm.add_constant(x_origin)  # 若模型中有截距，必须有这一步
    lm = sm.OLS(y, x).fit()
    print(lm.summary())
    print(total2_df["vaccine_percent"].corr(lm.fittedvalues))
    print("VIF:")
    for i in x_origin.columns:
        print(i, '\t', vif(df=x_origin, col_i=i))

    #xx = np.matrix(x)
    #VIF_list = [variance_inflation_factor(xx, i) for i in range(xx.shape[1])]
    """
    '''
    #另一种计算方式，结果相同
    y = total2_df['vaccine_percent']
    x = total2_df[["new_cases", "new_deaths", "new_vaccinations", "total_cases", "total_deaths", "total_vaccinations", "reproduction_rate"]]
    x = sm.add_constant(x)  # 若模型中有截距，必须有这一步
    lm = sm.OLS(y, x).fit()  # 构建最小二乘模型并拟合
    print(lm.summary())
    print(total2_df["vaccine_percent"].corr(lm.fittedvalues))
    '''

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.plot(total2_df["date"], total2_df["vaccine_percent"], 'o', label='Vaccine Attention')
    ax.plot(total2_df["date"], lm.fittedvalues, 'r--', label="Estimated Data")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.set_ylim([0, 20])
    ax.set_ylabel('Vaccine Attention (%)', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    #plt.grid(True)
    plt.legend(fancybox=True, shadow=False, prop=font2)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=90)
    plt.show()

"""
    r, p = stats.pearsonr(total2_df["vaccine_percent"], total2_df["total_cases"])
    total2_df["new_deaths"].corr(total2_df["vaccine_percent"].shift(5))

    total2_df["vaccine_percent"].corr((total2_df["new_deaths"] + 0.0012 * total2_df["new_cases"]+ 0.0008 * total2_df["new_vaccinations"]))

    big_r, big_p, big_w = 0, 0, 0
    for w in np.arange(0.1, 0.2, 0.0001):
        r, p = stats.pearsonr(total2_df["vaccine_percent"], total2_df["new_cases"] + w * total2_df["new_vaccinations"])
        if r > big_r:
            big_r, big_p, big_w = r, p, w

    print("big r: {}, p: {}, w: {}".format(big_r, big_p, big_w))
    (total2_df["new_cases"] + 0.1 * total2_df["new_vaccinations"]).corr(total2_df["vaccine_percent"].shift(-1))

    from paper_analysis.paper_anlysis import paper_file
"""

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation.
    Shifted data filled with NaNs

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else:
        return datax.corr(datay.shift(lag))


def plot_time_sequence_relative():
    df = pd.read_csv(os.path.join(dir_name, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list)-1]
    total_list = []
    org_list = []
    male_list = []
    female_list = []
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        total_list.append(total_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

        org_df = total_df[total_df['org'] == 1]
        org_list.append(org_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

        person_df = total_df[total_df['org'] == 0]
        male_df = person_df[person_df['gender'] == 0]
        male_list.append(male_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

        female_df = person_df[person_df['gender'] == 1]
        female_list.append(female_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

    expand_timeline(date_list, [org_list, male_list, female_list], "20210304", "20210307")

    org_df = refine_timeline(date_list, org_list, "20200609", "20210307", ['date', 'org_percent'])
    male_df = refine_timeline(date_list, male_list, "20200609", "20210307", ['date', 'male_percent'])
    female_df = refine_timeline(date_list, female_list, "20200609", "20210307", ['date', 'female_percent'])

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    #ax.plot(date_list, Smooth(total_list, smooth_size=3), label='Total')
    ax.plot(org_df['date'], Smooth(Smooth(org_df['org_percent'], smooth_size=3), 3), label='Organizations')
    ax.plot(male_df['date'], Smooth(Smooth(male_df['male_percent'], smooth_size=3), 3), label='Males')
    ax.plot(female_df['date'], Smooth(Smooth(female_df['female_percent'], smooth_size=3), 3), label='Females')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.ylabel('Percentage (%)', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    plot_vlines(0, 7.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=90)
    # plt.grid(True)
    plt.legend(fancybox=True, shadow=False, prop=font2)
    plt.show()

def plot_time_sequence_OR():
    df = pd.read_csv(os.path.join(dir_name, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list)-1]
    total_list = []
    org_list = []
    gender_list = []
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        org_df = total_df[total_df['org'] == 1]
        person_df = total_df[total_df['org'] == 0]
        org_list.append(org_df.shape[0] / person_df.shape[0] * 89.94 / 10.06)

        male_df = person_df[person_df['gender'] == 0]
        female_df = person_df[person_df['gender'] == 1]
        gender_list.append(female_df.shape[0] / male_df.shape[0] * 52.74 / 47.26)

    expand_timeline(date_list, [org_list, gender_list], "20210304", "20210307")

    org_df = refine_timeline(date_list, org_list, "20200609", "20210307", ['date', 'OR'])
    gender_df = refine_timeline(date_list, gender_list, "20200609", "20210307", ['date', 'OR'])

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    #ax.plot(date_list, Smooth(total_list, smooth_size=3), label='Total')
    ax.plot(org_df['date'], Smooth(Smooth(org_df['OR'], smooth_size=3), 3), label='User Type')
    ax.plot(gender_df['date'], Smooth(Smooth(gender_df['OR'], smooth_size=3), 3), label='Gender')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.ylabel('OR', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    plot_vlines(0, 8)
    plt.hlines(1, "2020-06-09", "2020-03-07", colors='k', linestyles="dashed", linewidth=3)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=90)
    # plt.grid(True)
    plt.legend(fancybox=True, shadow=False, prop=font2)
    plt.show()

def plot_age_sequence():
    df = pd.read_csv(os.path.join(dir_name, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]
    age_list = [[], [], [], []]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        person_df = total_df[total_df['org'] == 0]

        for j in range(4):
            age_df = person_df[person_df['age'] == j]
            age_list[j].append(age_df.shape[0])

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.plot(date_list, Smooth(age_list[0], smooth_size=3), label="<=18")
    ax.plot(date_list, Smooth(age_list[1], smooth_size=3), label="19-29")
    ax.plot(date_list, Smooth(age_list[2], smooth_size=3), label="30-39")
    ax.plot(date_list, Smooth(age_list[3], smooth_size=3), label=">=40")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.ylabel('Count', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=20)
    plt.legend(fancybox=True, shadow=False, prop=font2)
    plt.show()

def plot_age_sequence_relative():
    df = pd.read_csv(os.path.join(dir_name, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]
    age_list = [[], [], [], []]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        person_df = total_df[total_df['org'] == 0]

        for j in range(4):
            age_df = person_df[person_df['age'] == j]
            age_list[j].append(age_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

    expand_timeline(date_list, age_list, "20210304", "20210307")

    age_df_list = []
    for j in range(4):
        age_df_list.append(refine_timeline(date_list, age_list[j], "20200609", "20210307", ['date', 'age']))

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    labels = ["≤18", "19-29", "30-39", "≥40"]
    for j in range(4):
        ax.plot(age_df_list[j]['date'], Smooth(Smooth(age_df_list[j]['age'], smooth_size=3), 3), label=labels[j])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.ylabel('Percentage (%)', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    plot_vlines(0, 4.5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=90)
    plt.legend(fancybox=True, shadow=False, prop=font2)
    plt.show()

def plot_age_sequence_OR():
    df = pd.read_csv(os.path.join(dir_name, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]
    age_list = [[], [], [], []]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        person_df = total_df[total_df['org'] == 0]

        for j in range(4):
            age_df = person_df[person_df['age'] == j]
            if age_df.shape[0] == 0:
                tmp = 1
            else:
                tmp = age_df.shape[0]
            age_list[j].append(tmp)

    age_or_list = [[],[],[]]
    age_per_list = [37.93, 38.42, 11.41, 12.24]
    for i in range(len(date_list)):
        for j in range(3):
            age_or_list[j].append(age_list[j+1][i] / age_list[0][i] * age_per_list[0] / age_per_list[j+1])

    expand_timeline(date_list, age_or_list, "20210304", "20210307")

    age_df_list = []
    for j in range(3):
        age_df_list.append(refine_timeline(date_list, age_or_list[j], "20200609", "20210307", ['date', 'age']))

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    labels = ["19-29", "30-39", "≥40"]
    for j in range(3):
        ax.plot(age_df_list[j]['date'], Smooth(Smooth(age_df_list[j]['age'], smooth_size=5), 3), label=labels[j])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.ylabel('OR', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    plot_vlines(0, 25)
    plt.hlines(1, "2020-06-09", "2020-03-07", colors='k', linestyles="dashed", linewidth=3)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=90)
    plt.legend(fancybox=True, shadow=False, prop=font2)
    plt.show()

def plot_sentiment_sequence():
    df = pd.read_csv(os.path.join(dir_name, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]
    total_list = []
    org_list = []
    person_list = []
    male_list = []
    female_list = []

    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        org_df = total_df[total_df['org'] == 1]
        person_df = total_df[total_df['org'] == 0]
        male_df = person_df[person_df['gender'] == 0]
        female_df = person_df[person_df['gender'] == 1]

        total_list.append(np.mean(total_df['senti-score']))
        org_list.append(np.mean(org_df['senti-score']))
        person_list.append(np.mean(person_df['senti-score']))
        male_list.append(np.mean(male_df['senti-score']))
        if female_df.shape[0] == 0:
            value = female_list[i-1]
        else:
            value = np.mean(female_df['senti-score'])
        female_list.append(value)

    expand_timeline(date_list, [org_list, male_list, female_list], "20210304", "20210307")

    org_df = refine_timeline(date_list, org_list, "20200609", "20210307", ['date', 'senti-score'])
    male_df = refine_timeline(date_list, male_list, "20200609", "20210307", ['date', 'senti-score'])
    female_df = refine_timeline(date_list, female_list, "20200609", "20210307", ['date', 'senti-score'])

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    #ax.plot(date_list, Smooth(total_list, smooth_size=3), label="Total")
    ax.plot(org_df['date'], Smooth(Smooth(org_df['senti-score'], smooth_size=10), 3), label="Organizations")
    ax.plot(male_df['date'], Smooth(Smooth(male_df['senti-score'], smooth_size=10), 3), label="Males")
    ax.plot(female_df['date'], Smooth(Smooth(female_df['senti-score'], smooth_size=10), 3), label="Females")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.ylabel('Sentiment Polarity', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    plot_vlines(-0.2, 0.4)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=90)
    plt.legend(fancybox=True, shadow=False, prop=font2)
    plt.show()

def plot_age_sentiment_sequence():
    df = pd.read_csv(os.path.join(dir_name, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]
    age_list = [[], [], [], []]

    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        person_df = total_df[total_df['org'] == 0]

        for j in range(4):
            age_df = person_df[person_df['age'] == j]
            if age_df.shape[0] == 0:
                value = 0
            else:
                value = np.mean(age_df['senti-score'])
            age_list[j].append(value)

    expand_timeline(date_list, age_list, "20210304", "20210307")
    age_df_list = []
    for j in range(4):
        age_df_list.append(refine_timeline(date_list, age_list[j], "20200609", "20210307", ['date', 'age']))

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    labels = ["≤18", "19-29", "30-39", "≥40"]
    for j in range(4):
        ax.plot(age_df_list[j]['date'], Smooth(Smooth(age_df_list[j]['age'], smooth_size=10), 3), label=labels[j])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.ylabel('Sentiment Polarity', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    plot_vlines(-0.2, 0.4)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=90)
    plt.legend(fancybox=True, shadow=False, prop=font2)
    plt.show()

def plot_emotion_relative_sequence(type = 'ekman'):
    df = pd.read_csv(os.path.join(dir_name, date_order_file))

    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]

    if type == 'ekman':
        labels = ekman_labels
        count_list = [[], [], [], [], [], []]
    elif type == 'plutchik':
        labels = plut_labels
        count_list = [[], [], [], [], [], [], [], []]
    else:
        labels = poms_labels
        count_list = [[], [], [], [], [], []]

    for i in range(len(date_list)):

        total_df = df[df['date'] == date_list[i]]
        emotions = Counter(total_df[type])
        for j in range(len(labels)):
            count_list[j].append(emotions[labels[j]] / covid_count[date_list[i]]['count'] * 100)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    for j in range(len(labels)):
        ax.plot(date_list, Smooth(count_list[j], smooth_size=3), label=labels[j])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.ylabel('Percentage (%)', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=20)
    plt.legend(fancybox=True, shadow=False, prop=font2)
    plt.show()

def plot_emotion_ratio_sequence(type = 'ekman'):
    df = pd.read_csv(os.path.join(dir_name, yearlong, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[1:len(date_list) - 1]

    if type == 'ekman':
        labels = ekman_labels
        count_list = [[], [], [], [], [], []]
    elif type == 'plutchik':
        labels = plut_labels
        count_list = [[], [], [], [], [], [], [], []]
    elif type == 'poms':
        labels = poms_labels
        count_list = [[], [], [], [], [], []]
    else:
        print("type error")
        return

    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        person_df = total_df[total_df['org'] == 0]
        tmp_df = person_df
        #male_df = person_df[person_df['gender'] == 0]

        emotions = Counter(tmp_df[type])
        for j in range(len(labels)):
            #count_list[j].append(emotions[labels[j]] / covid_count[date_list[i]]['count'] * 100)
            count_list[j].append(emotions[labels[j]] / tmp_df.shape[0] * 100)

    expand_timeline(date_list, count_list, "20210704", "20210705")
    df_list = []
    for j in range(len(labels)):
        df_list.append(refine_timeline(date_list, count_list[j], "20200609", "20210705", ['date', 'emotion']))

    # prepare second figure data
    origin_dir_name = "D:/twitter_data/origin_tweets/"
    origin_file_list = ['Sampled_Stream_detail_20200715_0720_origin',
                        'Sampled_Stream_detail_20200811_0815_origin',
                        'Sampled_Stream_detail_20201105_1110_origin',
                        'Sampled_Stream_detail_20201210_1214_origin',
                        'Sampled_Stream_detail_20210410_0416_origin'
                        ]
    emotions_list = []
    for i in range(len(date_l)):
        tmp_list = [date_l[i]]
        tmp_df = pd.read_csv(os.path.join(origin_dir_name, origin_file_list[i], "tweets_analysis_result.csv"))
        tmp_df = tmp_df[type]
        for j in range(len(labels)):
            percent_tmp = int(round(sum(tmp_df == labels[j]) / len(tmp_df) * 100))
            if j == len(labels) - 1:
                percent_tmp = 100 - sum(tmp_list[1:])
            tmp_list.append(percent_tmp)
        emotions_list.append(tmp_list)

    origin_emotions_df = pd.DataFrame(data=emotions_list, columns=['Date'] + labels)

    color_list = ['mediumorchid', 'royalblue', 'forestgreen', 'salmon', 'goldenrod', 'darkcyan']

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    font_size = 22
    for j in range(len(labels)):
        ax[1].plot(df_list[j]['date'], Smooth(Smooth(df_list[j]['emotion'], 3), 3), color=color_list[j],
                   label=labels[j])
    emotions_list = []
    for i in range(len(date_l)):
        tmp_list = [date_l[i]]
        for j in range(len(labels)):
            percent_tmp = int(round(df_list[j][df_list[j]['date'] == date_l[i]]['emotion'].values[0]))
            if j == len(labels) - 1:
                percent_tmp = 100 - sum(tmp_list[1:])
            tmp_list.append(percent_tmp)
        emotions_list.append(tmp_list.copy())
    vaccine_emotions_df = pd.DataFrame(data=emotions_list, columns=['Date'] + labels)

    ax[1].set_ylabel('Percentage (%)', fontsize=font_size)
    ax[1].set_xlabel('Time', fontsize=font_size)
    plot_vlines_ax(ax[1], 0, 60, legend=False)
    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax[1].legend(fancybox=True, shadow=False, prop=font2, loc='center', ncol=3, bbox_to_anchor=(0.48, 1.15))

    plt.subplots_adjust(wspace=0.4, hspace=0.15)

    bar_width = 0.35
    bottom1 = 0
    bottom2 = 0
    for j in range(len(labels)):
        ax[0].bar(vaccine_emotions_df['Date'], vaccine_emotions_df[labels[j]], bottom=bottom1, label=labels[j],
               color=color_list[j], alpha=0.8, width=bar_width)
        ax[0].bar(np.arange(len(origin_emotions_df['Date'])) + bar_width + 0.03, origin_emotions_df[labels[j]],
               bottom=bottom2, color=color_list[j], alpha=0.8, width=bar_width)
        bottom1 += vaccine_emotions_df[labels[j]]
        bottom2 += origin_emotions_df[labels[j]]

    for i in range(len(date_l)):
        text_pos1 = 0
        text_pos2 = 0
        for j in range(len(labels)):
            ax[0].text(i, text_pos1 + vaccine_emotions_df[labels[j]][i] / 2, vaccine_emotions_df[labels[j]][i],
                       ha='center', va='center', fontsize=font_size-2)
            ax[0].text(i + bar_width + 0.03, text_pos2 + origin_emotions_df[labels[j]][i] / 2,
                       origin_emotions_df[labels[j]][i], ha='center', va='center', fontsize=font_size-2)
            text_pos1 += vaccine_emotions_df[labels[j]][i]
            text_pos2 += origin_emotions_df[labels[j]][i]

        ax[0].text(i, -10, 'Vaccine', ha='center', va='center', fontsize=font_size, rotation=45)
        ax[0].text(i + bar_width + 0.03, -10, 'Original', ha='center', va='center', fontsize=font_size, rotation=45)

        ax[0].text(i + bar_width / 2, -20, date_l[i], ha='center', va='center', fontsize=font_size)

    #ax[1].set_title('Emotions', fontsize=font_size)
    ax[0].set_ylabel('Percentage (%)', fontsize=font_size)
    #ax[1].set_xlabel('Date', fontsize=font_size)
    #ax[1].tick_params(axis="x", labelsize=font_size)
    ax[0].tick_params(axis="y", labelsize=font_size)
    ax[0].legend(fancybox=True, shadow=False, prop=font2, loc='center', ncol=3, bbox_to_anchor=(0.48, 1.15))
    plt.show()

def plot_emotion_ratio_sequence_old(type = 'ekman'):
    df = pd.read_csv(os.path.join(dir_name, date_order_file))

    date_list = sorted(set(df['date']))
    date_list = date_list[1:len(date_list) - 1]

    if type == 'ekman':
        labels = ekman_labels
        count_list = [[], [], [], [], [], []]
    elif type == 'plutchik':
        labels = plut_labels
        count_list = [[], [], [], [], [], [], [], []]
    elif type == 'poms':
        labels = poms_labels
        count_list = [[], [], [], [], [], []]
    else:
        print("type error")
        return

    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        person_df = total_df[total_df['org'] == 0]
        tmp_df = person_df
        #male_df = person_df[person_df['gender'] == 0]

        emotions = Counter(tmp_df[type])
        for j in range(len(labels)):
            #count_list[j].append(emotions[labels[j]] / covid_count[date_list[i]]['count'] * 100)
            count_list[j].append(emotions[labels[j]] / tmp_df.shape[0] * 100)

    expand_timeline(date_list, count_list, "20210304", "20210307")
    df_list = []
    for j in range(len(labels)):
        df_list.append(refine_timeline(date_list, count_list[j], "20200609", "20210307", ['date', 'emotion']))

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    for j in range(len(labels)):
        ax.plot(df_list[j]['date'], Smooth(Smooth(df_list[j]['emotion'], 3), 3), label=labels[j])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.ylabel('Percentage (%)', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    plot_vlines(0, 60)
    plt.xlabel('Time', fontsize=22)
    # plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=90)
    plt.legend(fancybox=True, shadow=False, prop=font2)
    plt.show()

    origin_dir_name = "D:/twitter_data/origin_tweets/"
    origin_file_list = ['Sampled_Stream_detail_20200715_0720_origin',
                        'Sampled_Stream_detail_20200811_0815_origin',
                        'Sampled_Stream_detail_20200914_0917_origin',
                        'Sampled_Stream_detail_20201105_1110_origin',
                        'Sampled_Stream_detail_20201210_1214_origin'
                        ]
    emotions_list = []
    for i in range(len(date_l)):
        tmp_list = [date_l[i]]
        tmp_df = pd.read_csv(os.path.join(origin_dir_name, origin_file_list[i], "tweets_analysis_result.csv"))
        tmp_df = tmp_df[tmp_df['org'] == 0][type]
        for j in range(len(labels)):
            percent_tmp = int(round(sum(tmp_df == labels[j]) / len(tmp_df) * 100))
            if j == len(labels) - 1:
                percent_tmp = 100 - sum(tmp_list[1:])
            tmp_list.append(percent_tmp)
        emotions_list.append(tmp_list)

    origin_emotions_df = pd.DataFrame(data=emotions_list, columns=['Date'] + labels)

    font_size = 22
    color_list = ['steelblue', 'forestgreen', 'mediumorchid', 'orangered', 'goldenrod', 'darkcyan']
    emotions_list = []
    for i in range(len(date_l)):
        tmp_list = [date_l[i]]
        for j in range(len(labels)):
            percent_tmp = int(round(df_list[j][df_list[j]['date'] == date_l[i]]['emotion'].values[0]))
            if j == len(labels) - 1:
                percent_tmp = 100 - sum(tmp_list[1:])
            tmp_list.append(percent_tmp)
        emotions_list.append(tmp_list.copy())
    vaccine_emotions_df = pd.DataFrame(data=emotions_list, columns=['Date'] + labels)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    bar_width = 0.35
    bottom1 = 0
    bottom2 = 0
    for j in range(len(labels)):
        ax.bar(vaccine_emotions_df['Date'], vaccine_emotions_df[labels[j]], bottom=bottom1, label=labels[j],
               color=color_list[j], alpha=0.8, width=bar_width)
        ax.bar(np.arange(len(origin_emotions_df['Date'])) + bar_width + 0.03, origin_emotions_df[labels[j]],
               bottom=bottom2, color=color_list[j], alpha=0.8, width=bar_width)
        bottom1 += vaccine_emotions_df[labels[j]]
        bottom2 += origin_emotions_df[labels[j]]

    for i in range(len(date_l)):
        text_pos1 = 0
        text_pos2 = 0
        for j in range(len(labels)):
            ax.text(i, text_pos1 + vaccine_emotions_df[labels[j]][i] / 2, vaccine_emotions_df[labels[j]][i], ha='center', va='center',
                    fontsize=font_size)
            ax.text(i + bar_width + 0.03, text_pos2 + origin_emotions_df[labels[j]][i] / 2, origin_emotions_df[labels[j]][i],
                    ha='center', va='center',
                    fontsize=font_size)
            text_pos1 += vaccine_emotions_df[labels[j]][i]
            text_pos2 += origin_emotions_df[labels[j]][i]

            ax.text(i, -2, 'Vaccine', ha='center', va='center', fontsize=font_size)
            ax.text(i + bar_width + 0.03, -2, 'Original', ha='center', va='center', fontsize=font_size)

    ax.set_title('Emotions', fontsize=font_size)
    ax.set_ylabel('Percentage (%)', fontsize=font_size)
    ax.set_xlabel('Date', fontsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.legend(fancybox=True, shadow=False, prop=font2, loc='center', ncol=6, bbox_to_anchor=(0.48, 1.1))
    plt.show()


def plot_sentiment_sequence_state(states):
    df = pd.read_csv(os.path.join(dir_name, date_order_file))
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]

    for i in range(len(states)):
        tmp_df = df[df['state'] == states[i]]
        total_list = []

        for j in range(len(date_list)):
            total_df = tmp_df[tmp_df['date'] == date_list[j]]
            if len(total_df) == 0:
                total_list.append(0)
            else:
                total_list.append(np.mean(total_df['senti-score']))

        ax.plot(date_list, Smooth(Smooth(total_list, smooth_size=10), 3), label=states[i])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    plt.ylabel('Sentiment', fontsize=22)
    plt.xlabel('Time', fontsize=22)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=90)
    plt.legend(fancybox=True, shadow=False, prop=font2)
    plt.show()

def plot_state_two_in_one(states):
    df = pd.read_csv(os.path.join(dir_name, yearlong, date_order_file))
    fig, ax = plt.subplots(1, 2, figsize=(30, 10))

    #total_df_list = []
    for i in range(len(states)):
        date_list = sorted(set(df['date']))
        date_list = date_list[:len(date_list) - 1]
        tmp_df = df[df['state'] == states[i]]
        #total_list = []
        sentiment_list = []
        for j in range(len(date_list)):
            total_df = tmp_df[tmp_df['date'] == date_list[j]]
            #total_list.append(total_df.shape[0])
            if len(total_df) == 0:
                sentiment_list.append(0)
            else:
                sentiment_list.append(np.mean(total_df['senti-score']))

        expand_timeline(date_list, [sentiment_list], "20210704", "20210705")

        #total_df_list.append(refine_timeline(date_list, total_list, "20200609", "20210307", ['date', 'total']))
        senti_df = refine_timeline(date_list, sentiment_list, "20200609", "20210705", ['date', 'sentiment'])

        ax[0].plot(senti_df['date'], Smooth(Smooth(Smooth(Smooth(senti_df['sentiment'], smooth_size=15), 15), 5), 5), label=states[i])
    plot_vlines_ax(ax[0], -0.3, 0.4)
    ax[0].xaxis.set_major_locator(ticker.MultipleLocator(20))
    #ax[0].set_title('Daily Mean Sentiment Polarity', fontsize=22)
    ax[0].tick_params(axis="x", labelsize=32, labelrotation=90)
    ax[0].tick_params(axis="y", labelsize=32)
    ax[0].set_ylabel('Sentiment Polarity', fontsize=32)
    ax[0].set_xlabel('Time', fontsize=32)

    plt.subplots_adjust(wspace=0.3, hspace=0.15)

    total_sentiment_list = []
    for i in range(len(states)):
        tmp_df = df[df['state'] == states[i]]
        total_sentiment_list.append([states[i],
                                     tmp_df[tmp_df['senti-degree'] < 0].shape[0],
                                     tmp_df[tmp_df['senti-degree'] == 0].shape[0],
                                     tmp_df[tmp_df['senti-degree'] > 0].shape[0]])

    total_sentiment_df = pd.DataFrame(data=total_sentiment_list, columns=['State', 'Negative', 'Neutral', 'Positive'])
    ax[1].bar(total_sentiment_df['State'], total_sentiment_df['Negative'], label='Negative')
    ax[1].bar(total_sentiment_df['State'], total_sentiment_df['Neutral'], bottom=total_sentiment_df['Negative'],
              label='Neutral')
    ax[1].bar(total_sentiment_df['State'], total_sentiment_df['Positive'],
              bottom=total_sentiment_df['Negative'] + total_sentiment_df['Neutral'], label='Positive')


    for i in range(len(states)):
        sum_tmp = total_sentiment_df['Negative'][i] + total_sentiment_df['Neutral'][i] + total_sentiment_df['Positive'][i]
        neg_percent = '{}%'.format(int(total_sentiment_df['Negative'][i] / sum_tmp * 100))
        neu_percent = '{}%'.format(int(total_sentiment_df['Neutral'][i] / sum_tmp * 100))
        pos_percent = '{}%'.format(int(total_sentiment_df['Positive'][i] / sum_tmp * 100))

        ax[1].text(i, total_sentiment_df['Negative'][i] / 2, neg_percent, ha='center', va='center', fontsize=20)
        ax[1].text(i, total_sentiment_df['Negative'][i] + total_sentiment_df['Neutral'][i] / 2, neu_percent, ha='center', va='center', fontsize=20)
        ax[1].text(i, total_sentiment_df['Negative'][i] + total_sentiment_df['Neutral'][i] +
                   total_sentiment_df['Positive'][i] / 2, pos_percent, ha='center', va='center', fontsize=20)

    #ax[1].set_title('Statistics of the Three Sentimental Types', fontsize=22)
    ax[1].set_ylabel('Count', fontsize=22)
    ax[1].set_xlabel('States', fontsize=22)
    ax[1].tick_params(axis="x", labelsize=20)
    ax[1].tick_params(axis="y", labelsize=20)
    ax[1].legend(fancybox=True, shadow=False, prop=font2)
    plt.show()

def plot_sequence_job():
    df = pd.read_csv(os.path.join(dir_name, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]

    job_list = [[], [], []]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        person_df = total_df[total_df['org'] == 0]

        for j in range(3):
            job_df = person_df[person_df['job_type'] == j]
            job_list[j].append(job_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

    expand_timeline(date_list, job_list, "20210304", "20210307")
    job_df_list = []
    for j in range(3):
        job_df_list.append(refine_timeline(date_list, job_list[j], "20200609", "20210307", ['date', 'job']))

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    labels = ["Professional Occ.",
              "Managers, Directors, Senior Officials, Associate Profess., Technical Occ.",
              "Administrative Secretaries, Skilled Trades, Services, Sales, and Other Occ."]
    for j in range(3):
        ax.plot(job_df_list[j]["date"], Smooth(Smooth(job_df_list[j]["job"], 4), 3), label=labels[j])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.set_ylabel('Percentage (%)', fontsize=22)
    plot_vlines(0, 4.5)
    plt.xlabel('Time', fontsize=22)
    #plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=90)
    plt.legend(fancybox=True, shadow=False, prop=font2)
    plt.show()

def plot_sentiment_sequence_job():
    df = pd.read_csv(os.path.join(dir_name, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]

    job_list = [[], [], []]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        person_df = total_df[total_df['org'] == 0]

        for j in range(3):
            job_df = person_df[person_df['job_type'] == j]
            if job_df.shape[0] == 0:
                value = 0
            else:
                value = np.mean(job_df['senti-score'])
            job_list[j].append(value)

    expand_timeline(date_list, job_list, "20210304", "20210307")
    job_df_list = []
    for j in range(3):
        job_df_list.append(refine_timeline(date_list, job_list[j], "20200609", "20210307", ['date', 'job']))
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    labels = ["Professional Occ.",
              "Managers, Directors, Senior Officials, Associate Profess., Technical Occ.",
              "Administrative Secretaries, Skilled Trades, Services, Sales, and Other Occ."]
    for j in range(3):
        ax.plot(job_df_list[j]["date"], Smooth(Smooth(job_df_list[j]["job"], 10), 3), label=labels[j])
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.set_ylabel('Sentiment Polarity', fontsize=22)
    plot_vlines(-0.2, 0.8)
    plt.xlabel('Time', fontsize=22)
    #plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=90)
    plt.legend(fancybox=True, shadow=False, prop=font2)
    plt.show()

def plot_vlines_ax(ax, ymin, ymax, legend=True, loc='best', ncol=1):
    ax.vlines(date_l[0], ymin, ymax, colors="#31859c", linestyles="dashed", linewidth=3)
    ax.vlines(date_l[1], ymin, ymax, colors="#7F659F", linestyles="dashed", linewidth=3)
    ax.vlines(date_l[2], ymin, ymax, colors="#799540", linestyles="dashed", linewidth=3)
    ax.vlines(date_l[3], ymin, ymax, colors="#BF9000", linestyles="dashed", linewidth=3)
    ax.vlines(date_l[4], ymin, ymax, colors="#AF3116", linestyles="dashed", linewidth=3)
    ax.set_ylim(ymin, ymax)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.tick_params(axis="x", labelsize=20, labelrotation=90)
    ax.tick_params(axis="y", labelsize=20)
    if legend:
        ax.legend(fancybox=True, shadow=False, prop=font2, loc=loc, ncol=ncol)

def plot_time_sequence_four_in_one():
    dir_name_tmp = os.path.join(dir_name, yearlong)
    #user type, gender
    df = pd.read_csv(os.path.join(dir_name_tmp, date_order_file), keep_default_na=False, na_values=['_'], engine='python')
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]
    total_list = []
    org_list = []
    person_list = []
    male_list = []
    female_list = []
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        #date_str = datetime.strptime(date_list[i], '%Y/%m/%d').strftime('%Y-%m-%d')
        date_str = date_list[i]
        total_list.append(total_df.shape[0] / covid_count[date_str]['count'] * 100)

        org_df = total_df[total_df['org'] == 1]
        org_list.append(org_df.shape[0] / covid_count[date_str]['count'] * 100)

        person_df = total_df[total_df['org'] == 0]
        person_list.append(person_df.shape[0] / covid_count[date_str]['count'] * 100)

        male_df = person_df[person_df['gender'] == 0]
        male_list.append(male_df.shape[0] / covid_count[date_str]['count'] * 100)

        female_df = person_df[person_df['gender'] == 1]
        female_list.append(female_df.shape[0] / covid_count[date_str]['count'] * 100)

    expand_timeline(date_list, [total_list, org_list, person_list, male_list, female_list], "20210704", "20210705")

    tot_df = refine_timeline(date_list, total_list, "20200609", "20210705", ['date', 'percent'])
    org_df = refine_timeline(date_list, org_list, "20200609", "20210705", ['date', 'percent'])
    ind_df = refine_timeline(date_list, person_list, "20200609", "20210705", ['date', 'percent'])
    male_df = refine_timeline(date_list, male_list, "20200609", "20210705", ['date', 'percent'])
    female_df = refine_timeline(date_list, female_list, "20200609", "20210705", ['date', 'percent'])

    # age
    df = pd.read_csv(os.path.join(dir_name_tmp, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]
    age_list = [[], [], [], []]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        person_df = total_df[total_df['org'] == 0]

        #date_str = datetime.strptime(date_list[i], '%Y/%m/%d').strftime('%Y-%m-%d')
        date_str = date_list[i]
        for j in range(4):
            age_df = person_df[person_df['age'] == j]
            age_list[j].append(age_df.shape[0] / covid_count[date_str]['count'] * 100)

    expand_timeline(date_list, age_list, "20210704", "20210705")
    age_df_list = []
    for j in range(4):
        age_df_list.append(refine_timeline(date_list, age_list[j], "20200609", "20210705", ['date', 'age']))

    # Occupation
    df = pd.read_csv(os.path.join(dir_name_tmp, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]
    job_list = [[], [], []]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        person_df = total_df[total_df['org'] == 0]

        for j in range(3):
            job_df = person_df[person_df['job_type'] == j]
            job_list[j].append(job_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

    expand_timeline(date_list, job_list, "20210704", "20210705")
    job_df_list = []
    for j in range(3):
        job_df_list.append(refine_timeline(date_list, job_list[j], "20200609", "20210705", ['date', 'job']))

    # location
    df = pd.read_csv(os.path.join(dir_name_tmp, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]
    labels = ['Africa', 'Asia', 'Europe', 'North America'] #'Oceania', 'South America'
    continent_list = [[], [], [], []]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        for j in range(len(continent_list)):
            tmp_df = total_df[total_df['continent'] == labels[j]]
            continent_list[j].append(tmp_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

    labels = ['US', 'GB', 'IN']
    country_list = [[], [], []]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        for j in range(len(country_list)):
            tmp_df = total_df[total_df['country'] == labels[j]]
            country_list[j].append(tmp_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

    # twitter age and follower
    df = pd.read_csv(os.path.join(dir_name_tmp, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]
    total_list = []
    twi_age_list = [[], [], []]
    twi_age_th = [0, 5, 10]
    twi_follow_list = [[], [], []]
    twi_follow_th = [0, 500, 5000]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        total_df = total_df[total_df['org'] == 0]
        total_df = total_df[total_df['twitter_age'] > 0]
        total_list.append(total_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

        for j in range(len(twi_age_list)):
            if j < len(twi_age_th) - 1:
                veri_df = total_df[
                    (total_df['twitter_age'] >= twi_age_th[j]) & (total_df['twitter_age'] < twi_age_th[j + 1])]
            else:
                veri_df = total_df[total_df['twitter_age'] >= twi_age_th[j]]

            twi_age_list[j].append(veri_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

        for j in range(len(twi_follow_list)):
            if j < len(twi_follow_list) - 1:
                veri_df = total_df[
                    (total_df['followers_count'] >= twi_follow_th[j]) & (
                                total_df['followers_count'] < twi_follow_th[j + 1])]
            else:
                veri_df = total_df[total_df['followers_count'] >= twi_follow_th[j]]

            twi_follow_list[j].append(veri_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

    date_list_tmp = date_list.copy()
    expand_timeline(date_list, twi_age_list, "20210704", "20210705")
    twi_age_df_list = []
    for j in range(len(twi_age_list)):
        twi_age_df_list.append(refine_timeline(date_list, twi_age_list[j], "20200609", "20210705", ['date', 'value']))

    date_list = date_list_tmp.copy()
    expand_timeline(date_list, twi_follow_list, "20210704", "20210705")
    twi_follow_df_list = []
    for j in range(len(twi_follow_list)):
        twi_follow_df_list.append(
            refine_timeline(date_list, twi_follow_list[j], "20200609", "20210705", ['date', 'value']))

    date_list = date_list_tmp.copy()
    expand_timeline(date_list, continent_list, "20210704", "20210705")
    continent_df_list = []
    for j in range(len(continent_list)):
        continent_df_list.append(
            refine_timeline(date_list, continent_list[j], "20200609", "20210705", ['date', 'value']))

    date_list = date_list_tmp.copy()
    expand_timeline(date_list, country_list, "20210704", "20210705")
    country_df_list = []
    for j in range(len(country_list)):
        country_df_list.append(
            refine_timeline(date_list, country_list[j], "20200609", "20210705", ['date', 'value']))

    fig, ax = plt.subplots(4, 2, figsize=(20, 28))
    ax[0, 0].plot(org_df['date'], Smooth(Smooth(org_df['percent'], smooth_size=3), 3), label='Organizations (' + '$\it{r}$'+ '={:.4f})'.format(tot_df['percent'].corr(org_df['percent'])))
    ax[0, 0].plot(ind_df['date'], Smooth(Smooth(ind_df['percent'], smooth_size=3), 3), label='Individuals (' + '$\it{r}$'+ '={:.4f})'.format(tot_df['percent'].corr(ind_df['percent'])))
    ax[0, 1].plot(male_df['date'], Smooth(Smooth(male_df['percent'], smooth_size=3), 3), label='Males (' + '$\it{r}$'+ '={:.4f})'.format(tot_df['percent'].corr(male_df['percent'])))
    ax[0, 1].plot(female_df['date'], Smooth(Smooth(female_df['percent'], smooth_size=3), 3), label='Females (' + '$\it{r}$'+ '={:.4f})'.format(tot_df['percent'].corr(female_df['percent'])))
    labels = ["≤18", "19-29", "30-39", "≥40"]
    for j in range(4):
        ax[1, 0].plot(age_df_list[j]['date'], Smooth(Smooth(age_df_list[j]['age'], smooth_size=3), 3),
                      label='{} ('.format(labels[j]) + '$\it{r}$'+ '={:.4f})'.format(tot_df['percent'].corr(age_df_list[j]['age'])))
    labels = ["Professional Occ.",
              "Managers, Directors, Senior Officials, Technical Occ.",
              "Secretaries, Skilled Trades, Services, and Other Occ."]
    labels = ["Type 1",
              "Type 2",
              "Type 3"]

    for j in range(3):
        ax[1, 1].plot(job_df_list[j]["date"], Smooth(Smooth(job_df_list[j]["job"], 4), 3),
                      label='{} ('.format(labels[j]) + '$\it{r}$'+ '={:.4f})'.format(tot_df['percent'].corr(job_df_list[j]["job"])))

    labels = ['Africa', 'Asia', 'Europe', 'North America'] #'Australia', , 'South America'
    for j in range(len(continent_list)):
        ax[2, 0].plot(continent_df_list[j]['date'], Smooth(Smooth(continent_df_list[j]['value'], smooth_size=5), 3),
                      label='{} ('.format(labels[j]) + '$\it{r}$' + '={:.4f})'.format(
                          tot_df['percent'].corr(continent_df_list[j]['value'])))

    labels = ['United States', 'United Kingdom', 'India']
    for j in range(len(country_list)):
        ax[2, 1].plot(country_df_list[j]['date'], Smooth(Smooth(country_df_list[j]['value'], smooth_size=5), 3),
                      label='{} ('.format(labels[j]) + '$\it{r}$' + '={:.4f})'.format(
                          tot_df['percent'].corr(country_df_list[j]['value'])))

    labels = ["<5", "5-10", "≥10"]
    for j in range(len(twi_age_list)):
        ax[3, 0].plot(twi_age_df_list[j]['date'], Smooth(Smooth(twi_age_df_list[j]['value'], smooth_size=5), 3),
                      label='{} ('.format(labels[j]) + '$\it{r}$' + '={:.4f})'.format(
                          tot_df['percent'].corr(twi_age_df_list[j]['value'])))

    labels = ["<500", "500-5000", "≥5000"]
    for j in range(len(twi_follow_list)):
        ax[3, 1].plot(twi_follow_df_list[j]['date'], Smooth(Smooth(twi_follow_df_list[j]['value'], smooth_size=5), 3),
                   label='{} ('.format(labels[j]) + '$\it{r}$'+ '={:.4f})'.format(tot_df['percent'].corr(twi_follow_df_list[j]['value'])))

    plot_vlines_ax(ax[0, 0], 0, 12, True, 'upper left')
    plot_vlines_ax(ax[0, 1], 0, 8, True, 'upper left')
    plot_vlines_ax(ax[1, 0], 0, 8, True, 'upper left')
    plot_vlines_ax(ax[1, 1], 0, 8, True, 'upper left')
    plot_vlines_ax(ax[2, 0], 0, 8, True, 'upper left')
    plot_vlines_ax(ax[2, 1], 0, 8, True, 'upper left')
    plot_vlines_ax(ax[3, 0], 0, 8, True, 'upper left')
    plot_vlines_ax(ax[3, 1], 0, 8, True, 'upper left')
    title_fontsize = 26
    label_fontsize =26
    title_pos = (0.5, -0.5)
    ax[0, 0].set_title('(a) User Type', position=title_pos, fontsize=title_fontsize)
    ax[0, 1].set_title('(b) Gender', position=title_pos, fontsize=title_fontsize)
    ax[1, 0].set_title('(c) Age', position=title_pos, fontsize=title_fontsize)
    ax[1, 1].set_title('(d) Occupation', position=title_pos, fontsize=title_fontsize)
    ax[2, 0].set_title('(e) Location – Continent', position=title_pos, fontsize=title_fontsize)
    ax[2, 1].set_title('(f) Location – Country', position=title_pos, fontsize=title_fontsize)
    ax[3, 0].set_title('(g) Twitter Age', position=title_pos, fontsize=title_fontsize)
    ax[3, 1].set_title('(h) Follower Number', position=title_pos, fontsize=title_fontsize)

    for i in range(4):
        for j in range(2):
            ax[i, j].set_ylabel('Vaccine Attention (%)', fontsize=label_fontsize)
            ax[i, j].set_xlabel('Time', fontsize=label_fontsize)
    plt.subplots_adjust(wspace=0.3, hspace=0.15)
    plt.show()

def plot_time_sequence_sentiment_four_in_one():
    dir_name_tmp = os.path.join(dir_name, yearlong)
    #user type, gender
    df = pd.read_csv(os.path.join(dir_name_tmp, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]
    total_list = []
    org_list = []
    person_list = []
    male_list = []
    female_list = []
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        org_df = total_df[total_df['org'] == 1]
        person_df = total_df[total_df['org'] == 0]
        male_df = person_df[person_df['gender'] == 0]
        female_df = person_df[person_df['gender'] == 1]

        total_list.append(np.mean(total_df['senti-score']))
        org_list.append(np.mean(org_df['senti-score']))
        person_list.append(np.mean(person_df['senti-score']))
        male_list.append(np.mean(male_df['senti-score']))
        if female_df.shape[0] == 0:
            value = female_list[i - 1]
        else:
            value = np.mean(female_df['senti-score'])
        female_list.append(value)

    expand_timeline(date_list, [org_list, person_list, male_list, female_list], "20210704", "20210705")
    org_df = refine_timeline(date_list, org_list, "20200609", "20210705", ['date', 'percent'])
    ind_df = refine_timeline(date_list, person_list, "20200609", "20210705", ['date', 'percent'])
    male_df = refine_timeline(date_list, male_list, "20200609", "20210705", ['date', 'percent'])
    female_df = refine_timeline(date_list, female_list, "20200609", "20210705", ['date', 'percent'])

    # age
    df = pd.read_csv(os.path.join(dir_name_tmp, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]
    age_list = [[], [], [], []]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        person_df = total_df[total_df['org'] == 0]

        for j in range(4):
            age_df = person_df[person_df['age'] == j]
            if age_df.shape[0] == 0:
                value = 0
            else:
                value = np.mean(age_df['senti-score'])
            age_list[j].append(value)

    expand_timeline(date_list, age_list, "20210704", "20210705")
    age_df_list = []
    for j in range(4):
        age_df_list.append(refine_timeline(date_list, age_list[j], "20200609", "20210705", ['date', 'age']))

    # Occupation
    df = pd.read_csv(os.path.join(dir_name_tmp, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]
    job_list = [[], [], []]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        person_df = total_df[total_df['org'] == 0]

        for j in range(3):
            job_df = person_df[person_df['job_type'] == j]
            if job_df.shape[0] == 0:
                value = 0
            else:
                value = np.mean(job_df['senti-score'])
            job_list[j].append(value)

    expand_timeline(date_list, job_list, "20210704", "20210705")
    job_df_list = []
    for j in range(3):
        job_df_list.append(refine_timeline(date_list, job_list[j], "20200609", "20210705", ['date', 'job']))

    # location
    df = pd.read_csv(os.path.join(dir_name_tmp, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]
    labels = ['Africa', 'Asia', 'Europe', 'North America']
    continent_list = [[], [], [], []]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        total_df = total_df[total_df['org'] == 0]

        for j in range(len(continent_list)):
            tmp_df = total_df[total_df['continent'] == labels[j]]
            if tmp_df.shape[0] == 0:
                value = 0
            else:
                value = np.mean(tmp_df['senti-score'])
            continent_list[j].append(value)

    labels = ['US', 'GB', 'IN']
    country_list = [[], [], []]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        for j in range(len(country_list)):
            tmp_df = total_df[total_df['country'] == labels[j]]
            if tmp_df.shape[0] == 0:
                value = 0
            else:
                value = np.mean(tmp_df['senti-score'])
            country_list[j].append(value)

    #twitter age and follower
    df = pd.read_csv(os.path.join(dir_name_tmp, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list) - 1]
    total_list = []
    veri_list = [[], []]
    twi_age_list = [[], [], []]
    twi_age_th = [0, 5, 10]
    twi_cnt_list = [[], [], []]
    twi_cnt_th = [0, 1000, 2000]
    twi_follow_list = [[], [], []]
    twi_follow_th = [0, 500, 5000]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        total_df = total_df[total_df['org'] == 0]
        total_df = total_df[total_df['twitter_age'] > 0]
        total_list.append(total_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

        for j in range(len(twi_age_list)):
            if j < len(twi_age_th) - 1:
                veri_df = total_df[
                    (total_df['twitter_age'] >= twi_age_th[j]) & (total_df['twitter_age'] < twi_age_th[j + 1])]
            else:
                veri_df = total_df[total_df['twitter_age'] >= twi_age_th[j]]

            twi_age_list[j].append(np.mean(veri_df['senti-score']))

        for j in range(len(twi_follow_list)):
            if j < len(twi_follow_list) - 1:
                veri_df = total_df[
                    (total_df['followers_count'] >= twi_follow_th[j]) & (
                                total_df['followers_count'] < twi_follow_th[j + 1])]
            else:
                veri_df = total_df[total_df['followers_count'] >= twi_follow_th[j]]

            twi_follow_list[j].append(np.mean(veri_df['senti-score']))

    date_list_tmp = date_list.copy()
    expand_timeline(date_list, twi_age_list, "20210704", "20210705")
    twi_age_df_list = []
    for j in range(len(twi_age_list)):
        twi_age_df_list.append(refine_timeline(date_list, twi_age_list[j], "20200609", "20210705", ['date', 'value']))

    date_list = date_list_tmp.copy()
    expand_timeline(date_list, twi_follow_list, "20210704", "20210705")
    twi_follow_df_list = []
    for j in range(len(twi_follow_list)):
        twi_follow_df_list.append(
            refine_timeline(date_list, twi_follow_list[j], "20200609", "20210705", ['date', 'value']))

    date_list = date_list_tmp.copy()
    expand_timeline(date_list, continent_list, "20210704", "20210705")
    continent_df_list = []
    for j in range(len(continent_list)):
        continent_df_list.append(
            refine_timeline(date_list, continent_list[j], "20200609", "20210705", ['date', 'value']))

    date_list = date_list_tmp.copy()
    expand_timeline(date_list, country_list, "20210704", "20210705")
    country_df_list = []
    for j in range(len(country_list)):
        country_df_list.append(
            refine_timeline(date_list, country_list[j], "20200609", "20210705", ['date', 'value']))

    fig, ax = plt.subplots(4, 2, figsize=(20, 28))
    ax[0, 0].plot(org_df['date'], Smooth(Smooth(org_df['percent'], smooth_size=10), 3), label='Organizations')
    ax[0, 0].plot(ind_df['date'], Smooth(Smooth(ind_df['percent'], smooth_size=10), 3), label='Individuals')
    ax[0, 1].plot(male_df['date'], Smooth(Smooth(male_df['percent'], smooth_size=10), 3), label='Males')
    ax[0, 1].plot(female_df['date'], Smooth(Smooth(female_df['percent'], smooth_size=10), 3), label='Females')
    labels = ["≤18", "19-29", "30-39", "≥40"]
    for j in range(4):
        ax[1, 0].plot(age_df_list[j]['date'], Smooth(Smooth(age_df_list[j]['age'], smooth_size=15), 3), label=labels[j])
    labels = ["Professional Occ.",
              "Managers, Directors, Senior Officials, Technical Occ.",
              "Secretaries, Skilled Trades, Services, and Other Occ."]
    labels = ["Type 1",
              "Type 2",
              "Type 3"]
    for j in range(3):
        ax[1, 1].plot(job_df_list[j]["date"], Smooth(Smooth(job_df_list[j]["job"], 15), 3), label=labels[j])

    #忽略掉Antarctica洲
    labels = ['Africa', 'Asia', 'Europe', 'North America']
    for j in range(len(continent_list)):
        ax[2, 0].plot(continent_df_list[j]['date'], Smooth(Smooth(Smooth(continent_df_list[j]['value'], smooth_size=15), 3), 3), label=labels[j])

    labels = ['United States', 'United Kingdom', 'India']
    for j in range(len(country_list)):
        ax[2, 1].plot(country_df_list[j]['date'], Smooth(Smooth(country_df_list[j]['value'], smooth_size=15), 3), label=labels[j])

    labels = ["<5", "5-10", "≥10"]
    for j in range(len(twi_age_list)):
        ax[3, 0].plot(twi_age_df_list[j]['date'],
                   Smooth(Smooth(Smooth(twi_age_df_list[j]['value'], smooth_size=10), 3), 3),
                   label=labels[j])
    labels = ["<500", "500-5000", "≥5000"]
    for j in range(len(twi_follow_list)):
        ax[3, 1].plot(twi_follow_df_list[j]['date'],
                   Smooth(Smooth(Smooth(twi_follow_df_list[j]['value'], smooth_size=10), 3), 3),
                   label=labels[j])
    title_fontsize = 26
    label_fontsize = 26
    y_min = -0.2
    y_max = 0.4
    plot_vlines_ax(ax[0, 0], y_min, y_max, True, 'upper right')
    plot_vlines_ax(ax[0, 1], y_min, y_max, True, 'upper right')
    plot_vlines_ax(ax[1, 0], y_min, y_max, True, 'upper right')
    plot_vlines_ax(ax[1, 1], y_min, y_max, True, 'upper right')
    plot_vlines_ax(ax[2, 0], y_min, y_max, True, 'upper right', 2)
    plot_vlines_ax(ax[2, 1], y_min, y_max, True, 'upper right')
    plot_vlines_ax(ax[3, 0], y_min, y_max, True, 'upper right')
    plot_vlines_ax(ax[3, 1], y_min, y_max, True, 'upper right')
    title_pos = (0.5, -0.5)
    ax[0, 0].set_title('(a) User Type', position=title_pos, fontsize=title_fontsize)
    ax[0, 1].set_title('(b) Gender', position=title_pos, fontsize=title_fontsize)
    ax[1, 0].set_title('(c) Age', position=title_pos, fontsize=title_fontsize)
    ax[1, 1].set_title('(d) Occupation', position=title_pos, fontsize=title_fontsize)
    ax[2, 0].set_title('(e) Location – Continent', position=title_pos, fontsize=title_fontsize)
    ax[2, 1].set_title('(f) Location – Country', position=title_pos, fontsize=title_fontsize)
    ax[3, 0].set_title('(g) Twitter Age', position=title_pos, fontsize=title_fontsize)
    ax[3, 1].set_title('(h) Follower Number', position=title_pos, fontsize=title_fontsize)

    for i in range(4):
        for j in range(2):
            ax[i, j].set_ylabel('Sentiment Polarity', fontsize=label_fontsize)
            ax[i, j].set_xlabel('Time', fontsize=label_fontsize)
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    #plt.savefig('./vaccine_sentiment_logitude.pdf')
    plt.show()

    #########################
    # five date sentiment
    #########################
    print('user type senti: ')
    for i in range(len(date_l)):
        print('{}: {:.4f}, {:.4f}'.format(i, org_df[org_df['date'] == date_l[i]]['percent'].values[0],
                                            ind_df[ind_df['date'] == date_l[i]]['percent'].values[0]))

    print('gender senti: ')
    for i in range(len(date_l)):
        print('{}: {:.4f}, {:.4f}'.format(i, male_df[male_df['date'] == date_l[i]]['percent'].values[0],
                                          female_df[female_df['date'] == date_l[i]]['percent'].values[0]))

    print('age senti: ')
    for i in range(len(date_l)):
        senti_str = '{}: '.format(i)
        for j in range(4):
            senti_str += '{:.4f}, '.format(age_df_list[j][age_df_list[j]['date'] == date_l[i]]['age'].values[0])
        print(senti_str)

    print('job senti: ')
    for i in range(len(date_l)):
        senti_str = '{}: '.format(i)
        for j in range(3):
            senti_str += '{:.4f}, '.format(job_df_list[j][job_df_list[j]['date'] == date_l[i]]['job'].values[0])
        print(senti_str)

    print('twi age senti: ')
    for i in range(len(date_l)):
        senti_str = '{}: '.format(i)
        for j in range(3):
            senti_str += '{:.4f}, '.format(twi_age_df_list[j][twi_age_df_list[j]['date'] == date_l[i]]['value'].values[0])
        print(senti_str)

    print('follower senti: ')
    for i in range(len(date_l)):
        senti_str = '{}: '.format(i)
        for j in range(3):
            senti_str += '{:.4f}, '.format(twi_follow_df_list[j][twi_follow_df_list[j]['date'] == date_l[i]]['value'].values[0])
        print(senti_str)


def plot_time_sequence_stats():
    df = pd.read_csv(os.path.join(dir_name, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list)-1]
    total_list = []
    veri_list = [[], []]
    twi_age_list = [[], [], []]
    twi_age_th = [0, 5, 10]
    twi_cnt_list = [[], [], []]
    twi_cnt_th = [0, 1000, 2000]
    twi_follow_list = [[], [], []]
    twi_follow_th = [0, 500, 5000]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        total_df = total_df[total_df['org'] == 0]
        total_df = total_df[total_df['twitter_age'] > 0]
        total_list.append(total_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

        for j in range(len(veri_list)):
            veri_df = total_df[total_df['verified'] == j]
            veri_list[j].append(veri_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

        for j in range(len(twi_age_list)):
            if j < len(twi_age_th) - 1:
                veri_df = total_df[(total_df['twitter_age'] >= twi_age_th[j]) & (total_df['twitter_age'] < twi_age_th[j+1])]
            else:
                veri_df = total_df[total_df['twitter_age'] >= twi_age_th[j]]

            twi_age_list[j].append(veri_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

        for j in range(len(twi_cnt_list)):
            if j < len(twi_cnt_list) - 1:
                veri_df = total_df[
                    (total_df['tweet_count'] / total_df['twitter_age'] >= twi_cnt_th[j]) & (total_df['tweet_count'] /
                    total_df['twitter_age'] < twi_cnt_th[j + 1])]
            else:
                veri_df = total_df[total_df['tweet_count'] / total_df['twitter_age'] >= twi_cnt_th[j]]

            twi_cnt_list[j].append(veri_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

        for j in range(len(twi_follow_list)):
            if j < len(twi_follow_list) - 1:
                veri_df = total_df[
                    (total_df['followers_count'] >= twi_follow_th[j]) & (total_df['followers_count'] < twi_follow_th[j + 1])]
            else:
                veri_df = total_df[total_df['followers_count'] >= twi_follow_th[j]]

            twi_follow_list[j].append(veri_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

    date_list_tmp = date_list.copy()
    expand_timeline(date_list, veri_list, "20210304", "20210307")
    veri_df_list = []
    for j in range(len(veri_list)):
        veri_df_list.append(refine_timeline(date_list, veri_list[j], "20200609", "20210307", ['date', 'value']))

    date_list = date_list_tmp.copy()
    expand_timeline(date_list, twi_age_list, "20210304", "20210307")
    twi_age_df_list = []
    for j in range(len(twi_age_list)):
        twi_age_df_list.append(refine_timeline(date_list, twi_age_list[j], "20200609", "20210307", ['date', 'value']))

    date_list = date_list_tmp.copy()
    expand_timeline(date_list, twi_cnt_list, "20210304", "20210307")
    twi_cnt_df_list = []
    for j in range(len(twi_cnt_list)):
        twi_cnt_df_list.append(refine_timeline(date_list, twi_cnt_list[j], "20200609", "20210307", ['date', 'value']))

    date_list = date_list_tmp.copy()
    expand_timeline(date_list, twi_follow_list, "20210304", "20210307")
    twi_follow_df_list = []
    for j in range(len(twi_follow_list)):
        twi_follow_df_list.append(refine_timeline(date_list, twi_follow_list[j], "20200609", "20210307", ['date', 'value']))

    #plot
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    labels = ['Unverified', 'Verified']
    """
    for j in range(len(veri_list)):
        ax[0, 0].plot(veri_df_list[j]['date'], Smooth(Smooth(veri_df_list[j]['value'], smooth_size=3), 3), label=labels[j])
    """
    labels = ["<5 years", "5-10 years", "≥10 years"]
    for j in range(len(twi_age_list)):
        ax[0].plot(twi_age_df_list[j]['date'], Smooth(Smooth(twi_age_df_list[j]['value'], smooth_size=5), 3),
                      label=labels[j])
    """
    labels = ["≤1000", "1000-2000", "≥2000"]
    for j in range(len(twi_cnt_list)):
        ax[1, 0].plot(twi_cnt_df_list[j]['date'], Smooth(Smooth(twi_cnt_df_list[j]['value'], smooth_size=3), 3),
                      label=labels[j])
    """
    labels = ["<500", "500-5000", "≥5000"]
    for j in range(len(twi_follow_list)):
        ax[1].plot(twi_follow_df_list[j]['date'], Smooth(Smooth(twi_follow_df_list[j]['value'], smooth_size=5), 3),
                      label=labels[j])

    #plot_vlines_ax(ax[0, 0], 0, 10)
    plot_vlines_ax(ax[0], 0, 4)
    #plot_vlines_ax(ax[1, 0], 0, 6)
    plot_vlines_ax(ax[1], 0, 5)
    #ax[0, 0].set_title('Users Verified or Not', fontsize=20)
    ax[0].set_title('Twitter Age', fontsize=20)
    #ax[1, 0].set_title('Posted Twitters (per year)', fontsize=20)
    ax[1].set_title('Follower Number', fontsize=20)
    #for i in range(1):
    for j in range(2):
        ax[j].set_ylabel('Percentage (%)', fontsize=22)
        ax[j].set_xlabel('Time', fontsize=22)
    plt.subplots_adjust(wspace=0.3, hspace=0.15)
    plt.show()

def plot_time_sequence_stats_sentiment():
    df = pd.read_csv(os.path.join(dir_name, date_order_file))
    date_list = sorted(set(df['date']))
    date_list = date_list[:len(date_list)-1]
    total_list = []
    veri_list = [[], []]
    twi_age_list = [[], [], []]
    twi_age_th = [0, 5, 10]
    twi_cnt_list = [[], [], []]
    twi_cnt_th = [0, 1000, 2000]
    twi_follow_list = [[], [], []]
    twi_follow_th = [0, 500, 5000]
    for i in range(len(date_list)):
        total_df = df[df['date'] == date_list[i]]
        total_df = total_df[total_df['org'] == 0]
        total_df = total_df[total_df['twitter_age'] > 0]
        total_list.append(total_df.shape[0] / covid_count[date_list[i]]['count'] * 100)

        for j in range(len(twi_age_list)):
            if j < len(twi_age_th) - 1:
                veri_df = total_df[(total_df['twitter_age'] >= twi_age_th[j]) & (total_df['twitter_age'] < twi_age_th[j+1])]
            else:
                veri_df = total_df[total_df['twitter_age'] >= twi_age_th[j]]

            twi_age_list[j].append(np.mean(veri_df['senti-score']))

        for j in range(len(twi_follow_list)):
            if j < len(twi_follow_list) - 1:
                veri_df = total_df[
                    (total_df['followers_count'] >= twi_follow_th[j]) & (total_df['followers_count'] < twi_follow_th[j + 1])]
            else:
                veri_df = total_df[total_df['followers_count'] >= twi_follow_th[j]]

            twi_follow_list[j].append(np.mean(veri_df['senti-score']))

    date_list_tmp = date_list.copy()
    expand_timeline(date_list, twi_age_list, "20210304", "20210307")
    twi_age_df_list = []
    for j in range(len(twi_age_list)):
        twi_age_df_list.append(refine_timeline(date_list, twi_age_list[j], "20200609", "20210307", ['date', 'value']))

    date_list = date_list_tmp.copy()
    expand_timeline(date_list, twi_follow_list, "20210304", "20210307")
    twi_follow_df_list = []
    for j in range(len(twi_follow_list)):
        twi_follow_df_list.append(refine_timeline(date_list, twi_follow_list[j], "20200609", "20210307", ['date', 'value']))

    #plot
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    labels = ["<5 years", "5-10 years", "≥10 years"]
    for j in range(len(twi_age_list)):
        ax[0].plot(twi_age_df_list[j]['date'], Smooth(Smooth(Smooth(twi_age_df_list[j]['value'], smooth_size=10), 3), 3),
                      label=labels[j])

    labels = ["<500", "500-5000", "≥5000"]
    for j in range(len(twi_follow_list)):
        ax[1].plot(twi_follow_df_list[j]['date'], Smooth(Smooth(Smooth(twi_follow_df_list[j]['value'], smooth_size=10), 3), 3),
                      label=labels[j])

    #plot_vlines_ax(ax[0, 0], 0, 10)
    plot_vlines_ax(ax[0], -0.15, 0.3)
    #plot_vlines_ax(ax[1, 0], 0, 6)
    plot_vlines_ax(ax[1], -0.15, 0.3)
    #ax[0, 0].set_title('Users Verified or Not', fontsize=20)
    ax[0].set_title('Twitter Age', fontsize=20)
    #ax[1, 0].set_title('Posted Twitters (per year)', fontsize=20)
    ax[1].set_title('Follower Number', fontsize=20)
    #for i in range(1):
    for j in range(2):
        ax[j].set_ylabel('Sentiment Polarity', fontsize=22)
        ax[j].set_xlabel('Time', fontsize=22)
    plt.subplots_adjust(wspace=0.3, hspace=0.15)
    plt.show()


if __name__ == '__main__':
    #1. 数据日期排序
    #sort_result()
    # 2. 绘制时间序列总图
    #plot_time_sequence_total()
    # 3. 绘制关注度时间序列8张图
    #plot_time_sequence_four_in_one()
    # 4. 绘制sentiment时间序列8张图，以及sentiment时间点值
    #plot_time_sequence_sentiment_four_in_one()
    #5. emotion情感两张子图
    plot_emotion_ratio_sequence()

    #6. 截取第一张图
    plot_state_two_in_one(['CA', 'NY', 'IL', 'FL', 'TX'])

    # plot_sentiment_sequence_state(['CA', 'NY', 'IL', 'TX', 'FL'])
    #plot_time_sequence_stats()
    #plot_time_sequence_stats_sentiment()
    #plot_age_sequence_OR()
    #plot_time_sequence_OR()
    #plot_time_sequence()
    #plot_time_sequence_relative()
    #plot_age_sequence()
    #plot_age_sequence_relative()
    #plot_sentiment_sequence()
    #plot_age_sentiment_sequence()
    #plot_sentiment_sequence_job()
    #plot_sequence_job()
    #plot_emotion_ratio_sequence('ekman')
    #plot_emotion_ratio_sequence('plutchik')
    #plot_emotion_ratio_sequence('poms')
