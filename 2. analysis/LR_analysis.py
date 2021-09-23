#  Copyright (c) 2021.
#  Chunyan Zhang
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import seaborn as sns
import numpy as np

dir_name = "D:/twitter_data/vaccine_covid_origin_tweets/"
feature_names = ['gender', 'age']
senti_name = 'senti'

def LineRegression_model():
    df = pd.read_csv(os.path.join(dir_name, "tweets_analysis_country_state_result.csv"), keep_default_na=False,
                     na_values=['_'], engine='python')
    #remove organizations
    df = df[df['org'] == 0]

    date_list = sorted(set(df['date']))
    data_list = []
    for d in range(len(date_list)):
        date_df = df[df['date'] == date_list[d]]
        for i in range(2):
            gender_df = date_df[date_df['gender'] == i]
            for j in range(4):
                age_df = gender_df[gender_df['age'] == i]
                if age_df.shape[0] != 0:
                    data_list.append([i, j, np.mean(age_df['senti'])])
    data_df = pd.DataFrame(data=data_list, columns=['gender', 'age', 'senti'])
    linreg = LinearRegression()
    model = linreg.fit(data_df[feature_names], data_df[senti_name])

    lm = ols('senti ~ gender + age', data=data_df).fit()
    print(lm.summary())

    sns.pairplot(data_df, x_vars=feature_names, y_vars=senti_name, kind="reg")
    plt.show()


if __name__ == '__main__':
    LineRegression_model()

