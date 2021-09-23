#  Copyright (c) 2021.
#  Chunyan Zhang

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

country_map = gpd.read_file(r'ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
country_map['coords'] = country_map['geometry'].apply(lambda x: x.representative_point().coords[:])
country_map['coords'] = [coords[0] for coords in country_map['coords']]
#new_data = pd.read_csv('country_count.csv', keep_default_na=False, na_values=['_'])
new_data = pd.read_csv('country_count_yearlong.csv', keep_default_na=False, na_values=['_'])
countries_list = list(set(new_data['Country']))

def read_country_sentiment():
    geo_senti_list = []
    #result_df = pd.read_csv(os.path.join(dir_name, "tweets_analysis_country_state_result.csv"))
    result_df = pd.read_csv("tweets_analysis_country_state_result_order_new.csv", keep_default_na=False, na_values=['_'], engine='python')
    countries = list(set(result_df['country']))
    for i in range(len(countries)):
        tmp_df = result_df[result_df['country'] == countries[i]]
        if tmp_df.shape[0] < 10:
            geo_senti_list.append([countries[i], 0])
        else:
            geo_senti_list.append([countries[i], np.mean(tmp_df['senti-score'])])

    return pd.DataFrame(data=geo_senti_list, columns=['Country', 'Sentiment'])

def get_value(x, data):
    if x in countries_list:
        return data[x]
    else:
        return 0

def plot_world_map():
    #labels = ['US', 'GB', 'IN', 'AU', 'ER', 'UA']
    labels = ['US', 'IN', 'GB', 'CA', 'AU']
    percents = ['57.53%', '10.63%', '10.14%', '1.71%', '1.65%']
    country_data = new_data.groupby('Country').Count.mean()
    country_map['Count'] = country_map.ISO_A2.apply(lambda x: get_value(x, country_data))
    p = country_map.plot(column='Count', legend=True, edgecolor='white', linewidth = .5, cmap='Reds')
    #p.set_xlim((-130,-65))
    #country_map.plot(figsize=(15,12))
    for idx, row in country_map.iterrows():
        if row['ISO_A2'] in labels:
            i = 0
            for i in range(len(labels)):
                if row['ISO_A2'] == labels[i]:
                    break
            plt.annotate(s=row['ISO_A2'] + ':{}'.format(percents[i]), xy=row['coords'], horizontalalignment='center')
    #plt.savefig("country_count.pdf", bbox_inches='tight')
    plt.show()

def plot_sentiment_map():
    labels = ['BF', 'HR', 'BW', 'WF', 'CO', 'KN', 'RW']
    labels = ['BF', 'CI', 'WF', 'BS', 'SI', 'CO', 'KR', 'BW', 'KN', 'RW']
    labels = ['US', 'GB', 'IN', 'CA', 'AU']
    geo_senti_df = read_country_sentiment()
    senti_data = geo_senti_df.groupby('Country').Sentiment.mean()
    country_map['Sentiment'] = country_map.ISO_A2.apply(lambda x: get_value(x, senti_data))
    # usa_map[~usa.state_abbr.isin(['AK', 'HI'])].plot(column='Sentiment', legend=True, cmap='cool', figsize=(12, 4))
    p = country_map.plot(column='Sentiment', legend=True, cmap='cool')
    for idx, row in country_map.iterrows():
        if row['ISO_A2'] in labels:
            plt.annotate(s=row['ISO_A2'] + ':{:.4f}'.format(row['Sentiment']), xy=row['coords'], horizontalalignment='center')
    plt.show()

if __name__ == '__main__':
    plot_sentiment_map()
    plot_world_map()

"""
df = gpd.read_file(r'ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp')

states = df[df['iso_a2']=='US']
m = states.iso_3166_2 == "US-HI"
states[m] = states[m].set_geometry(states[m].translate(54))
m = states.iso_3166_2 == "US-AK"
states[m] = states[m].set_geometry(states[m].rotate(0.63).scale(0.38, 0.38, 0.38).translate(-60, -35))
p = states.plot(figsize=(15,12))
p.set_xlim((-140,-60))
plt.show()
"""