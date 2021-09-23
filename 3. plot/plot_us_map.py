#  Copyright (c) 2021.
#  Chunyan Zhang

import us
import json
import requests
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import geopandas as gpd

dir_name = "D:/twitter_data/vaccine_covid_origin_tweets/"
plt.rcParams['font.sans-serif'] = 'Computer Modern Sans Serif'
yearlong = 'yearlong'

def read_state_sentiment():
    geo_senti_list = []
    result_df = pd.read_csv(os.path.join(dir_name, yearlong, "tweets_analysis_country_state_result_order_new.csv"))
    us_df = result_df[(result_df['country'] == 'US') & (result_df['state'] != 'null loc')]
    states = list(set(us_df['state']))
    for i in range(len(states)):
        tmp_df = result_df[result_df['state'] == states[i]]
        if tmp_df.shape[0] < 10:
            geo_senti_list.append([states[i], 0])
        else:
            geo_senti_list.append([states[i], np.mean(tmp_df['senti-score'])])

    return pd.DataFrame(data=geo_senti_list, columns=['State', 'Sentiment'])

usa_map = None

def read_map_data(map_selection=1):
    global usa_map
    if map_selection == 1:
        usa_map = gpd.read_file('geo_export_cea5f2de-f4e2-4b99-8bdf-554f49a3744f.dbf', encoding='utf-8')
        m = usa_map.state_abbr == "HI"
        usa_map[m] = usa_map[m].set_geometry(usa_map[m].translate(54, 5))
        m = usa_map.state_abbr == "AK"
        usa_map[m] = usa_map[m].set_geometry(usa_map[m].rotate(0.63).scale(0.38, 0.38, 0.38).translate(35, -35))
    else:
        usa_map = gpd.read_file(r'ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp')

        usa_map = usa_map[usa_map['iso_a2'] == 'US']
        usa_map['state_abbr'] = [abbr[3:] for abbr in usa_map['iso_3166_2']]
        m = usa_map.state_abbr == "HI"
        usa_map[m] = usa_map[m].set_geometry(usa_map[m].translate(52, 5))
        m = usa_map.state_abbr == "AK"
        usa_map[m] = usa_map[m].set_geometry(usa_map[m].rotate(0.63).scale(0.22, 0.38, 0.38).translate(-82, -33))

    usa_map['coords'] = usa_map['geometry'].apply(lambda x: x.representative_point().coords[:])
    usa_map['coords'] = [coords[0] for coords in usa_map['coords']]


def plot_us_map_sentiment():
    labels = ['CA', 'NY', 'IL', 'TX', 'FL']
    geo_senti_df = read_state_sentiment()
    state_mean = geo_senti_df.groupby('State').Sentiment.mean()
    usa_map['Sentiment'] = usa_map.state_abbr.apply(lambda x: state_mean[x])
    #usa_map[~usa.state_abbr.isin(['AK', 'HI'])].plot(column='Sentiment', legend=True, cmap='cool', figsize=(12, 4))
    #p = usa_map.plot(column='Sentiment', legend=True, scheme="User_Defined", classification_kwds=dict(bins=[-0.05,0.05]), cmap='tab20', figsize=(12, 4))
    p = usa_map.plot(column='Sentiment', legend=True, cmap='cool', figsize=(12, 4))
    p.set_xlim((-130,-65))
    for idx, row in usa_map.iterrows():
        color = 'dimgrey'
        if row['state_abbr'] in labels:
            color = 'black'
        plt.annotate(s=row['state_abbr'], xy=row['coords'], horizontalalignment='center', color=color)
    plt.savefig("us_sentiment.pdf", bbox_inches='tight')
    plt.show()

def plot_us_map():
    labels = ['CA', 'NY', 'IL', 'TX', 'FL']
    new_data = pd.read_csv('us_state_count.csv')
    #state_has_max_concern = new_data.groupby('State').idxmax(axis=1).droplevel(1)
    #usa['Metric'] = usa.state_abbr.apply(lambda x: str(state_has_max_concern[x]))
    #usa[~usa.state_abbr.isin(['AK', 'HI'])].plot(column='Metric', legend=True, figsize=(12, 4))
    #plt.show()
    #print("end")
    #usa.plot()
    state_mean = new_data.groupby('State').Count.mean()
    usa_map['Count'] = usa_map.state_abbr.apply(lambda x: state_mean[x])
    p = usa_map.plot(column='Count', legend=True, cmap='Reds', figsize=(15, 4))
    p.set_xlim((-130,-65))
    for idx, row in usa_map.iterrows():
        color = 'dimgrey'
        if row['state_abbr'] in labels:
            color = 'black'
        plt.annotate(s=row['state_abbr'], xy=row['coords'], horizontalalignment='center', color=color)
    plt.savefig("us_count.pdf", bbox_inches='tight')
    plt.show()

def plot_us_map_death():
    labels = ['CA', 'NY', 'PA', 'TX', 'FL']
    new_data = pd.read_csv('all-states-history.csv')
    new_data = new_data[new_data['date'] == '2021-03-06'][['state', 'death']]
    state_mean = new_data.groupby('state').death.mean()
    usa_map['Count'] = usa_map.state_abbr.apply(lambda x: state_mean[x])
    p = usa_map.plot(column='Count', legend=True, cmap='Reds', figsize=(15, 4))
    p.set_xlim((-130,-65))
    for idx, row in usa_map.iterrows():
        color = 'dimgrey'
        if row['state_abbr'] in labels:
            color = 'black'
        plt.annotate(s=row['state_abbr'], xy=row['coords'], horizontalalignment='center', color=color)
    #plt.savefig("us_count.pdf", bbox_inches='tight')
    plt.show()

import us
def plot_us_map_death_20210705():
    labels = ['CA', 'NY', 'TX', 'FL', 'PA']
    new_data = pd.read_csv('us-states.csv')
    new_data = new_data[new_data['date'] == '2021-07-05'][['state', 'deaths']]
    state_abbrs = []
    for index, row in new_data.iterrows():
        state_abbrs.append(us.states.lookup(row['state']).abbr)
    new_data['state'] = state_abbrs
    state_mean = new_data.groupby('state').deaths.mean()
    counts = []
    for index, state in usa_map.iterrows():
        counts.append(state_mean[state['state_abbr']])
    usa_map['Count'] = counts
    #usa_map['Count'] = usa_map.state_abbr.apply(lambda x: state_mean[x])
    p = usa_map.plot(column='Count', legend=True, cmap='Reds', figsize=(15, 4))
    p.set_xlim((-130,-65))
    for idx, row in usa_map.iterrows():
        color = 'dimgrey'
        if row['state_abbr'] in labels:
            color = 'black'
        plt.annotate(s=row['state_abbr'], xy=row['coords'], horizontalalignment='center', color=color)
    #plt.savefig("us_count.pdf", bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    read_map_data(2)
    plot_us_map_death_20210705()
    #plot_us_map()
    read_state_sentiment()