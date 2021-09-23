#  Copyright (c) 2021.
#  Chunyan Zhang

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
from datetime import datetime

plt.rcParams['font.sans-serif'] = 'Times New Roman'
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size'  : 20}

paper_file = "./export_vaccine_papers.csv"
#############################################################
##      by day
#############################################################
paper_df = pd.read_csv(paper_file, encoding="ISO-8859-1")
date_count = Counter(paper_df["Entry Date"]).most_common()
date_count_df = pd.DataFrame(data=date_count, columns=['date', 'count'])
date_count_df = date_count_df.sort_values(by='date')
date_count_df.reset_index(drop=True, inplace=True)
dates_dict = date_count_df.set_index('date').to_dict()['count']

accumulated_date_count = []
accumulated_count = 0
for date in pd.date_range(start="20200101", end="20210701"):
    date_int = int(date.strftime("%Y%m%d"))
    if date_int in dates_dict:
        accumulated_count += dates_dict[date_int]
    accumulated_date_count.append([date.strftime("%Y-%m-%d"), accumulated_count])
accumulated_date_count_df = pd.DataFrame(data=accumulated_date_count, columns=['date', 'count'])
#accumulated_date_count_df.to_csv("covid_date_count.csv", index=False)

date_count2 = []
for date in pd.date_range(start="20200101", end="20210701"):
    date_int = int(date.strftime("%Y%m%d"))
    if date_int in dates_dict:
        date_count2.append([date.strftime("%Y-%m-%d"), dates_dict[date_int]])
    else:
        date_count2.append([date.strftime("%Y-%m-%d"), 0])
date_count_df2 = pd.DataFrame(data=date_count2, columns=['date', 'count'])

fig, ax = plt.subplots(1, 1, figsize=(20, 10))

ax.plot(accumulated_date_count_df['date'], accumulated_date_count_df['count'], label='Total numbers')
ax.plot(date_count_df2['date'], date_count_df2['count'], label='Daily numbers')
ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
plt.ylabel('Number', fontsize=22)
plt.xlabel('Time', fontsize=22)
plt.grid(True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(rotation=20)
plt.legend(fancybox=True, shadow=False, prop=font2)
plt.show()


#############################################################
##      by month
#############################################################
month_list = ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06", "2020-07", "2020-08", "2020-09",
              "2020-10", "2020-11", "2020-12", "2021-01", "2021-02", "2021-03", "2021-04", "2021-05", "2021-06",
              "2021-07"]
month_count = []
month_index = 0
tmp_count = 0
accumulated_count = 0
for index, item in date_count_df2.iterrows():
    if datetime.strptime(item['date'], '%Y-%m-%d') < datetime.strptime(month_list[month_index+1], '%Y-%m'):
        tmp_count += item['count']
    else:
        accumulated_count += tmp_count
        month_count.append([month_list[month_index], tmp_count, accumulated_count])
        tmp_count = item['count']
        month_index += 1


month_count_df = pd.DataFrame(data=month_count, columns=['date', 'count', 'accumulated_count'])
font_size = 28
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
plt.grid(True)
plt.bar(month_count_df['date'], month_count_df['count'], width=0.6)
for x1, yy in zip(month_count_df['date'], month_count_df['count']):
    plt.text(x1, yy + 1, str(yy), ha='center', va='bottom', fontsize=font_size, rotation=0)
ax.plot(month_count_df['date'], month_count_df['count'], "r", marker='.', ms=15, linewidth=2, label='Monthly number of papers')
#ax.plot(month_count_df['date'], month_count_df['accumulated_count'], "g", marker='*', ms=15)
plt.ylabel('Number of Papers', fontsize=font_size)
plt.xlabel('Time', fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.xticks(rotation=font_size)
#plt.legend(fancybox=True, shadow=False, prop=font2)
plt.show()


#############################################################
##      Cross-Sectional and attitude papers
#############################################################
paper_df = pd.read_csv(paper_file, encoding="ISO-8859-1")
cspaper_df = pd.DataFrame()
for index, item in paper_df.iterrows():
    if str(item[" Abstract"]) != 'nan':
        if "cross-sectional" in item[" Abstract"].lower() or "longitudinal" in item[" Abstract"].lower():
            cspaper_df = cspaper_df.append(item)
            continue
    if "cross-sectional" in item["Title"].lower() or "longitudinal" in item["Title"].lower():
        cspaper_df = cspaper_df.append(item)

cspaper_df.to_csv("cross_sectional_longitudinal_vaccine_papers.csv", index=False)

attpaper_df = pd.DataFrame()
for index, item in cspaper_df.iterrows():
    if "attitude" in item["Title"].lower():
        attpaper_df = attpaper_df.append(item)

cross_type = []
long_type = []
for index, item in attpaper_df.iterrows():
    if str(item[" Abstract"]) != 'nan':
        if "cross-sectional" in item[" Abstract"].lower() or "cross-sectional" in item["Title"].lower():
            cross_type.append(1)
        else:
            cross_type.append(0)

        if "longitudinal" in item[" Abstract"].lower() or "longitudinal" in item["Title"].lower():
            long_type.append(1)
        else:
            long_type.append(0)

    if str(item[" Abstract"]) == 'nan':
        if "cross-sectional" in item["Title"].lower():
            cross_type.append(1)
        else:
            cross_type.append(0)

        if "longitudinal" in item["Title"].lower():
            long_type.append(1)
        else:
            long_type.append(0)

attpaper_df['cross_type'] = cross_type
attpaper_df['long_type'] = long_type
attpaper_df.to_csv("attitude_vaccine_papers_type.csv", index=False)
