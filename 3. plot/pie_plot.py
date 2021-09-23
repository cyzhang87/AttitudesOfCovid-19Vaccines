#  Copyright (c) 2021.
#  Chunyan Zhang

import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import Counter

font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size'  : 22}

lang_count = pd.read_csv('../analysis/country_count.csv')
lang_count_head = lang_count.head(5)
lang_count_tail = lang_count.tail(len(lang_count) - 5)
vaccine_lang_df = lang_count_head.append(pd.DataFrame(data=[['Others', sum(lang_count_tail['Count'])]],
                                                      columns=['Country', 'Count']))
explode = [x * 0.05 for x in range(vaccine_lang_df.shape[0])]  # 与labels一一对应，数值越大离中心区越远
plt.figure(figsize=(20, 10))
plt.axes(aspect=1)  # 设置X轴 Y轴比例
# labeldistance标签离中心距离  pctdistance百分百数据离中心区距离 autopct 百分比的格式 shadow阴影
patches,l_text,p_text = plt.pie(x=vaccine_lang_df['Count'], labels=vaccine_lang_df['Country'], explode=explode, autopct='%3.2f%%',
        shadow=True, labeldistance=1.2, startangle=0, pctdistance=0.8, center=(-1, 0))
for t in l_text:
    t.set_size(24)
    t.set_fontname('Times New Roman')
for t in p_text:
    t.set_size(24)
    t.set_fontname('Times New Roman')
# 控制位置：bbox_to_anchor数组中，前者控制左右移动，后者控制上下。ncol控制 图例所列的列数。默认值为1。fancybox 圆边
#plt.legend(loc='center', bbox_to_anchor=(0.5, -0.05), ncol=6, fancybox=True, shadow=False, prop=font1)
figure_path = 'pie_plot.pdf'
#plt.savefig(figure_path)
plt.show()


dir_name = "D:/twitter_data/origin_tweets/"
origin_file_list = ['Sampled_Stream_detail_20200715_0720_origin',
                    'Sampled_Stream_detail_20200811_0815_origin',
                    'Sampled_Stream_detail_20200914_0917_origin',
                    'Sampled_Stream_detail_20201105_1110_origin',
                    'Sampled_Stream_detail_20201210_1214_origin'
                    ]

total_df = pd.read_csv(os.path.join(dir_name, origin_file_list[1], "vaccine_location_country_state_2.csv"), keep_default_na=False, na_values=['_'], engine='python')
country_df = total_df[total_df['country'] != 'null loc']
lang_count = pd.DataFrame(Counter(country_df['country']).most_common(), columns=["Country", "Count"])
lang_count_head = lang_count.head(6)
lang_count_tail = lang_count.tail(len(lang_count) - 6)
origin_lang_pie_df = lang_count_head.append(pd.DataFrame(data=[['Others', sum(lang_count_tail['Count'])]],
                                                      columns=['Country', 'Count']))
origin_lang_df = lang_count.copy()

explode = [x * 0.05 for x in range(origin_lang_pie_df.shape[0])]  # 与labels一一对应，数值越大离中心区越远
plt.figure(figsize=(10, 5))
plt.axes(aspect=1)  # 设置X轴 Y轴比例
# labeldistance标签离中心距离  pctdistance百分百数据离中心区距离 autopct 百分比的格式 shadow阴影
patches,l_text,p_text = plt.pie(x=origin_lang_pie_df['Count'], labels=origin_lang_pie_df['Country'], explode=explode, autopct='%3.2f%%',
        shadow=True, labeldistance=1.1, startangle=0, pctdistance=0.8, center=(-1, 0))
for t in l_text:
    t.set_size(12)
    t.set_fontname('Times New Roman')
for t in p_text:
    t.set_size(12)
    t.set_fontname('Times New Roman')
# 控制位置：bbox_to_anchor数组中，前者控制左右移动，后者控制上下。ncol控制 图例所列的列数。默认值为1。fancybox 圆边
#plt.legend(loc='center', bbox_to_anchor=(0.5, -0.05), ncol=6, fancybox=True, shadow=False, prop=font1)
figure_path = 'pie_plot.pdf'
plt.savefig(figure_path)
plt.show()

country_labels = ['US', 'GB', 'IN', 'AU', 'ER', 'UA']
country_labels = ['US', 'GB', 'IN', 'CA', 'AU']
OR_list = []
vaccine_total = sum(vaccine_lang_df['Count'])
origin_total = sum(origin_lang_df['Count'])
rb = vaccine_lang_df[vaccine_lang_df['Country'] == country_labels[0]]['Count'].values[0] / vaccine_total
rd = origin_lang_df[origin_lang_df['Country'] == country_labels[0]]['Count'].values[0] / origin_total
for i in range(1, vaccine_lang_df.shape[0] - 1):
    ra = vaccine_lang_df[vaccine_lang_df['Country'] == country_labels[i]]['Count'].values[0] / vaccine_total
    rc = origin_lang_df[origin_lang_df['Country'] == country_labels[i]]['Count'].values[0] / origin_total
    OR_list.append(ra / rb * rd / rc)

print(OR_list)