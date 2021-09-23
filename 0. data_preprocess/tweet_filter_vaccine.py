import pandas as pd
import re, pickle, os
import logging
import numpy as np
import ast

logging.basicConfig(level=logging.INFO, filename="./log/log-filter_vaccine.txt",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s')
print("log is saving into log.txt ...")

dictionary_file = "VaccineDictionary.csv"
src_file_path = "/data/twitter_data/covid_origin_tweets/Sampled_Stream_detail_20201205_1209_origin_covid/"
dst_file_path = "/data/twitter_data/vaccine_covid_origin_tweets/Sampled_Stream_detail_20201205_1209_origin_covid_vaccine/"
VACCINE_TWEET_FILE = dst_file_path + 'vaccine_tweets.csv'

if not os.path.exists(dst_file_path):
    os.mkdir(dst_file_path)

def build_match_word_regex():
    dict_df = pd.DataFrame()
    if os.path.isfile(dictionary_file):
        dict_df = pd.read_csv(dictionary_file, usecols=['name'], encoding="ISO-8859-1")
    else:
        print("dictionary file is not exit.")
        exit()
    print('Generate word_pattern ...\n')
    word_set = set()
    for index, word in dict_df.iterrows():
        if type(word['name']) == type('a'):
            word_name = re.sub(r'[^a-z0-9 ]', ' ', word['name'].lower()).strip()
            word_name = re.sub(' +', ' ', word_name.strip())
            word_set.add(r'\b' + word_name + r'(?![\w-])')

    word_list = list(word_set)
    word_list.sort(key=lambda i: len(i), reverse=True)
    word_pattern = re.compile('|'.join(word_list), re.IGNORECASE)
    return word_pattern

def vaccine_filter():
    word_pattern = build_match_word_regex()
    if os.path.isfile(VACCINE_TWEET_FILE):
        print('VACCINE_TWEET_FILE already exits\n')
    else:
        print('Start filter vaccine from covid...\n')
        file_object = open(VACCINE_TWEET_FILE, 'a', encoding='utf8')
        for root, dirs, files in os.walk(src_file_path):
            for file in files:
                if file.endswith('.csv'):
                    total_count = 0
                    covid_count = 0
                    for line in open(os.path.join(src_file_path, file),'r', encoding='utf-8'):
                        total_count += 1
                        record = ast.literal_eval(line) #json.dumps
                        if 'data' in record:
                            if record['data']['lang'] != 'en':
                                continue
                            if 'text' in record['data']:
                                content = record['data']['text'].lower()
                                if re.search(word_pattern, content):
                                    covid_count += 1
                                    file_object.write("{}".format(line))
                                    continue
                        if 'includes' in record and 'users' in record['includes']:
                            bio = record['includes']['users'][0]['description'].lower()
                            if re.search(word_pattern, bio):
                                covid_count += 1
                                file_object.write("{}".format(line))
                                continue

                    print('{} total_count: {}, vaccine_count: {}, percent: {:.2%}'.format(file, total_count, covid_count, covid_count / total_count))
                    logging.info('{} total_count: {}, vaccine_count: {}, percent: {:.2%}'.format(file, total_count, covid_count, covid_count / total_count))

        #程序结束前关闭文件指针
        if file_object != None:
            file_object.close()
        return

if __name__ == "__main__":
    vaccine_filter()
    print("filter end")
