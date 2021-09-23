import pandas as pd
import re, os
import logging
import ast

logging.basicConfig(level=logging.INFO, filename="./log/log-filter_vaccine_batch.txt",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s')
print("log is saving into log.txt ...")

dictionary_file = "VaccineDictionary.csv"
covid_file_path = "/data/twitter_data/covid_origin_tweets/"
dst_file_path = "/data/twitter_data/vaccine_covid_origin_tweets/"
#covid_file_path = "D:/twitter_data/covid_origin_tweets/"
#dst_file_path = "D:/twitter_data/vaccine_covid_origin_tweets/"
covid_file_list = ['Sampled_Stream_20200602_0603_origin_covid',
                    'Sampled_Stream_20200604_0608_origin_covid',
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
                    'Sampled_Stream_detail_20200910_0917_origin_covid',
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

def vaccine_filter(word_pattern, covid_file_dir, vaccine_file_path):

    print('covid_origin: {}, vaccine:{}'.format(covid_file_dir, vaccine_file_path))
    logging.warning('covid_origin: {}, vaccine:{}'.format(covid_file_dir, vaccine_file_path))

    if not os.path.exists(covid_file_dir):
        print("no file in {}.".format(covid_file_dir))
        logging.warning("no file in {}.".format(covid_file_dir))
        return

    if not os.path.exists(vaccine_file_path):
        os.mkdir(vaccine_file_path)

    VACCINE_TWEET_FILE = vaccine_file_path + 'vaccine_tweets.csv'
    if os.path.isfile(VACCINE_TWEET_FILE):
        print("{} exists.".format(VACCINE_TWEET_FILE))
        logging.warning("{} exists.".format(VACCINE_TWEET_FILE))
    else:
        print('Start filter vaccine from covid...\n')
        file_object = open(VACCINE_TWEET_FILE, 'a', encoding='utf8')
        for root, dirs, files in os.walk(covid_file_dir):
            for file in files:
                if file.endswith('.csv'):
                    total_count = 0
                    covid_count = 0
                    for line in open(os.path.join(covid_file_dir, file),'r', encoding='utf-8'):
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
    word_pattern = build_match_word_regex()

    for file in covid_file_list:
        covid_file_dir = covid_file_path + file
        if os.path.exists(covid_file_dir):
            vaccine_file_path = dst_file_path + file + '_vaccine/'
            # Get vaccine tweets
            vaccine_filter(word_pattern, covid_file_dir, vaccine_file_path)
        else:
            print('{} not exsits.'.format(covid_file_dir))
            logging.warning('{} not exsits.'.format(covid_file_dir))

    print("filter end")
