import ast
import logging
import os
from spacy.lang.en import English
from spacy.pipeline import EntityRuler

logging.basicConfig(level=logging.INFO, filename="./log/log-filter_covid_batch.txt",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s')
print("log is saving into log-filter.txt ...")

org_file_path = "/data/twitter_data/origin_tweets/"
dst_file_path = "/data/twitter_data/covid_origin_tweets/"
#org_file_path = "D:/twitter_data/origin_tweets/"
#dst_file_path = "D:/twitter_data/covid_origin_tweets/"
keyword_file = "twitter_covid19_keywords_all.txt"
org_file_list = ['Filterd_Stream_20200629_origin',
                'Sampled_Stream_20200602_0603_origin',
                'Sampled_Stream_20200604_0608_origin',
                'Sampled_Stream_detail_20200608_0614_origin',
                'Sampled_Stream_detail_20200614_0619_origin',
                'Sampled_Stream_detail_20200619_0630_origin',
                'Sampled_Stream_detail_20200715_0720_origin',
                'Sampled_Stream_detail_20200720_0726_origin',
                'Sampled_Stream_detail_20200726_0731_origin',
                'Sampled_Stream_detail_20200731_0804_origin',
                'Sampled_Stream_detail_20200804_0807_origin',
                'Sampled_Stream_detail_20200807_0811_origin',
                'Sampled_Stream_detail_20200811_0815_origin',
                'Sampled_Stream_detail_20200816_0821_origin',
                'Sampled_Stream_detail_20200821_0824_origin',
                'Sampled_Stream_detail_20200825_0828_origin',
                'Sampled_Stream_detail_20200828_0830_origin',
                'Sampled_Stream_detail_20200830_0904_origin',
                'Sampled_Stream_detail_20200904_0908_origin',
                'Sampled_Stream_detail_20200910_0914_origin',
                'Sampled_Stream_detail_20200910_0917_origin',
                'Sampled_Stream_detail_20200914_0917_origin',
                'Sampled_Stream_detail_20200917_0921_origin',
                'Sampled_Stream_detail_20200921_0924_origin',
                'Sampled_Stream_detail_20200924_0928_origin',
                'Sampled_Stream_detail_20200928_1002_origin',
                'Sampled_Stream_detail_20201002_1006_origin',
                'Sampled_Stream_detail_20201006_1009_origin',
                'Sampled_Stream_detail_20201009_1012_origin',
                'Sampled_Stream_detail_20201017_1020_origin',
                'Sampled_Stream_detail_20201020_1023_origin',
                'Sampled_Stream_detail_20201023_1031_origin',
                'Sampled_Stream_detail_20201031_1104_origin',
                'Sampled_Stream_detail_20201105_1110_origin',
                'Sampled_Stream_detail_20201110_1119_origin',
                'Sampled_Stream_detail_20201119_1124_origin',
                'Sampled_Stream_detail_20201124_1129_origin',
                'Sampled_Stream_detail_20201129_1205_origin',
                'Sampled_Stream_detail_20201205_1209_origin',
                'Sampled_Stream_detail_20201210_1214_origin',
                'Sampled_Stream_detail_20201214_1218_origin',
                'Sampled_Stream_detail_20201218_1224_origin',
                'Sampled_Stream_detail_20201224_1229_origin',
                'Sampled_Stream_detail_20201229_0103_origin',
                'Sampled_Stream_detail_20210103_0108_origin',
                'Sampled_Stream_detail_20210108_0112_origin',
                'Sampled_Stream_detail_20210112_0122_origin',
                'Sampled_Stream_detail_20210122_0128_origin',
                'Sampled_Stream_detail_20210128_0202_origin',
                'Sampled_Stream_detail_20210202_0205_origin',
                'Sampled_Stream_detail_20210206_0208_origin',
                'Sampled_Stream_detail_20210209_0213_origin',
                'Sampled_Stream_detail_20210213_0216_origin',
                'Sampled_Stream_detail_20210216_0219_origin',
                'Sampled_Stream_detail_20210219_0222_origin',
                'Sampled_Stream_detail_20210222_0225_origin',
                'Sampled_Stream_detail_20210225_0228_origin',
                'Sampled_Stream_detail_20210228_0304_origin',
                 # update new data as of 2021.6
                 'Sampled_Stream_detail_20210304_0308_origin',
                 'Sampled_Stream_detail_20210308_0312_origin',
                 'Sampled_Stream_detail_20210312_0319_origin',
                 'Sampled_Stream_detail_20210319_0323_origin',
                 'Sampled_Stream_detail_20210323_0326_origin',
                 'Sampled_Stream_detail_20210326_0329_origin',
                 'Sampled_Stream_detail_20210329_0402_origin',
                 'Sampled_Stream_detail_20210402_0406_origin',
                 'Sampled_Stream_detail_20210406_0410_origin',
                 'Sampled_Stream_detail_20210410_0416_origin',
                 'Sampled_Stream_detail_20210416_0420_origin',
                 'Sampled_Stream_detail_20210420_0423_origin',
                 'Sampled_Stream_detail_20210423_0427_origin',
                 'Sampled_Stream_detail_20210427_0501_origin',
                 'Sampled_Stream_detail_20210501_0506_origin',
                 'Sampled_Stream_detail_20210506_0512_origin',
                 'Sampled_Stream_detail_20210512_0517_origin',
                 'Sampled_Stream_detail_20210517_0522_origin',
                 'Sampled_Stream_detail_20210522_0527_origin',
                 'Sampled_Stream_detail_20210527_0530_origin',
                 'Sampled_Stream_detail_20210530_0603_origin',
                 'Sampled_Stream_detail_20210607_0616_origin',
                 'Sampled_Stream_detail_20210616_0620_origin',
                 'Sampled_Stream_detail_20210620_0624_origin',
                 'Sampled_Stream_detail_20210624_0629_origin',
                 'Sampled_Stream_detail_20210629_0703_origin',
                 ]

keywords_patterns = []
pattern_template = {"label": "covid", "pattern": "covid-19"}
nlp = English()
ruler = EntityRuler(nlp)
for word in open(keyword_file, 'r', encoding='utf-8'):
    pos = word.find('\n')
    if pos == -1:
        pos = len(word)
    keyword = word[:pos].lower()
    pattern_template = {"label": "covid"}
    pattern_template["pattern"] = keyword
    keywords_patterns.append(pattern_template)

ruler.add_patterns(keywords_patterns)
nlp.add_pipe(ruler)

def has_covid_pattern_in_content(content):
    doc = nlp(content)
    if len(doc.ents):
        return True
    return False
    #print([(ent.text, ent.label_) for ent in doc.ents])

def covid_filter(org_file_dir, covid_file_path):
    print('origin: {}, covid_origin:{}'.format(org_file_dir, covid_file_path))
    logging.warning('origin: {}, covid_origin:{}'.format(org_file_dir, covid_file_path))

    if not os.path.exists(org_file_dir):
        print("no file in {}.".format(org_file_dir))
        logging.warning("no file in {}.".format(org_file_dir))
        return

    if not os.path.exists(covid_file_path):
        os.mkdir(covid_file_path)

    COVID_TWEET_FILE = covid_file_path + 'covid_tweets.csv'
    if os.path.isfile(COVID_TWEET_FILE):
        print("{} exists.".format(COVID_TWEET_FILE))
        logging.warning("{} exists.".format(COVID_TWEET_FILE))
    else:
        print('Start filter covid from origin...\n')
        file_object = open(COVID_TWEET_FILE, 'a', encoding='utf8')
        for root, dirs, files in os.walk(org_file_dir):
            for file in files:
                if file.endswith('.csv'):
                    total_count = 0
                    covid_count = 0
                    for line in open(os.path.join(org_file_dir, file), 'r', encoding='utf-8'):
                        total_count += 1
                        record = ast.literal_eval(line) #json.dumps
                        if 'data' in record:
                            if 'text' in record['data']:
                                content = record['data']['text']
                                content = content.lower()
                                if has_covid_pattern_in_content(content):
                                    covid_count += 1
                                    file_object.write("{}".format(line))
                                    continue

                    print('{} total_count: {}, covid_count: {}, percent: {:.2%}'.format(file, total_count, covid_count, covid_count / total_count))
                    logging.info('{} total_count: {}, covid_count: {}, percent: {:.2%}'.format(file, total_count, covid_count, covid_count / total_count))

        #程序结束前关闭文件指针
        if file_object != None:
            file_object.close()
        return

if __name__ == "__main__":
    for file in org_file_list:
        org_file_dir = org_file_path + file
        if os.path.exists(org_file_dir):
            covid_file_path = dst_file_path + file + '_covid/'
            # Get covid tweets
            covid_filter(org_file_dir, covid_file_path)
        else:
            print('{} not exsits.'.format(org_file_dir))
            logging.warning('{} not exsits.'.format(org_file_dir))

    print("filter end")
