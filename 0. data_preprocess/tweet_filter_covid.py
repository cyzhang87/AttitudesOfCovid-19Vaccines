import ast
import logging
import re, pickle, os
import pandas as pd

logging.basicConfig(level=logging.INFO, filename="./log/log-filter_covid.txt",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s')
print("log is saving into log-filter.txt ...")


src_file_path = "/data/twitter_data/origin_tweets/Sampled_Stream_detail_20201210_1214_origin/"
dst_file_path = "/data/twitter_data/covid_origin_tweets/Sampled_Stream_detail_20201210_1214_origin_covid/"
keyword_file = "twitter_covid19_keywords_all.txt"
covid_dictionary_file = "twitter_covid19_keywords_all.csv"
COVID_TWEET_FILE = dst_file_path + 'covid_tweets.csv'


keywords_list = []

if not os.path.exists(dst_file_path):
    os.mkdir(dst_file_path)

from spacy.lang.en import English
from spacy.pipeline import EntityRuler

keywords_patterns = []
pattern_template = {"label": "covid", "pattern": "covid-19"}
#nlp = spacy.load("en_core_web_sm")
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

def covid_filter():
    if not os.path.exists(src_file_path):
        print("no file in {}.".format(src_file_path))
        return

    if os.path.isfile(COVID_TWEET_FILE):
        print('COVID_TWEET_FILE already exits\n')
    else:
        print('Start filter covid from origin...\n')
        file_object = open(COVID_TWEET_FILE, 'a', encoding='utf8')
        for root, dirs, files in os.walk(src_file_path):
            for file in files:
                if file.endswith('.csv'):
                    total_count = 0
                    covid_count = 0
                    for line in open(os.path.join(src_file_path, file), 'r', encoding='utf-8'):
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
    covid_filter()
    #generate_covid_label()
    print("filter end")

"""
{'data':
     {
         'source': 'Twitter for Android',
         'created_at': '2020-12-19T01:44:49.000Z',
         'entities':
             {'mentions':
                  [{'start': 0, 'end': 10, 'username': 'ttaekilla'}]},
         'text': '@ttaekilla cu do murilo $3,50', 'lang': 'ro',
         'id': '1340110809347842048', 'conversation_id': '1340105202897195009',
         'referenced_tweets': [{'type': 'replied_to', 'id': '1340105202897195009'}], 'possibly_sensitive': False,
         'public_metrics': {'retweet_count': 0, 'reply_count': 0, 'like_count': 0, 'quote_count': 0},
         'in_reply_to_user_id': '1183149832720080896', 'author_id': '1231085462523125761'},
    'includes': {'users': [{'id': '1231085462523125761',
                            'verified': False,
                            'public_metrics': {'followers_count': 11, 'following_count': 159, 'tweet_count': 207, 'listed_count': 0},
                            'url': '', 'profile_image_url': 'https://pbs.twimg.com/profile_images/1339970287832580096/8zGAgV4B_normal.jpg',
                            'username': 'musanca', 'description': 'gordin mais lindo que existe', 'created_at': '2020-02-22T05:17:12.000Z',
                            'name': 'muu☠ #DropGameOver', 'protected': False}, {'pinned_tweet_id': '1324182155203956739', 'id': '1183149832720080896',
                                                                                'verified': False,
                                                                                'public_metrics': {'followers_count': 473, 'following_count': 529, 'tweet_count': 14357, 'listed_count': 2},
                                                                                'url': 'https://t.co/AHvotXBklU', 'profile_image_url': 'https://pbs.twimg.com/profile_images/1337783836445433857/2tWswfwI_normal.jpg',
                                                                                'username': 'ttaekilla', 'description': 'legalize it🌿 (ela/dela)', 'created_at': '2019-10-12T22:38:09.000Z',
                                                                                'location': 'na merda', 'entities': {'url': {'urls': [{'start': 0, 'end': 23, 'url': 'https://t.co/AHvotXBklU',
                                                                                                                                       'expanded_url': 'http://armyblink.xn--8ci/', 'display_url': 'armyblink.✰'}]}}, 'name': 'ana', 'protected': False}],
                 'tweets': [{'entities': {'urls': [{'start': 37, 'end': 60, 'url': 'https://t.co/pTQFOL1P98',
                                                    'expanded_url': 'https://twitter.com/ttaekilla/status/1340105202897195009/photo/1',
                                                    'display_url': 'pic.twitter.com/pTQFOL1P98'}, {'start': 37, 'end': 60, 'url': 'https://t.co/pTQFOL1P98',
                                                                                                   'expanded_url': 'https://twitter.com/ttaekilla/status/1340105202897195009/photo/1',
                                                                                                   'display_url': 'pic.twitter.com/pTQFOL1P98'}, {'start': 37, 'end': 60, 'url': 'https://t.co/pTQFOL1P98',
                                                                                                                                                  'expanded_url': 'https://twitter.com/ttaekilla/status/1340105202897195009/photo/1', 'display_url': 'pic.twitter.com/pTQFOL1P98'},
                                                   {'start': 37, 'end': 60, 'url': 'https://t.co/pTQFOL1P98', 'expanded_url': 'https://twitter.com/ttaekilla/status/1340105202897195009/photo/1', 'display_url': 'pic.twitter.com/pTQFOL1P98'}]},
                             'source': 'Twitter for Android', 'created_at': '2020-12-19T01:22:32.000Z',
                             'text': 'lindas assinaturas saí com depressão https://t.co/pTQFOL1P98', 'lang': 'pt', 'attachments': {'media_keys': ['3_1340105186518429699', '3_1340105190918270976', '3_1340105194928021507', '3_1340105199709532161']}, 'id': '1340105202897195009', 'conversation_id': '1340105202897195009', 'possibly_sensitive': False, 'public_metrics': {'retweet_count': 0, 'reply_count': 1, 'like_count': 2, 'quote_count': 0},
                             'author_id': '1183149832720080896'}]}}
"""