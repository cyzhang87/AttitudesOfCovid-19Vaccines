import pandas as pd
import re, os
import matplotlib.pyplot as plt
from collections import Counter
import ast
import logging

logging.basicConfig(level=logging.INFO, filename="./log/log-filter_origin.txt",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s')
print("log is saving into log-filter_origin.txt ...")
src_file_path = "/data/twitter_data/raw_tweets/Sampled_Stream_detail_20201210_1214/"
dst_file_path = "/data/twitter_data/origin_tweets/Sampled_Stream_detail_20201210_1214_origin/"

records_per_file = 10000
count = 0
file_object = None
file_name = None

logging.info("begin filtering orginal tweets in " + src_file_path)

if not os.path.exists(dst_file_path):
    os.mkdir(dst_file_path)

def save_data(item, filename):
    global file_object, count, file_name
    if file_object is None:
        file_name = filename.replace("twitter_sample", "twitter_sample_origin")
        count += 1
        file_object = open("{}{}".format(dst_file_path, file_name), 'a', encoding='utf-8')
        file_object.write("{}".format(item))
        return
    if count == records_per_file:
        file_object.close()
        count = 1
        file_name = filename.replace("twitter_sample", "twitter_sample_origin")
        file_object = open("{}{}".format(dst_file_path, file_name), 'a', encoding='utf-8')
        file_object.write("{}".format(item))
        logging.warning("save " + str(records_per_file) + " items to file.")
    else:
        count += 1
        file_object.write("{}".format(item))

def get_origin_tweets():
    print('Start reading tweets from twitter-sample files...\n')
    if not os.path.exists(src_file_path):
        print("no file in {}.".format(src_file_path))
        return None

    for root, dirs, files in os.walk(src_file_path):
        for file in files:
            if file.endswith('.csv'):
                logging.warning(file)
                total_count = 0
                eng_count = 0
                origin_count = 0
                retweets_count = 0
                for line in open(os.path.join(src_file_path, file), 'r', encoding='utf-8'):
                    total_count += 1
                    record = ast.literal_eval(line)  # json.dumps
                    if 'data' in record:
                        if record['data']['lang'] != 'en':
                            continue
                        eng_count += 1

                        if 'text' in record['data']:
                            #get original tweets
                            if 'in_reply_to_user_id' in record['data'] or 'referenced_tweets' in record['data']:
                                retweets_count += 1
                                continue

                            origin_count += 1
                            save_data(line, file)

                print(
                    '{} total_count: {}, eng_count:{}, origin_count: {}, eng_percent:{:.2%}, orig_percent: {:.2%}, retweets_count: {}, percent: {:.2%}'.format(
                        file, total_count, eng_count, origin_count, eng_count / total_count, origin_count / eng_count,
                        retweets_count, retweets_count / eng_count))

                logging.info('{} total_count: {}, eng_count:{}, origin_count: {}, eng_percent:{:.2%}, orig_percent: {:.2%}, retweets_count: {}, percent: {:.2%}'.format(
                        file, total_count, eng_count, origin_count, eng_count / total_count, origin_count / eng_count,
                        retweets_count, retweets_count / eng_count))

        # 程序结束前关闭文件指针
        if file_object != None:
            file_object.close()


if __name__ == '__main__':
    # Get original tweets
    get_origin_tweets()

    print('DONE!')
