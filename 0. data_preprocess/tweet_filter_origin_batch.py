import os
import ast
import logging
from unrar import rarfile #export UNRAR_LIB_PATH=/home/amax/cyzhang/unrar/libunrar.so
import shutil

logging.basicConfig(level=logging.INFO, filename="./log/log-filter_origin_batch.txt",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s')
print("log is saving into log-filter_origin_batch.txt ...")
rar_file_path = "/data/twitter_data/twitter_compress_data/"
dst_file_path = "/data/twitter_data/origin_tweets/"
#rar_file_path = "D:/twitter_data/"
#dst_file_path = "D:/twitter_data/origin_tweets/"
rar_file_list = ['Sampled_Stream_20200602_0603.rar',
                 'Sampled_Stream_20200604_0608.rar',
                 'Sampled_Stream_detail_20200608_0614.rar',
                 'Sampled_Stream_detail_20200614_0619.rar',
                 'Sampled_Stream_detail_20200619_0630.rar',
                 'Filterd_Stream_20200629.rar',
                 'Sampled_Stream_detail_20200715_0720.rar',
                 'Sampled_Stream_detail_20200720_0726.rar',
                 'Sampled_Stream_detail_20200726_0731.rar',
                 'Sampled_Stream_detail_20200731_0804.rar',
                 'Sampled_Stream_detail_20200804_0807.rar',
                 'Sampled_Stream_detail_20200807_0811.rar',
                 'Sampled_Stream_detail_20200811_0815.rar',
                 'Sampled_Stream_detail_20200816_0821.rar',
                 'Sampled_Stream_detail_20200821_0824.rar',
                 'Sampled_Stream_detail_20200825_0828.rar',
                 'Sampled_Stream_detail_20200828_0830.rar',
                 'Sampled_Stream_detail_20200830_0904.rar',
                 'Sampled_Stream_detail_20200904_0908.rar',
                 'Sampled_Stream_detail_20200910_0917.rar',
                 'Sampled_Stream_detail_20200910_0914.rar',
                 'Sampled_Stream_detail_20200914_0917.rar',
                 'Sampled_Stream_detail_20200917_0921.rar',
                 'Sampled_Stream_detail_20200921_0924.rar',
                 'Sampled_Stream_detail_20200924_0928.rar',
                 'Sampled_Stream_detail_20200928_1002.rar',
                 'Sampled_Stream_detail_20201002_1006.rar',
                 'Sampled_Stream_detail_20201006_1009.rar',
                 'Sampled_Stream_detail_20201009_1012.rar',
                 'Sampled_Stream_detail_20201017_1020.rar', #10.12-10.17 missing
                 'Sampled_Stream_detail_20201020_1023.rar',
                 'Sampled_Stream_detail_20201023_1031.rar',
                 'Sampled_Stream_detail_20201031_1104.rar',
                 'Sampled_Stream_detail_20201105_1110.rar',
                 'Sampled_Stream_detail_20201110_1119.rar',
                 'Sampled_Stream_detail_20201119_1124.rar',
                 'Sampled_Stream_detail_20201124_1129.rar',
                 'Sampled_Stream_detail_20201129_1205.rar',
                 'Sampled_Stream_detail_20201205_1209.rar',
                 'Sampled_Stream_detail_20201210_1214.rar',
                 'Sampled_Stream_detail_20201214_1218.rar',
                 'Sampled_Stream_detail_20201218_1224.rar',
                 'Sampled_Stream_detail_20201224_1229.rar',
                 'Sampled_Stream_detail_20201229_0103.rar',
                 'Sampled_Stream_detail_20210103_0108.rar',
                 'Sampled_Stream_detail_20210108_0112.rar',
                 'Sampled_Stream_detail_20210112_0122.rar',
                 'Sampled_Stream_detail_20210122_0128.rar',
                 'Sampled_Stream_detail_20210128_0202.rar',
                 'Sampled_Stream_detail_20210202_0205.rar',
                 'Sampled_Stream_detail_20210206_0208.rar',
                 'Sampled_Stream_detail_20210209_0213.rar',
                 'Sampled_Stream_detail_20210213_0216.rar',
                 'Sampled_Stream_detail_20210216_0219.rar',
                 'Sampled_Stream_detail_20210219_0222.rar',
                 'Sampled_Stream_detail_20210222_0225.rar',
                 'Sampled_Stream_detail_20210225_0228.rar',
                 'Sampled_Stream_detail_20210228_0304.rar',
                 #update new data as of 2021.6
                 'Sampled_Stream_detail_20210304_0308.rar',
                 'Sampled_Stream_detail_20210308_0312.rar',
                 'Sampled_Stream_detail_20210312_0319.rar',
                 'Sampled_Stream_detail_20210319_0323.rar',
                 'Sampled_Stream_detail_20210323_0326.rar',
                 'Sampled_Stream_detail_20210326_0329.rar',
                 'Sampled_Stream_detail_20210329_0402.rar',
                 'Sampled_Stream_detail_20210402_0406.rar',
                 'Sampled_Stream_detail_20210406_0410.rar',
                 'Sampled_Stream_detail_20210410_0416.rar',
                 'Sampled_Stream_detail_20210416_0420.rar',
                 'Sampled_Stream_detail_20210420_0423.rar',
                 'Sampled_Stream_detail_20210423_0427.rar',
                 'Sampled_Stream_detail_20210427_0501.rar',
                 'Sampled_Stream_detail_20210501_0506.rar',
                 'Sampled_Stream_detail_20210506_0512.rar',
                 'Sampled_Stream_detail_20210512_0517.rar',
                 'Sampled_Stream_detail_20210517_0522.rar',
                 'Sampled_Stream_detail_20210522_0527.rar',
                 'Sampled_Stream_detail_20210527_0530.rar',
                 'Sampled_Stream_detail_20210530_0603.rar',
                 'Sampled_Stream_detail_20210607_0616.rar',
                 'Sampled_Stream_detail_20210616_0620.rar',
                 'Sampled_Stream_detail_20210620_0624.rar',
                 'Sampled_Stream_detail_20210624_0629.rar',
                 'Sampled_Stream_detail_20210629_0703.rar',]

records_per_file = 10000
count = 0
file_object = None
file_name = None

def save_data(item, filename, origin_file_path):
    global file_object, count, file_name
    if file_object is None:
        file_name = filename.replace("twitter_sample", "twitter_sample_origin")
        count += 1
        file_object = open("{}{}".format(origin_file_path, file_name), 'a', encoding='utf-8')
        file_object.write("{}".format(item))
        return
    if count == records_per_file:
        file_object.close()
        count = 1
        file_name = filename.replace("twitter_sample", "twitter_sample_origin")
        file_object = open("{}{}".format(origin_file_path, file_name), 'a', encoding='utf-8')
        file_object.write("{}".format(item))
        logging.warning("save " + str(records_per_file) + " items to file.")
    else:
        count += 1
        file_object.write("{}".format(item))

def get_origin_tweets(src_file_path, origin_file_path):
    global file_object, count
    logging.info("begin filtering orginal tweets in " + src_file_path)
    print("begin filtering orginal tweets in " + src_file_path)
    if not os.path.exists(src_file_path):
        print("no file in {}.".format(src_file_path))
        logging.info("no file in {}.".format(src_file_path))
        return None

    count = 0
    file_object = None
    if not os.path.exists(origin_file_path):
        os.mkdir(origin_file_path)

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
                            save_data(line, file, origin_file_path)

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
    for file in rar_file_list:
        rar_file_name = rar_file_path + file
        if os.path.exists(rar_file_name):
            src_file_path = rar_file_path + file.split('.rar')[0]
            origin_file_path = dst_file_path + file.split('.rar')[0] + '_origin/'
            if os.path.exists(origin_file_path):
                print("{} exists.".format(origin_file_path))
                logging.warning("{} exists.".format(origin_file_path))
                continue
            print("unrar {} ... ".format(rar_file_name))
            logging.info("unrar {} ... ".format(rar_file_name))
            rar = rarfile.RarFile(rar_file_path + file)
            rar.extractall(path=rar_file_path)

            # Get original tweets
            get_origin_tweets(src_file_path, origin_file_path)

            if os.path.exists(src_file_path):
                shutil.rmtree(src_file_path)
                print("delete {}. ".format(src_file_path))
                logging.info("delete {}.".format(src_file_path))

    print('DONE!')
