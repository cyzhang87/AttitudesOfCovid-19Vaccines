#  Copyright (c) 2021.
#  Chunyan Zhang

import pandas as pd
import os, ast
import logging
from geopy.geocoders import Nominatim
from collections import Counter

logging.basicConfig(level=logging.INFO, filename="extract_loc.txt",
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(funcName)s - %(levelname)s: %(message)s')

print("log is saving into log_download_tweets_from_twitterid_oauth1.txt ...")

dir_name = "D:/twitter_data/origin_tweets/"
vaccine_file_list = ['Sampled_Stream_detail_20200715_0720_origin',
                     'Sampled_Stream_detail_20200811_0815_origin',
                     'Sampled_Stream_detail_20200914_0917_origin',
                     'Sampled_Stream_detail_20201105_1110_origin',
                     'Sampled_Stream_detail_20201210_1214_origin',
                     'Sampled_Stream_detail_20210410_0416_origin']

import re
import numpy as np
import geonamescache
import us
import pycountry
import pycountry_convert as pc
continents = {
    'NA': 'North America',
    'SA': 'South America',
    'AS': 'Asia',
    'OC': 'Australia',
    'AF': 'Africa',
}

gc = geonamescache.GeonamesCache()
c = gc.get_cities()

from city_to_state_dict import city_to_state_dict, countries, country_abbr

states_abbr = [r'\b' + us.STATES[i].abbr + r'(?![\w-])' for i in range(len(us.STATES))]
states_abbr_pattern = re.compile('|'.join(states_abbr))
country_abbr_re = [r'\b' + country_abbr[i] + r'(?![\w-])' for i in range(len(country_abbr))]
country_abbr_pattern = re.compile('|'.join(country_abbr_re))

two_word_states = set()
for key in city_to_state_dict.keys():
    if len(city_to_state_dict[key].split()) >= 2:
        two_word_states.add(city_to_state_dict[key])


def get_state_abbr(x):
    # if "Seatac, WA" in x:
    #    print(x)
    # 1. two-word state search
    if re.search('({})'.format("|".join(two_word_states)).lower(), x.lower()):
        tokens = [re.search('({})'.format("|".join(two_word_states)).lower(), x.lower()).group(0)]
    # 2. city search
    elif re.search('({})'.format("|".join(city_to_state_dict.keys()).lower()), x.lower()):
        k = re.search('({})'.format("|".join(city_to_state_dict.keys()).lower()), x.lower()).group(0)
        tokens = [city_to_state_dict.get(k, np.nan)]
    # 3. state name search
    else:
        tokens = [j for j in re.split("\s|,|/|\.|#|\-|\|", x) if j not in ['in', 'la', 'me', 'oh', 'or']]
    for i in tokens:
        if re.search('[0-9]+', str(i)):
            continue
        if i == 'NYC':
            i = "New York"
        if re.search('\w+', str(i)):
            if us.states.lookup(str(i)):
                return us.states.lookup(str(i)).abbr

    # 4. state abbr search
    if re.search(states_abbr_pattern, x):
        return re.search(states_abbr_pattern, x).group(0).upper()

    return 'null loc'


def get_country_abbr(x):
    try:
        return pycountry.countries.search_fuzzy(x)[0].alpha_2
    except:

        x = x.replace('.', '')
        # country name
        if re.search('({})'.format("|".join(countries).lower()), x.lower()):
            tokens = [re.search('({})'.format("|".join(countries).lower()), x.lower()).group(0)]
        elif re.search(country_abbr_pattern, x.lower()):
            return re.search(country_abbr_pattern, x.lower()).group(0)
        else:
            tokens = [j for j in re.split("[^a-zA-Z]", x) if j not in ['in', 'la', 'me', 'oh', 'or']]

        for i in tokens:
            if re.search('[0-9]+', str(i)):
                continue
            if re.search('\w+', str(i)):
                try:
                    return pycountry.countries.search_fuzzy(i)[0].alpha_2

                except:
                    continue

        return 'null loc'


def read_tweets(tweet_file):
    count = 0
    tweet_loc_list = []
    with open(tweet_file, "r", encoding='utf-8') as fhIn:
        for line in fhIn:
            if isinstance(line, str):
                line = ast.literal_eval(line)  # to dict
                tmp_loc = ['null loc', 'null loc', 'null loc']
                if 'includes' in line:
                    if 'location' in line['includes']['users'][0]:
                        tmp_loc[0] = line['includes']['users'][0]['location']
                        tmp_loc[1] = get_state_abbr(tmp_loc[0])

                if 'data' in line:
                    if 'geo' in line['data']:
                        tmp_loc[2] = line['data']['geo']['place_id']
            else:
                print(line + "error")
                return None

            tweet_loc_list.append(tmp_loc.copy())
            count += 1

            if count % 5000 == 0:
                print(count)

    print("{} read end".format(tweet_file))
    return tweet_loc_list

def read_tweets_2(file_dir, tweet_file):
    count = 0
    tweet_loc_list = []
    outfile_object = open(os.path.join(file_dir, 'geo.csv'), 'a', encoding='utf8')
    with open(tweet_file, "r", encoding='utf-8') as fhIn:
        for line in fhIn:
            if isinstance(line, str):
                line = ast.literal_eval(line)  # to dict
                tmp_loc = ['null loc', 'null loc', 'null loc']

                if 'data' in line:
                    if 'geo' in line['data']:
                        if 'place_id' in line['data']['geo']:
                            tmp_loc[2] = line['data']['geo']['place_id']
                            flag = False
                            while flag == False:
                                flag = connect_to_endpoint(tmp_loc[2])

                        if flag['country_code'] == 'US':
                            tmp_loc[1] = get_state_abbr(flag['full_name'])

                if 'includes' in line:
                    if 'location' in line['includes']['users'][0]:
                        tmp_loc[0] = line['includes']['users'][0]['location']
                        if tmp_loc[2] == 'null loc':
                            tmp_loc[1] = get_state_abbr(tmp_loc[0])
            else:
                print(line + "error")
                logging.warning(line + "error")
                return None

            tweet_loc_list.append(tmp_loc.copy())
            outfile_object.write("{}\n".format(','.join(tmp_loc.copy())))
            count += 1

            if count % 5000 == 0:
                print(count)
                logging.warning(count)
    outfile_object.close()
    total_loc_df = pd.DataFrame(data=tweet_loc_list, columns=['location', 'state', 'geo'])
    total_loc_df.to_csv(os.path.join(file_dir, "vaccine_location_geo.csv"), index=False)

    print("{} read end".format(tweet_file))
    logging.warning("{} read end".format(tweet_file))
    return tweet_loc_list


import requests, json, time
from requests.auth import AuthBase
from requests import exceptions

proxies = {
    "http": "http://127.0.0.1:10809", "https": "https://127.0.0.1:10809"
}
# App: COVID-19-Research
consumer_key = "eKscEcVx30iJZ5pg3TQPnIvDv"  # Add your API key here
consumer_secret = "4650nNPGT6JWiKYKsxogrYVntCvKADCDnJCxAqNt9llDy8zj7y"


# Gets a bearer token
class BearerTokenAuth(AuthBase):
    def __init__(self, consumer_key, consumer_secret):
        self.bearer_token_url = "https://api.twitter.com/oauth2/token"
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.bearer_token = self.get_bearer_token()

    def get_bearer_token(self):
        try:
            response = requests.post(
                self.bearer_token_url,
                auth=(self.consumer_key, self.consumer_secret),
                data={'grant_type': 'client_credentials'},
                headers={"User-Agent": "TwitterDevSampledStreamQuickStartPython"},
                proxies=proxies)

            if response.status_code is not 200:
                raise Exception(f"Cannot get a Bearer token (HTTP %d): %s" % (response.status_code, response.text))

            body = response.json()
            return body['access_token']

        except requests.exceptions.ConnectionError:
            print("ConnectionError")
            logging.warning("ConnectionError")
        except:
            print('Unfortunitely -- An Unknow Error Happened')
            logging.warning('Unfortunitely -- An Unknow Error Happened')

    def __call__(self, r):
        r.headers['Authorization'] = f"Bearer %s" % self.bearer_token
        return r


bearer_token = BearerTokenAuth(consumer_key, consumer_secret)

def connect_to_endpoint(place_id):
    url = "https://api.twitter.com/1.1/geo/id/{}.json".format(place_id)
    try:
        response = requests.request("GET", url, auth=bearer_token, proxies=proxies)
        if response.status_code is not 200:
            print(str(response.status_code) + response.text)
            logging.warning(str(response.status_code) + response.text)
            if response.status_code == 429:
                print("{} sleep 60 sec".format(place_id))
                logging.warning("{} sleep 60 sec".format(place_id))
                time.sleep(60)
            return False

        return json.loads(response.content)

    except exceptions.Timeout as e:
        print('Timeout: ' + str(e))
        logging.warning('Timeout: ' + str(e))
    except exceptions.HTTPError as e:
        print('HTTPError: ' + str(e))
        logging.warning('HTTPError: ' + str(e))
    except requests.exceptions.ConnectionError as e:
        print('ConnectionError: ' + str(e))
        logging.warning('ConnectionError: ' + str(e))
    except requests.exceptions.ChunkedEncodingError as e:
        print('ChunkedEncodingError: ' + str(e))
        logging.warning('ChunkedEncodingError: ' + str(e))
    except:
        print('Unfortunitely, an unknow error happened, please wait 3 seconds')
        logging.warning('Unfortunitely, an unknow error happened, please wait 3 seconds')
    return False


def extract_location():
    total_loc_list = []
    for file in vaccine_file_list:
        print("{}".format(file))
        logging.warning("{}".format(file))
        file_dir = os.path.join(dir_name, file)

        if not os.path.exists(file_dir):
            print('{} not exsits.'.format(file_dir))
            logging.warning('{} not exsits.'.format(file_dir))
            continue

        for root, dirs, files in os.walk(file_dir):
            for filename in files:
                if filename[:21] == "twitter_sample_origin":
                    tweet_file = os.path.join(file_dir, filename)
                    break

        total_loc_list += read_tweets_2(file_dir, tweet_file)

    total_loc_df = pd.DataFrame(data=total_loc_list, columns=['location', 'state', 'geo'])
    total_loc_df.to_csv(os.path.join(dir_name, "vaccine_location_geo.csv"), index=False)


def extract_country():
    country_list = []
    total_df = pd.read_csv(os.path.join(dir_name, "vaccine_location_geo.csv"), engine='python')
    for index, item in total_df.iterrows():
        if index % 1000 == 0:
            print(index)

        if item['state'] != 'null loc':
            country_list.append('US')
            continue

        if item['location'] == 'null loc' and item['geo'] == 'null loc':
            country_list.append('null loc')
            continue

        if item['geo'] != 'null loc':
            flag = False
            while flag == False:
                flag = connect_to_endpoint(item['geo'])
            country_list.append(flag['country_code'])
            continue

        if item['location'] != 'null loc':
            country_list.append(get_country_abbr(item['location']))
            continue

        print('left')

    total_df['country'] = country_list
    total_df.to_csv(os.path.join(dir_name, "vaccine_location_country_state.csv"), index=False)


def extract_country_enhance(dir_name):
    geolocator = Nominatim(user_agent='test')
    total_df = pd.read_csv(os.path.join(dir_name, "vaccine_location_country_state_2.csv"), keep_default_na=False,
                           na_values=['_'], engine='python')
    for index, item in total_df.iterrows():
        if index % 1000 == 0:
            print(index)
        if item['country'] != 'null loc':
            continue

        if item['location'] == 'null loc':
            continue

        try_count = 5
        status = False
        while not status:
            try:
                location = geolocator.geocode(item['location'], addressdetails=True)
                status = True
            except exceptions.Timeout as e:
                print('Timeout: ' + str(e))
            except exceptions.HTTPError as e:
                print('HTTPError: ' + str(e))
            except requests.exceptions.ConnectionError as e:
                print('ConnectionError: ' + str(e))
            except requests.exceptions.ChunkedEncodingError as e:
                print('ChunkedEncodingError: ' + str(e))
            except:
                print('Unfortunitely, an unknow error happened, please wait 1 seconds')

            if not status:
                print('error. try_count:{}'.format(try_count))
                if try_count == 0:
                    location = None
                    break
                time.sleep(1)
                try_count -= 1

        country_code = 'null loc'
        state_name = 'null loc'
        if location != None:
            if 'address' in location.raw:
                if location.raw['importance'] > 0.55:
                    if 'country_code' in location.raw['address']:
                        country_code = location.raw['address']['country_code'].upper()
                    if 'state' in location.raw['address']:
                        state_name = location.raw['address']['state']
                    if country_code == 'US' and state_name != 'null loc':
                        if us.states.lookup(state_name):
                            state_name = us.states.lookup(state_name).abbr.upper()

                    total_df.loc[index]['country'] = country_code
                    total_df.loc[index]['state'] = state_name

    total_df.to_csv(os.path.join(dir_name, "vaccine_location_country_state_2.csv"), index=False)

if __name__ == '__main__':
    """
    total_df = pd.read_csv(os.path.join(dir_name, "vaccine_location_country_state_2.csv"), keep_default_na=False,
                           na_values=['_'], engine='python')
    count = 0
    for file in vaccine_file_list:
        file_dir = os.path.join(dir_name, file)
        total_df[count:count+10000].to_csv(os.path.join(file_dir, "vaccine_location_country_state_2.csv"), index=False)
        count += 10000
    """
    #extract_location()
    #extract_country()
    extract_country_enhance(os.path.join(dir_name, vaccine_file_list[2]))

    """
    total_df = pd.read_csv(os.path.join(dir_name, "vaccine_location_country_state_2.csv"), keep_default_na=False,
                           na_values=['_'], engine='python')
    country_df = total_df[total_df['country'] != 'null loc']
    country_df = pd.DataFrame(Counter(country_df['country']).most_common(), columns=["Country", "Count"])
    country_df.to_csv("country_count.csv", index=False)

    total_loc_df = pd.read_csv(os.path.join(dir_name, "vaccine_location_geo.csv"), engine='python')
    total_loc_df_2 = total_loc_df[['state', 'geo']]
    total_loc_df_2.to_csv(os.path.join(dir_name, "vaccine_location_geo_3.csv"), index=False)

    total_loc_df = pd.read_csv(os.path.join(dir_name, "vaccine_location_geo_3.csv"))
    us_df = total_df[(total_df['country'] == 'US') & (total_df['state'] != 'null loc')]
    us_df = pd.DataFrame.from_dict(Counter(us_df['state']), orient='index', columns=["Count"])
    us_df = us_df.reset_index().rename(columns={'index': 'State'})
    us_df.to_csv("us_state_count_3.csv", index=False)
    """
    print("end")