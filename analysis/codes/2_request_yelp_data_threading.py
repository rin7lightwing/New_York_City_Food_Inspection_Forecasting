from __future__ import print_function

import argparse
import json
import pprint
import requests
import sys
import urllib
import numpy as np
import pandas as pd
import json
import threading

# This client code can run on Python 2.x or 3.x.  Your imports can be
# simpler if you only need one of those.
try:
    # For Python 3.0 and later
    from urllib.error import HTTPError
    from urllib.parse import quote
    from urllib.parse import urlencode
except ImportError:
    # Fall back to Python 2's urllib2 and urllib
    from urllib2 import HTTPError
    from urllib import quote
    from urllib import urlencode


# Yelp Fusion no longer uses OAuth as of December 7, 2017.
# You no longer need to provide Client ID to fetch Data
# It now uses private keys to authenticate requests (API Key)
# You can find it on
# https://www.yelp.com/developers/v3/manage_app
API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
MATCH_PATH = '/v3/businesses/matches'
BUSINESS_PATH = '/v3/businesses/'  # Business ID will come after slash.
# Defaults for our simple example.
# DEFAULT_TERM = 'dinner'
DEFAULT_LOCATION = 'New York City, NY'
SEARCH_LIMIT = 10


def request(host, path, api_key, url_params=None):
    """Given your API_KEY, send a GET request to the API.
    Args:
        host (str): The domain host of the API.
        path (str): The path of the API after the domain.
        API_KEY (str): Your API Key.
        url_params (dict): An optional set of query parameters in the request.
    Returns:
        dict: The JSON response from the request.
    Raises:
        HTTPError: An error occurs from the HTTP request.
    """
    url_params = url_params or {}
    url = '{0}{1}'.format(host, quote(path.encode('utf8')))
    headers = {'Authorization': 'Bearer %s' % api_key,}
    response = requests.request('GET', url, headers=headers, params=url_params)
    return response.json()

def query_match(term, location, api_key):
    """Query the MATCH API by a term and location.
    Args:
        term (str): The search term passed to the API.
        location (str): The search location passed to the API.
        Note city and state parameters are set to default
    Returns:
        dict: The JSON response from the request.
    """
    name = '?name='+term.replace(' ','+')
    address = 'address1='+location.replace(' ','+')
    headers = {'Authorization': 'Bearer %s' % api_key,}
    url = API_HOST + MATCH_PATH + name +'&city=New+York+City&state=NY&country=US&'+ address
    return requests.request('GET', url, headers = headers, params = {'match_threshold': 'default'}).json()

def search(term, location, api_key):
    """Query the Search API by a search term and location.
    Args:
        term (str): The search term passed to the API.
        location (str): The search location passed to the API.
    Returns:
        dict: The JSON response from the request.
    """
    url_params = {
        'term': term.replace(' ', '+'),
        'location': location.replace(' ', '+'),
        'limit': SEARCH_LIMIT
    }
    return request(API_HOST, SEARCH_PATH, api_key, url_params=url_params)


def get_business(business_id, api_key):
    """Query the Business API by a business ID.
    Args:
        business_id (str): The ID of the business to query.
    Returns:
        dict: The JSON response from the request.
    """
    business_path = BUSINESS_PATH + business_id
    return request(API_HOST, business_path, api_key)

def query_api(term, api_key):
    """Queries the API by the input values from the user.
    Args:
        term (str): The search term to query.
        location (str): The location of the business to query.
    """
    response = search(term, DEFAULT_LOCATION, api_key)
    businesses = response.get('businesses')
    if not businesses:
        print(u'No businesses for {0} in {1} found.'.format(term, DEFAULT_LOCATION))
        return
    business_id = businesses[0]['id']
    response = get_business(business_id, api_key)
    return response


def batch_process_3000(start, end, api_key):

    local_df = df.iloc[start:end+1,:]

    for i in range(start,end+1):
        if i % 50 == 0:
            print("Thread-{}: Querying {}th entry".format(start, i))
        try:
            response = query_match(local_df['DBA'][i], local_df['ADDRESS'][i], api_key)
            if 'error' in response and 'code' in response['error'] and response['error']['code'] == 'ACCESS_LIMIT_REACHED':
                print("Thread-{} Stopped: ACCESS LIMIT REACHED!!!".format(start))
                break
            if 'error' in response:
                print(response)
                continue
            business = response['businesses']
            if business:
                data = get_business(business[0]['id'], api_key)
                for key in data.keys():
                    if key not in keep:
                        continue
                    elif key in keep1:
                        local_df.loc[i,'yelp_'+key] = data[key]
                    elif key == 'location':
                        for j in location_keys:
                            local_df.loc[i,'yelp_'+j] = data[key][j]
                    elif key == 'coordinates':
                        local_df.loc[i,'yelp_latitude'] = data[key]['latitude']
                        local_df.loc[i,'yelp_longitude']  = data[key]['longitude']
                    elif key == 'transactions':
                        local_df.loc[i,'yelp_'+key] = (', '.join(data['transactions']))
                    elif key == 'categories':
                        temp1 = json.dumps([item['alias'] for item in data[key]])
                        temp2 = json.dumps([item['title'] for item in data[key]])
                        local_df.loc[i,'yelp_'+key+'_a'] = temp1[1:len(temp1)-1]
                        local_df.loc[i,'yelp_'+key+'_t'] = temp2[1:len(temp2)-1]
                    elif key == 'hours':
                        dic = {item['day']:(item['start']+'-'+item['end'], item['is_overnight']) for item in data[key][0]['open']}
                        for day in range(7):
                            if day in dic.keys():
                                local_df.loc[i,'yelp_day'+str(day)], local_df.loc[i,'yelp_day'+str(day)+'overnight'] = dic[day]

                    elif key == 'price':
                        local_df.loc[i,'yelp_'+key] = len(data[key])

            else:
                print("Thread-{}: no business found for {}. Position-{}".format(start, local_df['DBA'][i], i))
        except json.decoder.JSONDecodeError:
            print("Thread-{}: Json Decoder Error in running {}. Position-{}".format(start, local_df['DBA'][i], i))
        except HTTPError as error:
            sys.exit(
                'Encountered HTTP error {0} on {1}:\n {2}\nAbort program.'.format(
                    error.code,error.url,error.read(),
                )
            )
        except Exception as e:
            print("Thread-{}: Run into error!!! Position-{}".format(start, i))
            print(e)

    local_df.to_csv('yelp_data_{}.csv'.format(start))
    return local_df


df = pd.read_csv('NYC_Inspection_Yelp.csv')
keep = ['name','id','is_closed','url','review_count','rating','phone','location','coordinates','categories','price','hours','transactions']
keep1 = ['name','id','is_closed','review_count','rating','phone','url']
location_keys = ['address1','city','state','zip_code']

# put your own api key into the list
API_KEYS = ['enter_api_key_0', 'enter_api_key_1']


threading.Thread(target=batch_process_3000, args=(0, 999, API_KEYS[0])).start()
threading.Thread(target=batch_process_3000, args=(1000, 1999, API_KEYS[0])).start()
threading.Thread(target=batch_process_3000, args=(2000, 2999, API_KEYS[1])).start()
threading.Thread(target=batch_process_3000, args=(3000, 3999, API_KEYS[1])).start()
# ...


