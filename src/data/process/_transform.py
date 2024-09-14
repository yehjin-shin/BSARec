# -*- coding: utf-8 -*-
# @Time    : 2020/4/4 8:18
# @Author  : Hui Wang

from collections import defaultdict
import random
import numpy as np
import pandas as pd
import json
import pickle
import gzip
import sys
import tqdm
import _utils as dutils
import datetime
import matplotlib.pyplot as plt

# return (user item timestamp) sort in get_interaction
def Amazon(dataset_name, rating_score):
    '''
    reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    asin - ID of the product, e.g. 0000013714
    reviewerName - name of the reviewer
    helpful - helpfulness rating of the review, e.g. 2/3
    --"helpful": [2, 3],
    reviewText - text of the review
    --"reviewText": "I bought this for my husband who plays the piano. ..."
    overall - rating of the product
    --"overall": 5.0,
    summary - summary of the review
    --"summary": "Heavenly Highway Hymns",
    unixReviewTime - time of the review (unix time)
    --"unixReviewTime": 1252800000,
    reviewTime - time of the review (raw)
    --"reviewTime": "09 13, 2009"
    '''
    datas = []
    data_flie = './raw/{}.json'.format(dataset_name)
    lines = open(data_flie).readlines()
    for line in tqdm.tqdm(lines):
        inter = json.loads(line.strip())
        if float(inter['overall']) <= rating_score: # Less than a certain percentage.
            continue
        user = inter['reviewerID']
        item = inter['asin']
        time = inter['unixReviewTime']
        # "reviewTime": "09 13, 2009"
        datas.append((user, item, int(time)))

    return datas

def ML1M():
    datas = []
    filepath = './raw/ml-1m/ratings.dat'
    # import pdb; pdb.set_trace()
    df = pd.read_csv(filepath, delimiter='::', header=None, engine='python', names=['user', 'item', 'rating', 'timestamp'])
    df = df[['user', 'item', 'timestamp']]

    for i in tqdm.tqdm(range(len(df))):
        datas.append(tuple(df.iloc[i]))

    return datas

def Yelp(date_min, date_max, rating_score):
    datas = []
    data_flie = './raw/yelp_academic_dataset_review.json'
    years = []
    lines = open(data_flie).readlines()
    for line in tqdm.tqdm(lines):
        review = json.loads(line.strip())
        user = review['user_id']
        item = review['business_id']
        rating = review['stars']
        date = review['date']
        # Exclude some examples
        years.append(int(date.split('-')[0]))
        if date < date_min or date > date_max or float(rating) <= rating_score:
            continue
        time = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        datas.append((user, item, int(time.timestamp())))

    return datas

def LastFM():
    datas = []
    data_file = './raw/LastFM/user_taggedartists-timestamps.dat'
    lines = open(data_file).readlines()

    for line in tqdm.tqdm(lines[1:]):
        user, item, _, timestamp = line.strip().split('\t')
        datas.append((user, item, int(timestamp)))

    return datas

def process(data_name='Beauty'):
    np.random.seed(12345)
    rating_score = 0.0  # rating score smaller than this score would be deleted
    # user 5-core item 5-core
    user_core = 5
    item_core = 5

    if data_name in ["Sports_and_Outdoors", "Toys_and_Games", "Beauty"]:
        datas = Amazon(data_name, rating_score)
    if data_name == 'Yelp':
        date_max = '2019-12-31 00:00:00'
        date_min = '2019-01-01 00:00:00'
        datas = Yelp(date_min, date_max, rating_score)
    elif data_name == 'ML-1M':
        datas = ML1M()
    elif data_name == 'LastFM':
        datas = LastFM()

    user_items, time_interval = dutils.get_interaction(datas, data_name)
    print(f'{data_name} Raw data has been processed!')
    # raw_id user: [item1, item2, item3...]
    user_items, time_interval = dutils.filter_Kcore(user_items, time_interval, user_core=user_core, item_core=item_core)
    print(f'User {user_core}-core complete! Item {item_core}-core complete!')
    user_items, time_interval, user_num, item_num, data_maps = dutils.id_map(user_items, time_interval)  # new_num_id

    avg_seqlen = np.mean([len(seq) for seq in user_items.values()])
    user_count, item_count, _ = dutils.check_Kcore(user_items, user_core=user_core, item_core=item_core)
    user_count_list = list(user_count.values())

    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100
    show_info = f'\n====={data_name}=====\n' + \
                f'Total User: {user_num}, Avg User: {user_avg:.2f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.2f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Avg Sequence Length: {avg_seqlen:.1f}, Sparsity: {sparsity:.2f}%'
    print(show_info)

    item_file = '../'+ data_name + '.txt'
    with open(item_file, 'w') as out:
        for user, items in user_items.items():
            out.write(user + ' ' + ' '.join(items) + '\n')

if __name__ == "__main__":
    dataname = sys.argv[1]
    available_datasets = ['Beauty', 'Sports_and_Outdoors', 'Toys_and_Games', 'LastFM', 'ML-1M'] # TODO: Yelp
    if dataname == 'all':
        for name in available_datasets:
            process(name)
    elif dataname in available_datasets:
        process(dataname)
    else:
        print('Invalid dataset name')
        print(f"Available datasets: {available_datasets}")
        print("To transform all datasets at once, enter 'all' as the dataset name.")