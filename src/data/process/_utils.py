from collections import defaultdict
import random
import numpy as np
import pandas as pd
import json
import pickle
import gzip
import tqdm

def add_comma(num):
    # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num)-i-1) % 3 == 0:
            res_num += ','
    return res_num[:-1]

def get_interaction(datas, data_name):
    if data_name == 'LastFM':
        # Repeated items
        user_seq = {}
        user_seq_notime = {}
        for data in datas:
            user, item, time = data
            if user in user_seq:
                if item not in user_seq_notime[user]:
                    user_seq[user].append((item, time))
                    user_seq_notime[user].append(item)
                else:
                    continue
            else:
                user_seq[user] = []
                user_seq_notime[user] = []

                user_seq[user].append((item, time))
                user_seq_notime[user].append(item)
    else:
        user_seq = {}
        for data in datas:
            user, item, time = data
            if user in user_seq:
                user_seq[user].append((item, time))
            else:
                user_seq[user] = []
                user_seq[user].append((item, time))
                
    time_interval = {}
    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # Sort individual data sets separately
        items = []
        times = []
        for i in range(len(item_time)):
            item = item_time[i][0]
            curr_time = item_time[i][1]
            prev_time = item_time[i-1][1] if i > 0 else item_time[i][1]
            items.append(item)
            times.append(curr_time-prev_time)

        user_seq[user] = items
        time_interval[user] = times

    return user_seq, time_interval

def id_map(user_items, time_interval): # user_items dict

    user2id = {} # raw 2 uid
    item2id = {} # raw 2 iid
    id2user = {} # uid 2 raw
    id2item = {} # iid 2 raw
    user_id = 1
    item_id = 1
    final_data = {}
    for user, items in user_items.items():
        if user not in user2id:
            user2id[user] = str(user_id)
            id2user[str(user_id)] = user
            user_id += 1
        iids = [] # item id lists
        for item in items:
            if item not in item2id:
                item2id[item] = str(item_id)
                id2item[str(item_id)] = item
                item_id += 1
            iids.append(item2id[item])
        uid = user2id[user]
        final_data[uid] = iids

    final_delta = {}
    for uid in id2user.keys():
        final_delta[uid] = time_interval[id2user[uid]]
        
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item
    }
    return final_data, final_delta, user_id-1, item_id-1, data_maps

# Circular filtration K-core
def filter_Kcore(user_items, time_interval, user_core, item_core): # User to all items
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core: # Remove the user directly.
                user_items.pop(user)
                time_interval.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        idx = user_items[user].index(item)
                        user_items[user].remove(item)
                        if idx != len(user_items[user]):        
                            time_interval[user][idx+1] += time_interval[user][idx]
                        time_interval[user].pop(idx)
                time_interval[user][0] = 0
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items, time_interval

# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True # Kcore is guaranteed.