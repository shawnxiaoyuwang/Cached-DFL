import numpy as np
from aggregation import average_weights, sum_weights, div_weights, mul_weights
import random,copy
import cvxpy as cp
import seed_setter
seed_setter.set_seed()
import collections

def get_mixing_weight(current_time,cached_time):
    a = 0.5
    b = 3
    #return (current_time-cached_time+1)**a
    return 1

def prune_cache(cache, type_limits, overall_limit, score: str, group_type: str):
    # Create a default dictionary to count occurrences
    type_counts = collections.defaultdict(list)

    # Populate the dictionary with elements grouped by type
    for key, value in cache.items():
        type_counts[value[group_type]].append((key, value))

    # Sort each type group by score in descending order
    for type_group in type_counts.values():
        type_group.sort(key=lambda x: x[1][score], reverse=True)

    # Create a new dictionary to hold the pruned cache
    pruned_cache = {}

    # Add elements to the pruned cache based on type limits
    for type_key, type_group in type_counts.items():
        pruned_cache.update({key: value for key, value in type_group})

    # If the total elements exceed the overall limit, remove the lowest scoring elements
    while len(pruned_cache) > overall_limit:
        # Find the group that surpasses its limit the most
        most_excess_group = None
        most_excess_count = 0
        
        for type_key, type_group in type_counts.items():
            limit = type_limits[type_key]
            excess_count = len(type_group) - limit
            if excess_count > most_excess_count:
                most_excess_group = type_key
                most_excess_count = excess_count

        # If there's no excess group, stop the loop
        if not most_excess_group:
            break

        # Remove the lowest-scoring item from the most excess group
        if most_excess_group:
            lowest_scoring_item = type_counts[most_excess_group].pop(-1)
            del pruned_cache[lowest_scoring_item[0]]

            # Update the type counts after removal
            if len(type_counts[most_excess_group]) == 0:
                del type_counts[most_excess_group]
    
    return pruned_cache

# def prune_cache_old(cache, type_limits, overall_limit,score:str,group_type:str):
#     # Create a default dictionary to count occurrences
#     type_counts = collections.defaultdict(list)

#     # Populate the dictionary with elements grouped by type
#     for key, value in cache.items():
#         type_counts[value[group_type]].append((key, value))

#     # Sort each type group by score in descending order
#     for type_group in type_counts.values():
#         type_group.sort(key=lambda x: x[1][score], reverse=True)

#     # Create a new dictionary to hold the pruned cache
#     pruned_cache = {}

#     # Add elements to the pruned cache based on type limits
#     for type_key, type_group in type_counts.items():
#         limit = type_limits.get(type_key, len(type_group))
#         for i in range(min(limit, len(type_group))):
#             key, value = type_group[i]
#             pruned_cache[key] = value

#     # If the total elements exceed the overall limit, remove the lowest scoring elements
#     if len(pruned_cache) > overall_limit:
#         # Convert the pruned_cache to a list and sort by score in ascending order
#         sorted_pruned_cache = sorted(pruned_cache.items(), key=lambda x: x[1][score])
        
#         # Remove elements until the total number of elements is within the overall limit
#         while len(pruned_cache) > overall_limit:
#             key_to_remove, _ = sorted_pruned_cache.pop(0)
#             del pruned_cache[key_to_remove]

#     return pruned_cache

def delete_smallest_value(d,term: str):
    # random delete the min time value
    #random shuffle the dictionary
    l = list(d.items())
    random.shuffle(l)
    d = dict(l)
    
    if not d:
        return d  # Return None if the dictionary is empty

    min_time = float('inf')
    min_key = None

    for key, value in d.items():
        if term in value and value[term] < min_time:
            min_time = value[term]
            min_key = key

    if min_key is not None:
        del d[min_key]
    
    return d

def delete_random(d):
    # random delete the min time value
    #random shuffle the dictionary
    l = list(d.items())
    random.shuffle(l)
    d = dict(l)
    
    if not d:
        return d  # Return None if the dictionary is empty
    del_key = None

    for key, value in d.items():
        del_key = key

    if del_key is not None:
        del d[del_key]
    
    return d

def delete_cache_global(d,cache_info):
    # random delete the min time value
    #random shuffle the dictionary
    l = list(d.items())
    random.shuffle(l)
    d = dict(l)
    
    if not d:
        return d,cache_info  # Return None if the dictionary is empty

    min_time = float('inf')
    max_count = 0
    remove_key = None

    #first is to delete the most cached item keep balance, then under the same balance, delete the oldest one
    for key, value in d.items():
        if max_count < cache_info[key]:
            max_count = cache_info[key]
            remove_key = key
        elif max_count == cache_info[key] and 'time' in value and value['time'] < min_time:
            min_time = value['time']
            remove_key = key

    if remove_key is not None:
        del d[remove_key]
        cache_info[remove_key] -= 1
    
    return d,cache_info

# def delete_smallest_time_count(d):
#     # random delete the min time value
#     #random shuffle the dictionary
#     l = list(d.items())
#     random.shuffle(l)
#     d = dict(l)
#     term_1 = 'time'
#     term_2 = 'count'
#     if not d:
#         return None  # Return None if the dictionary is empty

#     min_time = float('inf')
#     min_key = None

#     for key, value in d.items():
#         if term_1 in value and value[term_1] < min_time:
#             min_time = value[term]
#             min_key = key
#         elif 
#     if min_key is not None:
#         del d[min_key]
    
#     return d
def kick_out_timeout_model_cache_info(d,kick_out_time,cache_info):
    if not d:
        return d,cache_info  # Return None if the dictionary is empty
    keys = []
    for key, value in d.items():
        if 'time' in value and value['time'] <= kick_out_time:
            keys.append(key)
    
    for key in keys:
        cache_info[key] -= 1
        # print('delete '+str(key)+':'+str(d[key]['time'])+' from cache')
        del d[key]
    return d,cache_info


def kick_out_timeout_model(d,kick_out_time):
    if not d:
        return d  # Return None if the dictionary is empty
    keys = []
    for key, value in d.items():
        if 'time' in value and value['time'] <= kick_out_time:
            keys.append(key)
    
    for key in keys:
        # print('delete '+str(key)+':'+str(d[key]['time'])+' from cache')
        del d[key]
    return d

def kick_out_timeout_model_list(lst, kick_out_time):
    if not lst:
        return lst  # Return the empty list if it's empty

    # Filter out items where 'time' is less than or equal to kick_out_time
    pruned_list = [item for item in lst if item.get('time', float('inf')) > kick_out_time]

    return pruned_list


def duration_in_future(i,pair,duration, num_round):
    if i>=num_round:
        return np.zeros([10,10])
    duration[i] = duration_in_future(i+1,pair,duration, num_round)
    for a,b in pair[i]:
        duration[i][a][b] = num_round - i - 1
        duration[i][b][a] = num_round - i - 1
    return duration[i]

def merge_dictionaries(dict1, dict2):
    merged_dict = copy.deepcopy(dict1)

    for key, value in dict2.items():
        if key in merged_dict:
            # If the key is already in merged_dict, compare values and take the higher one
            merged_dict[key]['time'] = max(merged_dict[key]['time'], value['time'])
        else:
            # If the key is not in merged_dict, add it to the merged dictionary
            merged_dict[key] = value
    #random shuffle the dictionary
    l = list(merged_dict.items())
    random.shuffle(l)
    merged_dict = dict(l)
    # sort the dictionary based on values
    merged_dict = dict(sorted(merged_dict.items(), key=lambda item: item[1]['time'],reverse=True)) 
    return merged_dict




def non_confliction_on_slot(cache_a,cache_b,slot):
        string1 = cache_a[slot]['mixing_record']
        string2 = cache_b[slot]['mixing_record']
        set1 = set(string1)
        set2 = set(string2)
        
        # Check if there's an intersection between the sets
        common_chars = set1.intersection(set2)
        
        # If the intersection set is not empty, there are common characters
        if common_chars:
            return False
        else:
            return True
           
        
def update_model_cache_only_one(local_cache,model_a,model_b,a,b,round_index,cache_size, kick_out):
    
    #update other's model into cache
    temp_model_a = copy.deepcopy(model_a)
    temp_model_b = copy.deepcopy(model_b)
    local_cache[a][b] = {'model' :temp_model_b,'time' : round_index}
    local_cache[b][a] = {'model' : temp_model_a,'time' : round_index}
    
    
    #kick out time-out model
    # if kick_out == True:
    #     kick_out_timeout_model(local_cache[a],round_index-cache_size)
    #     kick_out_timeout_model(local_cache[b],round_index-cache_size)
        
    #keep satisfying the cache size
    while len(local_cache[a])>cache_size:
        local_cache[a] = delete_smallest_value(local_cache[a],'time')
    while len(local_cache[b])>cache_size:
        local_cache[b] = delete_smallest_value(local_cache[b],'time')
        
        
# def put_own_model_into_cache(local_cache, model_list,index,round_index):
#     #update own model into cache
#     local_cache[index][index] = {'model' : model_list[index],'time' : round_index}


def update_model_cache_fresh(local_cache, model_a,model_b,a,b,round_index,cache_size, model_fresh_table,metric:str, kick_out):
    
    old_local_cache_a = copy.deepcopy(local_cache[a])
    old_local_cache_b = copy.deepcopy(local_cache[b])
    temp_model_fresh_table = copy.deepcopy(model_fresh_table)
    temp_model_a = copy.deepcopy(model_a)
    temp_model_b = copy.deepcopy(model_b)
    
    #update other's model into cache
    if metric =='mean':
        fresh_metric_b = temp_model_fresh_table[b].mean()
        fresh_metric_a = temp_model_fresh_table[a].mean()
    elif metric == 'min':
        fresh_metric_b = temp_model_fresh_table[b].min()
        fresh_metric_a = temp_model_fresh_table[a].min()
    else:print('please provide correct prompt!')
    local_cache[a][b] = {'model' : temp_model_b,'time' : round_index,'class_fresh': temp_model_fresh_table[b],'fresh':fresh_metric_b}
    local_cache[b][a] = {'model' : temp_model_a,'time' : round_index,'class_fresh': temp_model_fresh_table[a],'fresh':fresh_metric_a}
    
    #note here 'time' means the update round, while 'fresh' indicates the model freshness, those two are closed but not exactly the same.
    #update cache by fetching other's cache
    for key in old_local_cache_a:
        if key == b:
            continue;
        if key not in local_cache[b]:
            local_cache[b][key] = old_local_cache_a[key].copy()
        elif local_cache[b][key]['fresh']<old_local_cache_a[key]['fresh']:
            local_cache[b][key] = old_local_cache_a[key].copy()
            
    for key in old_local_cache_b:
        if key == a:
            continue;
        if key not in local_cache[a]:
            local_cache[a][key] = old_local_cache_b[key].copy()
        elif local_cache[a][key]['fresh']<old_local_cache_b[key]['fresh']:
            local_cache[a][key] = old_local_cache_b[key].copy()

    #kick out time-out model
    # if kick_out == True:
    #     local_cache[a] = kick_out_timeout_model(local_cache[a],round_index-cache_size)
    #     local_cache[b] = kick_out_timeout_model(local_cache[b],round_index-cache_size)
    
    
    #keep satisfying the cache size
    while len(local_cache[a])>cache_size:
        local_cache[a] = delete_smallest_value(local_cache[a],'fresh')
    while len(local_cache[b])>cache_size:
        local_cache[b] = delete_smallest_value(local_cache[b],'fresh')
        
        
        

        
        
        
def update_model_cache_fresh_count(local_cache, model_a,model_b,a,b,round_index,cache_size, model_fresh_table, cache_statistic_table, kick_out):
    
    alpha = 0.01 
    old_local_cache_a = copy.deepcopy(local_cache[a])
    old_local_cache_b = copy.deepcopy(local_cache[b])
    temp_model_fresh_table = copy.deepcopy(model_fresh_table)
    temp_model_a = copy.deepcopy(model_a)
    temp_model_b = copy.deepcopy(model_b)
    
    # cache_statistic_table[a][round_index] += 1
    # cache_statistic_table[b][round_index] += 1
    #update other's model into cache
    fresh_metric_b = temp_model_fresh_table[b].mean()
    fresh_metric_a = temp_model_fresh_table[a].mean()
    local_cache[a][b] = {'model' : temp_model_b,'time' : round_index,'class_fresh': temp_model_fresh_table[b],'fresh':fresh_metric_b,'cache_score':0}
    local_cache[b][a] = {'model' : temp_model_a,'time' : round_index,'class_fresh': temp_model_fresh_table[a],'fresh':fresh_metric_a,'cache_score':0}
    
    
    
    # refresh update count information    
    for key in local_cache[a]:
        # local_cache[a][key]['update_count'] = cache_statistic_table[key][local_cache[a][key]['time']]
        local_cache[a][key]['cache_score'] = local_cache[a][key]['time'] - cache_statistic_table[key][local_cache[a][key]['time']]*alpha
        
    for key in local_cache[b]:
        # local_cache[b][key]['update_count'] = cache_statistic_table[key][local_cache[b][key]['time']]
        local_cache[b][key]['cache_score'] = local_cache[b][key]['time'] - cache_statistic_table[key][local_cache[b][key]['time']]*alpha
    
    #note here 'time' means the update round, while 'fresh' indicates the model freshness, those two are closed but not exactly the same.
    #update cache by fetching other's cache
    for key in old_local_cache_a:
        if key == b:
            continue;
        if key not in local_cache[b]:
            local_cache[b][key] = old_local_cache_a[key].copy()
        elif local_cache[b][key]['cache_score']<old_local_cache_a[key]['cache_score']:
            local_cache[b][key] = old_local_cache_a[key].copy()
            
    for key in old_local_cache_b:
        if key == a:
            continue;
        if key not in local_cache[a]:
            local_cache[a][key] = old_local_cache_b[key].copy()
        elif local_cache[a][key]['cache_score']<old_local_cache_b[key]['cache_score']:
            local_cache[a][key] = old_local_cache_b[key].copy()

    #kick out time-out model
    # if kick_out == True:
    #     local_cache[a] = kick_out_timeout_model(local_cache[a],round_index-cache_size)
    #     local_cache[b] = kick_out_timeout_model(local_cache[b],round_index-cache_size)
    
    
    # #keep satisfying the cache size
    # while len(local_cache[a])>cache_size:
    #     local_cache[a] = delete_smallest_time_count(local_cache[a])
    # while len(local_cache[b])>cache_size:
    #     local_cache[b] = delete_smallest_time_count(local_cache[b])
    #keep satisfying the cache size
    while len(local_cache[a])>cache_size:
        local_cache[a] = delete_smallest_value(local_cache[a],'cache_score')
    while len(local_cache[b])>cache_size:
        local_cache[b] = delete_smallest_value(local_cache[b],'cache_score')
        
        
def update_model_cache_fresh_v2(local_cache, model_a,model_b,a,b,round_index,cache_size, model_fresh_table, cache_statistic_table, kick_out):
     
    old_local_cache_a = copy.deepcopy(local_cache[a])
    old_local_cache_b = copy.deepcopy(local_cache[b])
    temp_model_fresh_table = copy.deepcopy(model_fresh_table)
    temp_model_a = copy.deepcopy(model_a)
    temp_model_b = copy.deepcopy(model_b)
    
    # cache_statistic_table[a][round_index] += 1
    # cache_statistic_table[b][round_index] += 1
    #update other's model into cache
    fresh_metric_b = temp_model_fresh_table[b].mean()
    fresh_metric_a = temp_model_fresh_table[a].mean()
    local_cache[a][b] = {'model' : temp_model_b,'time' : round_index,'class_fresh': temp_model_fresh_table[b],'fresh':fresh_metric_b}
    local_cache[b][a] = {'model' : temp_model_a,'time' : round_index,'class_fresh': temp_model_fresh_table[a],'fresh':fresh_metric_a}
    
    
    
    # refresh update count information    
    for key in local_cache[a]:
        # local_cache[a][key]['update_count'] = cache_statistic_table[key][local_cache[a][key]['time']]
        local_cache[a][key]['cache_score_a'] = sum(filter(lambda x:x>0, local_cache[a][key]['class_fresh'] - temp_model_fresh_table[a]))/max(1,sum(1 for x in local_cache[a][key]['class_fresh'] - temp_model_fresh_table[a] if x > 0)) #- cache_statistic_table[key][local_cache[a][key]['time']]*alpha
        local_cache[a][key]['cache_score_b'] =  sum(filter(lambda x:x>0, local_cache[a][key]['class_fresh'] - temp_model_fresh_table[b])) /max(1,sum(1 for x in local_cache[a][key]['class_fresh'] - temp_model_fresh_table[b] if x > 0))
        
    for key in local_cache[b]:
        # local_cache[b][key]['update_count'] = cache_statistic_table[key][local_cache[b][key]['time']]
        local_cache[b][key]['cache_score_a'] =  sum(filter(lambda x:x>0, local_cache[b][key]['class_fresh'] - temp_model_fresh_table[a]))/max(1,sum(1 for x in local_cache[b][key]['class_fresh'] - temp_model_fresh_table[a] if x > 0))  #- cache_statistic_table[key][local_cache[b][key]['time']]*alpha
        local_cache[b][key]['cache_score_b'] =  sum(filter(lambda x:x>0, local_cache[b][key]['class_fresh'] - temp_model_fresh_table[b])) /max(1,sum(1 for x in local_cache[b][key]['class_fresh'] - temp_model_fresh_table[b] if x > 0))
        
    #note here 'time' means the update round, while 'fresh' indicates the model freshness, those two are closed but not exactly the same.
    #update cache by fetching other's cache
    for key in old_local_cache_a:
        if key == b:
            continue;
        if key not in local_cache[b]:
            local_cache[b][key] = old_local_cache_a[key].copy()
        elif local_cache[b][key]['cache_score_b']<old_local_cache_a[key]['cache_score_b']:
            local_cache[b][key] = old_local_cache_a[key].copy()
            
    for key in old_local_cache_b:
        if key == a:
            continue;
        if key not in local_cache[a]:
            local_cache[a][key] = old_local_cache_b[key].copy()
        elif local_cache[a][key]['cache_score_a']<old_local_cache_b[key]['cache_score_a']:
            local_cache[a][key] = old_local_cache_b[key].copy()

    #kick out time-out model
    # if kick_out == True:
    #     local_cache[a] = kick_out_timeout_model(local_cache[a],round_index-cache_size)
    #     local_cache[b] = kick_out_timeout_model(local_cache[b],round_index-cache_size)
    
    for key in local_cache[a]:
        local_cache[a][key]['cache_score'] = local_cache[a][key]['cache_score_a']
    for key in local_cache[b]:
        local_cache[b][key]['cache_score'] = local_cache[b][key]['cache_score_b']
    # #keep satisfying the cache size
    # while len(local_cache[a])>cache_size:
    #     local_cache[a] = delete_smallest_time_count(local_cache[a])
    # while len(local_cache[b])>cache_size:
    #     local_cache[b] = delete_smallest_time_count(local_cache[b])
    #keep satisfying the cache size
    while len(local_cache[a])>cache_size:
        local_cache[a] = delete_smallest_value(local_cache[a],'cache_score')
    while len(local_cache[b])>cache_size:
        local_cache[b] = delete_smallest_value(local_cache[b],'cache_score')
        
        

def update_model_cache_fresh_v3(local_cache, model_a,model_b,a,b,round_index,cache_size, model_fresh_table,  kick_out):
     
    old_local_cache_a = copy.deepcopy(local_cache[a])
    old_local_cache_b = copy.deepcopy(local_cache[b])
    temp_model_fresh_table = copy.deepcopy(model_fresh_table)
    temp_model_a = copy.deepcopy(model_a)
    temp_model_b = copy.deepcopy(model_b)
    
    # cache_statistic_table[a][round_index] += 1
    # cache_statistic_table[b][round_index] += 1
    #update other's model into cache
    # fresh_metric_b = temp_model_fresh_table[b].mean()
    # fresh_metric_a = temp_model_fresh_table[a].mean()
    local_cache[a][b] = {'model' : temp_model_b,'time' : round_index,'fresh': temp_model_fresh_table[b]}
    local_cache[b][a] = {'model' : temp_model_a,'time' : round_index,'fresh': temp_model_fresh_table[a]}
    
    
    
    # # refresh update count information    
    # for key in local_cache[a]:
    #     # local_cache[a][key]['update_count'] = cache_statistic_table[key][local_cache[a][key]['time']]
    #     local_cache[a][key]['cache_score_a'] = sum(filter(lambda x:x>0, local_cache[a][key]['class_fresh'] - temp_model_fresh_table[a]))/max(1,sum(1 for x in local_cache[a][key]['class_fresh'] - temp_model_fresh_table[a] if x > 0)) #- cache_statistic_table[key][local_cache[a][key]['time']]*alpha
    #     local_cache[a][key]['cache_score_b'] =  sum(filter(lambda x:x>0, local_cache[a][key]['class_fresh'] - temp_model_fresh_table[b])) /max(1,sum(1 for x in local_cache[a][key]['class_fresh'] - temp_model_fresh_table[b] if x > 0))
        
    # for key in local_cache[b]:
    #     # local_cache[b][key]['update_count'] = cache_statistic_table[key][local_cache[b][key]['time']]
    #     local_cache[b][key]['cache_score_a'] =  sum(filter(lambda x:x>0, local_cache[b][key]['class_fresh'] - temp_model_fresh_table[a]))/max(1,sum(1 for x in local_cache[b][key]['class_fresh'] - temp_model_fresh_table[a] if x > 0))  #- cache_statistic_table[key][local_cache[b][key]['time']]*alpha
    #     local_cache[b][key]['cache_score_b'] =  sum(filter(lambda x:x>0, local_cache[b][key]['class_fresh'] - temp_model_fresh_table[b])) /max(1,sum(1 for x in local_cache[b][key]['class_fresh'] - temp_model_fresh_table[b] if x > 0))
        
    #note here 'time' means the update round, while 'fresh' indicates the model freshness, those two are closed but not exactly the same.
    #update cache by fetching other's cache
    for key in old_local_cache_a:
        if key == b:
            continue;
        if key not in local_cache[b]:
            local_cache[b][key] = old_local_cache_a[key].copy()
        elif local_cache[b][key]['fresh']<old_local_cache_a[key]['fresh']:
            local_cache[b][key] = old_local_cache_a[key].copy()
            
    for key in old_local_cache_b:
        if key == a:
            continue;
        if key not in local_cache[a]:
            local_cache[a][key] = old_local_cache_b[key].copy()
        elif local_cache[a][key]['fresh']<old_local_cache_b[key]['fresh']:
            local_cache[a][key] = old_local_cache_b[key].copy()

    #kick out time-out model
    # if kick_out == True:
    #     local_cache[a] = kick_out_timeout_model(local_cache[a],round_index-cache_size)
    #     local_cache[b] = kick_out_timeout_model(local_cache[b],round_index-cache_size)
    
    # #keep satisfying the cache size
    # while len(local_cache[a])>cache_size:
    #     local_cache[a] = delete_smallest_time_count(local_cache[a])
    # while len(local_cache[b])>cache_size:
    #     local_cache[b] = delete_smallest_time_count(local_cache[b])
    #keep satisfying the cache size
    while len(local_cache[a])>cache_size:
        local_cache[a] = delete_smallest_value(local_cache[a],'fresh')
    while len(local_cache[b])>cache_size:
        local_cache[b] = delete_smallest_value(local_cache[b],'fresh')
        
    
        
        
def update_model_cache_combination(local_cache, model_a,model_b,a,b,round_index,cache_size, model_time_table,model_combination_table):
    
    old_local_cache_a = copy.deepcopy(local_cache[a])
    old_local_cache_b = copy.deepcopy(local_cache[b])
    temp_model_combination_table = copy.deepcopy(model_combination_table)
    temp_model_time_table = copy.deepcopy(model_time_table)
    temp_model_a = copy.deepcopy(model_a)
    temp_model_b = copy.deepcopy(model_b)
    
    #update other's model into cache
    local_cache[a][b] = {'model' : temp_model_b,'time' : temp_model_time_table[b],'combination': temp_model_combination_table[b]}
    local_cache[b][a] = {'model' : temp_model_a,'time' : temp_model_time_table[a],'combination': temp_model_combination_table[a]}
    
    
    #update cache by fetching other's cache
    for key in old_local_cache_a:
        if key == b:
            continue;
        if key not in local_cache[b]:
            local_cache[b][key] = old_local_cache_a[key].copy()
        elif local_cache[b][key]['time']<old_local_cache_a[key]['time']:
            local_cache[b][key] = old_local_cache_a[key].copy()
            
    for key in old_local_cache_b:
        if key == a:
            continue;
        if key not in local_cache[a]:
            local_cache[a][key] = old_local_cache_b[key].copy()
        elif local_cache[a][key]['time']<old_local_cache_b[key]['time']:
            local_cache[a][key] = old_local_cache_b[key].copy()

    #kick out time-out model
    
    # local_cache[a] = kick_out_timeout_model(local_cache[a],round_index-cache_size)
    # local_cache[b] = kick_out_timeout_model(local_cache[b],round_index-cache_size)
    
    
    #keep satisfying the cache size
    while len(local_cache[a])>cache_size:
        local_cache[a] = delete_smallest_value(local_cache[a],'time')
    while len(local_cache[b])>cache_size:
        local_cache[b] = delete_smallest_value(local_cache[b],'time')


def update_best_model_cache(local_cache, model_a,model_b,a,b,round_index,cache_size, test_score):
    
    old_local_cache_a = copy.deepcopy(local_cache[a])
    old_local_cache_b = copy.deepcopy(local_cache[b])
    temp_model_a = copy.deepcopy(model_a)
    temp_model_b = copy.deepcopy(model_b)
    
    #update other's model into cache
    local_cache[a][b] = {'model' : temp_model_b,'time' : test_score[b]}
    local_cache[b][a] = {'model' : temp_model_a,'time' : test_score[a]}
    
    
    #update cache by fetching other's cache
    for key in old_local_cache_a:
        if key == b:
            continue;
        if key not in local_cache[b]:
            local_cache[b][key] = old_local_cache_a[key].copy()
        elif local_cache[b][key]['time']<old_local_cache_a[key]['time']:
            local_cache[b][key] = old_local_cache_a[key].copy()
            
    for key in old_local_cache_b:
        if key == a:
            continue;
        if key not in local_cache[a]:
            local_cache[a][key] = old_local_cache_b[key].copy()
        elif local_cache[a][key]['time']<old_local_cache_b[key]['time']:
            local_cache[a][key] = old_local_cache_b[key].copy()

    # #kick out time-out model
    # if kick_out == True:
    #     local_cache[a] = kick_out_timeout_model(local_cache[a],round_index-cache_size)
    #     local_cache[b] = kick_out_timeout_model(local_cache[b],round_index-cache_size)
    
    
    #keep satisfying the cache size
    while len(local_cache[a])>cache_size:
        local_cache[a] = delete_smallest_value(local_cache[a],'time')
    while len(local_cache[b])>cache_size:
        local_cache[b] = delete_smallest_value(local_cache[b],'time')

def update_model_cache_distribution(local_cache, model_a,model_b,a,b,round_index,cache_size,age_threshold,statistic_data,max_std,alpha):
    
    temp_model_a = copy.deepcopy(model_a)
    temp_model_b = copy.deepcopy(model_b)
    
    #update other's model into cache
    local_cache[a][b] = {'model' : temp_model_b,'time' : round_index,'distribution':statistic_data[b]}
    local_cache[b][a] = {'model' : temp_model_a,'time' : round_index,'distribution':statistic_data[a]}
    
    #update cache score
    for key in local_cache[a]:
        age_score = ( local_cache[a][key]['time']-round_index)/age_threshold
        distribution_score = np.var(statistic_data[a]-local_cache[a][key]['distribution'])/max_std[a]
        # print(age_score)
        # print(distribution_score)
        local_cache[a][key]['cache_score'] = alpha*age_score + (1-alpha)*distribution_score

    for key in local_cache[b]:
        age_score = (local_cache[b][key]['time']-round_index)/age_threshold
        distribution_score = np.var(statistic_data[b]-local_cache[b][key]['distribution'])/max_std[b]
        local_cache[b][key]['cache_score'] = alpha*age_score + (1-alpha)*distribution_score

    old_local_cache_a = copy.deepcopy(local_cache[a])
    old_local_cache_b = copy.deepcopy(local_cache[b])

    #update cache by fetching other's cache, based on cache score
    for key in old_local_cache_a:
        if key == b:
            continue
        if key not in local_cache[b]:
            local_cache[b][key] = old_local_cache_a[key].copy()
        elif local_cache[b][key]['cache_score']<old_local_cache_a[key]['cache_score']:
            local_cache[b][key] = old_local_cache_a[key].copy()
            
    for key in old_local_cache_b:
        if key == a:
            continue
        if key not in local_cache[a]:
            local_cache[a][key] = old_local_cache_b[key].copy()
        elif local_cache[a][key]['cache_score']<old_local_cache_b[key]['cache_score']:
            local_cache[a][key] = old_local_cache_b[key].copy()

    #kick out time-out model
    # if kick_out == True:
    #     local_cache[a] = kick_out_timeout_model(local_cache[a],round_index-cache_size)
    #     local_cache[b] = kick_out_timeout_model(local_cache[b],round_index-cache_size)
    
    
    #keep satisfying the cache size
    while len(local_cache[a])>cache_size:
        local_cache[a] = delete_smallest_value(local_cache[a],'cache_score')
    while len(local_cache[b])>cache_size:
        local_cache[b] = delete_smallest_value(local_cache[b],'cache_score')
def update_model_cache_mixing_old(local_cache, model_list,a,b,round_index,mixing_table):
    #update own model into cache
    local_cache[a]['self'] = {'model' : {a:model_list[a]},'time' : [round_index],'mixing_record':str(a)}
    local_cache[b]['self'] = {'model' : {b:model_list[b]},'time' : [round_index],'mixing_record':str(b)}
    
    

    # # #update cache based on other cache:
    # print('update based on cache')
    # old_local_cache_a = copy.deepcopy(local_cache[a])
    # old_local_cache_b = copy.deepcopy(local_cache[b])
    # for slot in range(cache_size):
    #     #update a to b
    #     if slot in old_local_cache_a:
    #         if str(b) in old_local_cache_a[slot]['mixing_record']:
    #             pass;
    #         elif not slot in local_cache[b]:
    #             local_cache[b][slot] = old_local_cache_a[slot].copy()
    #             print('copy '+str(a)+' to '+str(b)+' on slot '+str(slot))
    #         elif non_confliction_on_slot(old_local_cache_a,local_cache[b],slot):
    #             for key in old_local_cache_a[slot]['model']:
    #                 local_cache[b][slot]['model'][key] = old_local_cache_a[slot]['model'][key]
    #             print('mix '+str(b)+' to '+str(a)+' on slot '+str(slot))
    #             print('detail: '+old_local_cache_a[slot]['mixing_record']+' with '+local_cache[b][slot]['mixing_record'])
    #             local_cache[b][slot]['time'] += old_local_cache_a[slot]['time']
    #             local_cache[b][slot]['mixing_record'] += old_local_cache_a[slot]['mixing_record']
    #         elif np.mean(old_local_cache_a[slot]['time'])>np.mean(local_cache[b][slot]['time']):
    #             local_cache[b][slot] = old_local_cache_a[slot].copy()
    #             print('replace '+str(a)+' to '+str(b)+' on slot '+str(slot))
    #     #update b to a        
    #     if slot in old_local_cache_b:
    #         #if cache b has a model, skip
    #         if str(a) in old_local_cache_b[slot]['mixing_record']:
    #             pass;
    #         # if slot on cache a is empty, copy from cache b
    #         elif not slot in local_cache[a]:
    #             local_cache[a][slot] = old_local_cache_b[slot].copy()
    #             print('copy '+str(b)+' to '+str(a)+' on slot '+str(slot))
    #         # if models in cache b or cache a are non-overlap, add to cache a
    #         elif non_confliction_on_slot(local_cache[a],old_local_cache_b,slot):
    #             for key in old_local_cache_b[slot]['model']:
    #                 local_cache[a][slot]['model'][key] = old_local_cache_b[slot]['model'][key]
    #             print('mix '+str(a)+' to '+str(b)+' on slot '+str(slot))
    #             print('detail: '+old_local_cache_b[slot]['mixing_record']+' with '+local_cache[a][slot]['mixing_record'])
    #             local_cache[a][slot]['time'] += old_local_cache_b[slot]['time']
    #             local_cache[a][slot]['mixing_record'] += old_local_cache_b[slot]['mixing_record']
    #         # else compare the freshness, choose the newest one.
    #         elif np.mean(old_local_cache_b[slot]['time'])>np.mean(local_cache[a][slot]['time']):
    #             local_cache[a][slot] = old_local_cache_b[slot].copy()
    #             print('replace '+str(b)+' to '+str(a)+' on slot '+str(slot))
           


    # for cache a
    #check if there is already model existing
    slot = mixing_table[b]
    # already exist model on slot
    print('update based on model')
    
    if slot in local_cache[a]:
        # exist old model, evict all the model, replace with newest model
        if str(b) in local_cache[a][slot]['mixing_record']:
            local_cache[a][slot] = {'model' : {b:model_list[b]},'time' : [round_index],'mixing_record':str(b)}
            print('replace '+str(b)+' to '+str(a)+' on slot '+str(slot))
        # no existed old model, add to mix
        else:
            print('mix '+str(b)+' to '+str(a)+' on slot '+str(slot))
            # print(local_cache[b][slot]['model'].keys())
            print('detail: '+local_cache[a][slot]['mixing_record']+',whose content:'+str(local_cache[a][slot]['model'].keys())+' with '+str(b))
            local_cache[a][slot]['model'][b] = model_list[b]
            local_cache[a][slot]['time'] += [round_index]
            local_cache[a][slot]['mixing_record'] += str(b)
    # no model on slot
    else:
        local_cache[a][slot] =  {'model' : {b:model_list[b]},'time' : [round_index],'mixing_record':str(b)}
        print('since slot is empty put '+str(b)+' to '+str(a)+' on slot '+str(slot))
        
    # for cache b
    #check if there is already model existing
    slot = mixing_table[a]
    # already exist model on slot
    if  slot in local_cache[b]:
        # exist old model, evict all the model, replace with newest model
        if str(a) in local_cache[b][slot]['mixing_record']:
            local_cache[b][slot] = {'model' : {a:model_list[a]},'time' : [round_index],'mixing_record':str(a)}
            print('replace '+str(a)+' to '+str(b)+' on slot '+str(slot))
        # no existed old model, add to mix
        else:
            print('mix '+str(a)+' to '+str(b)+' on slot '+str(slot))
            print('detail: '+local_cache[b][slot]['mixing_record']+',whose content:'+str(local_cache[b][slot]['model'].keys())+' with '+str(a))
            local_cache[b][slot]['model'][a]=model_list[a]
            local_cache[b][slot]['time'] += [round_index]
            local_cache[b][slot]['mixing_record'] += str(a)
    # no model on slot
    else:
        local_cache[b][slot] =  {'model' : {a:model_list[a]},'time' : [round_index],'mixing_record':str(a)}
        print('since slot is empty put '+str(a)+' to '+str(b)+' on slot '+str(slot))


def update_model_cache_mixing_old_v2(local_cache, model_a,model_b,a,b,round_index,mixing_table):#####random method
    #update own model into cache
    # local_cache[a]['self'] = {'model' : {a:model_a},'time' : [round_index],'mixing_record':str(a)}
    # local_cache[b]['self'] = {'model' : {b:model_b},'time' : [round_index],'mixing_record':str(b)}
    old_local_cache_a = copy.deepcopy(local_cache[a])
    old_local_cache_b = copy.deepcopy(local_cache[b])
    temp_model_a = copy.deepcopy(model_a)
    temp_model_b = copy.deepcopy(model_b)



    #######################################################################
    #update newest model a b to related cache in slot
    #######################################################################

    # for cache a
    #check if there is already model existing
    slot = mixing_table[b]
    # already exist model on slot
    print('update based on model')
    
    if slot in local_cache[a]:
        # exist old model, evict all the model, replace with newest model
        if b in old_local_cache_a[slot]['mixing_record'] and round_index> old_local_cache_a[slot]['time_list'][b]:
            local_cache[a][slot] = {'model' : {b:temp_model_b},'time' : round_index,'time_list':{b:round_index},'mixing_record':[b]}
            print('replace '+str(b)+' to '+str(a)+' on slot '+str(slot))
        # no existed old model, add to mix
        else:
            print('mix '+str(b)+' to '+str(a)+' on slot '+str(slot))
            # print(local_cache[b][slot]['model'].keys())
            print('detail: '+str(local_cache[a][slot]['mixing_record'])+',whose content:'+str(local_cache[a][slot]['model'].keys())+' with '+str(b))
            local_cache[a][slot]['model'][b] = temp_model_b
            local_cache[a][slot]['time_list'][b] = round_index
            local_cache[a][slot]['mixing_record'] += [b]
    # no model on slot
    else:
        local_cache[a][slot] =  {'model' : {b:temp_model_b},'time' : round_index,'time_list':{b:round_index},'mixing_record':[b]}
        print('since slot is empty put '+str(b)+' to '+str(a)+' on slot '+str(slot))
        
    # for cache b
    #check if there is already model existing
    slot = mixing_table[a]
    # already exist model on slot
    if  slot in local_cache[b]:
        # exist old model, evict all the model, replace with newest model
        if a in old_local_cache_b[slot]['mixing_record'] and round_index> old_local_cache_b[slot]['time_list'][a]:
            local_cache[b][slot] = {'model' : {a:temp_model_a},'time' : round_index,'time_list':{a:round_index},'mixing_record':[a]}
            print('replace '+str(b)+' to '+str(a)+' on slot '+str(slot))
        # no existed old model, add to mix
        else:
            print('mix '+str(a)+' to '+str(b)+' on slot '+str(slot))
            print('detail: '+str(local_cache[b][slot]['mixing_record'])+',whose content:'+str(local_cache[b][slot]['model'].keys())+' with '+str(a))
            local_cache[b][slot]['model'][a]=temp_model_a
            local_cache[b][slot]['time_list'][a]= round_index
            local_cache[b][slot]['mixing_record'] += [a]
    # no model on slot
    else:
        local_cache[b][slot] =  {'model' : {a:temp_model_a},'time' : round_index,'time_list':{a:round_index},'mixing_record':[a]}
        print('since slot is empty put '+str(a)+' to '+str(b)+' on slot '+str(slot))


    #######################################################################
    #update cache based on other cache:
    #######################################################################
    for slot in local_cache[a]:
        if slot == mixing_table[b]:
            continue;
        ########### if same on slot a, b both have cached models
        elif slot in old_local_cache_b:
            if non_confliction_on_slot(local_cache[a],old_local_cache_b,slot):
                for key in old_local_cache_b[slot]['model']:
                    local_cache[a][slot]['model'][key] = old_local_cache_b[slot]['model'][key]
                print('mix '+str(b)+' to '+str(a)+' on slot '+str(slot))
                print('detail: '+str(old_local_cache_b[slot]['mixing_record'])+' with '+str(local_cache[a][slot]['mixing_record']))
                local_cache[a][slot]['time'] = min(old_local_cache_b[slot]['time'],local_cache[a][slot]['time'])
                local_cache[a][slot]['time_list'].update(old_local_cache_b[slot]['time_list'])
                local_cache[a][slot]['mixing_record'] += old_local_cache_b[slot]['mixing_record']
            elif old_local_cache_b[slot]['time']>local_cache[a][slot]['time']:
                local_cache[a][slot] = old_local_cache_b[slot].copy()
                print('replace '+str(b)+' to '+str(a)+' on slot '+str(slot))

    for slot in local_cache[b]:
        if slot == mixing_table[a]:
            continue;
        ########### if same on slot a, b both have cached models
        elif slot in old_local_cache_a:
            if non_confliction_on_slot(local_cache[b],old_local_cache_a,slot):
                for key in old_local_cache_a[slot]['model']:
                    local_cache[b][slot]['model'][key] = old_local_cache_a[slot]['model'][key]
                print('mix '+str(a)+' to '+str(b)+' on slot '+str(slot))
                print('detail: '+str(old_local_cache_a[slot]['mixing_record'])+' with '+str(local_cache[b][slot]['mixing_record']))
                local_cache[b][slot]['time'] = min(old_local_cache_a[slot]['time'],local_cache[b][slot]['time'])
                local_cache[b][slot]['time_list'].update(old_local_cache_a[slot]['time_list'])
                local_cache[b][slot]['mixing_record'] += old_local_cache_a[slot]['mixing_record']
            elif old_local_cache_a[slot]['time']>local_cache[b][slot]['time']:
                local_cache[b][slot] = old_local_cache_a[slot].copy()
                print('replace '+str(a)+' to '+str(b)+' on slot '+str(slot))


    # for slot in local_cache[a]:
    #     local_cache[a][slot]['time'] = sum(local_cache[a][slot]['time_list'].values())/len(local_cache[a][slot]['time_list'])
    # for slot in local_cache[b]:
    #     local_cache[b][slot]['time'] = sum(local_cache[b][slot]['time_list'].values())/len(local_cache[b][slot]['time_list'])

    for slot in local_cache[a]:
        local_cache[a][slot]['time'] = max(local_cache[a][slot]['time_list'].values())
    for slot in local_cache[b]:
        local_cache[b][slot]['time'] = max(local_cache[b][slot]['time_list'].values())
def remove_confliction(temp_list):
    i = 0
    while i < len(temp_list)-1:
        set1 = set(temp_list[i]['models'].keys())
        set2 = set(temp_list[i+1]['models'].keys())
        if set1.intersection(set2):
            if temp_list[i]['time'] > temp_list[i+1]['time']:
                temp_list.pop(i+1)
            else:
                temp_list.pop(i)
        else:
            i += 1
    return temp_list

def zip_slot(local_cache):
    zip_treshold = 100
    # zip the list to keep the slot balance, try to merge the slot with smallest number of model first
    zip_space = 0
    for i in range(len(local_cache)):
        if len(local_cache[i]['models'])<zip_treshold:
            zip_space += 1
    if zip_space >=2 :
        local_cache = sorted(local_cache, key = lambda x: len(x['models']))
        local_cache[0]['models'].update(local_cache[1]['models'])
        local_cache[0]['time'] = min(local_cache[0]['time'],local_cache[1]['time'])
        local_cache.pop(1)
    else:
        local_cache = sorted(local_cache, key = lambda x: x['time'])
        local_cache.pop(0)
    return local_cache


def update_model_cache_mixing(local_cache, model_a,model_b,a,b,round_index,slot_size): ####fold method, always want to keep the slot balance
    #update own model into cache
    # local_cache[a]['self'] = {'model' : {a:model_a},'time' : [round_index],'mixing_record':str(a)}
    # local_cache[b]['self'] = {'model' : {b:model_b},'time' : [round_index],'mixing_record':str(b)}
    # old_local_cache_a = copy.deepcopy(local_cache[a])
    # old_local_cache_b = copy.deepcopy(local_cache[b])
    temp_model_a = copy.deepcopy(model_a)
    temp_model_b = copy.deepcopy(model_b)



    #######################################################################
    #update newest model a b to related cache in slot
    #######################################################################

    # for cache a
    #check if there is already model existing
    # already exist model on slot
    # print('update based on model')
    # if b in local_cache[a]['model']:
    #     local_cache[a]['model'][b] = {'model' : temp_model_b,'time' : round_index}

    # # for cache a
    # #check if there is already model existing, if so delete it
    for i in range(len(local_cache[a])):
        if b in local_cache[a][i]['models']:
            local_cache[a].pop(i)
            break;  
    
    ## for cache b
    #check if there is already model existing, if so delete it
    for i in range(len(local_cache[b])):
        if a in local_cache[b][i]['models']:
            local_cache[b].pop(i)
            break;  

    # # for cache a
    # #check if there is already model existing
    # b_exist = False
    # for i in range(len(local_cache[a])):
    #     #if b already in cache a, replace it
    #     if b in local_cache[a][i]['models']:
    #         local_cache[a][i]['models'] = {b:{'model' : temp_model_b,'time' : round_index}}
    #         local_cache[a][i]['time'] = round_index
    #         print('replace '+str(b)+' to '+str(a)+' on slot '+str(i))
    #         b_exist = True
    #         break;
    # if not b_exist:
    #     local_cache[a].append({'models':{b:{'model' : temp_model_b,'time' : round_index}},'time':round_index})
    #     print('Add '+str(b)+' to '+str(a))

    # # for cache b
    # #check if there is already model existing
    # a_exist = False
    # for i in range(len(local_cache[b])):
    #     #if a already in cache b, replace it
    #     if a in local_cache[b][i]['models']:
    #         local_cache[b][i]['models'] = {a:{'model' : temp_model_a,'time' : round_index}}
    #         local_cache[b][i]['time'] = round_index
    #         print('replace '+str(a)+' to '+str(b)+' on slot '+str(i))
    #         a_exist = True
    #         break;
    # if not a_exist:
    #     local_cache[b].append({'models':{a:{'model' : temp_model_a,'time' : round_index}},'time':round_index})
    #     print('Add '+str(a)+' to '+str(b))

    #######################################################################
    #take out common items
    #######################################################################
    temp_list = []
    for i in range(len(local_cache[a])):
        if b not in local_cache[a][i]['models']:
            temp_list.append(local_cache[a][i])
    for i in range(len(local_cache[b])):
        if a not in local_cache[b][i]['models']:
            temp_list.append(local_cache[b][i])

    
    
    # sort the list based on time
    temp_list = sorted(temp_list, key = lambda x: x['time'])
    # check whether each pair of elements in the temp_list share the same keys in models, if so, keep the one with the latest time
    temp_list = remove_confliction(temp_list)


    ### put it back to local_cache:
    local_cache[a] = []
    local_cache[a].append({'models':{b:{'model' : temp_model_b,'time' : round_index}},'time':round_index})

    local_cache[b] = []
    local_cache[b].append({'models':{a:{'model' : temp_model_a,'time' : round_index}},'time':round_index})

    for i in range(len(temp_list)):
        local_cache[a].append(copy.deepcopy(temp_list[i]))
        local_cache[b].append(copy.deepcopy(temp_list[i]))

    #### zip the list to keep the slot size
    while len(local_cache[a])>slot_size:
        local_cache[a] = zip_slot(local_cache[a])

    while len(local_cache[b])>slot_size:
        local_cache[b] = zip_slot(local_cache[b])    


    # for slot in local_cache[a]:
    #     local_cache[a][slot]['time'] = max(local_cache[a][slot]['time_list'].values())
    # for slot in local_cache[b]:
    #     local_cache[b][slot]['time'] = max(local_cache[b][slot]['time_list'].values())


def update_model_cache(local_cache, model_a,model_b,a,b,round_index,cache_size, kick_out ):
    
    old_local_cache_a = copy.deepcopy(local_cache[a])
    old_local_cache_b = copy.deepcopy(local_cache[b])
    temp_model_a = copy.deepcopy(model_a)
    temp_model_b = copy.deepcopy(model_b)
    
    #update other's model into cache
    local_cache[a][b] = {'model' : temp_model_b,'time' : round_index}
    local_cache[b][a] = {'model' : temp_model_a,'time' : round_index}
    
    
    #update cache by fetching other's cache
    for key in old_local_cache_a:
        if key == b:
            continue
        if key not in local_cache[b]:
            local_cache[b][key] = old_local_cache_a[key].copy()
        elif local_cache[b][key]['time']<old_local_cache_a[key]['time']:
            local_cache[b][key] = old_local_cache_a[key].copy()
            
    for key in old_local_cache_b:
        if key == a:
            continue
        if key not in local_cache[a]:
            local_cache[a][key] = old_local_cache_b[key].copy()
        elif local_cache[a][key]['time']<old_local_cache_b[key]['time']:
            local_cache[a][key] = old_local_cache_b[key].copy()

    #kick out time-out model
    # if kick_out == True:
    #     local_cache[a] = kick_out_timeout_model(local_cache[a],round_index-cache_size)
    #     local_cache[b] = kick_out_timeout_model(local_cache[b],round_index-cache_size)
    
    
    #keep satisfying the cache size
    while len(local_cache[a])>cache_size:
        local_cache[a] = delete_smallest_value(local_cache[a],'time')
    while len(local_cache[b])>cache_size:
        local_cache[b] = delete_smallest_value(local_cache[b],'time')


def update_model_cache_car_to_car_p(local_cache, model_a,model_b,a,b,round_index,cache_size, kick_out, car_type_list,type_limits_car ):
    
    old_local_cache_a = copy.deepcopy(local_cache[a])
    old_local_cache_b = copy.deepcopy(local_cache[b])
    temp_model_a = copy.deepcopy(model_a)
    temp_model_b = copy.deepcopy(model_b)
    
    #update other's model into cache
    local_cache[a][b] = {'model' : temp_model_b,'time' : round_index, 'car_type': str(car_type_list[b]),'from':'car'}
    local_cache[b][a] = {'model' : temp_model_a,'time' : round_index, 'car_type': str(car_type_list[a]),'from':'car'}
    
    
    #update cache by fetching other's cache
    for key in old_local_cache_a:
        if key == b:
            continue
        if key not in local_cache[b]:
            local_cache[b][key] = old_local_cache_a[key].copy()
        elif local_cache[b][key]['time']<old_local_cache_a[key]['time']:
            local_cache[b][key] = old_local_cache_a[key].copy()
            
    for key in old_local_cache_b:
        if key == a:
            continue
        if key not in local_cache[a]:
            local_cache[a][key] = old_local_cache_b[key].copy()
        elif local_cache[a][key]['time']<old_local_cache_b[key]['time']:
            local_cache[a][key] = old_local_cache_b[key].copy()

    #kick out time-out model
    # if kick_out == True:
    #     local_cache[a] = kick_out_timeout_model(local_cache[a],round_index-cache_size)
    #     local_cache[b] = kick_out_timeout_model(local_cache[b],round_index-cache_size)
    
    

    #keep satisfying the cache size
    if len(local_cache[a])>cache_size:
        local_cache[a] = prune_cache(local_cache[a], type_limits_car, cache_size,'time','car_type')
    if len(local_cache[b])>cache_size:
        local_cache[b] = prune_cache(local_cache[b], type_limits_car, cache_size,'time','car_type')
    # print('local cache a length:'+str(len(local_cache[a])))
    # print('local cache b length:'+str(len(local_cache[b])))  

    # if len(local_cache[a])>cache_size:
    #     local_cache[a] = prune_cache(local_cache[a], type_limits_car, cache_size,'time','from')
    # if len(local_cache[b])>cache_size:
    #     local_cache[b] = prune_cache(local_cache[b], type_limits_car, cache_size,'time','from')


    
    

#     #keep satisfying the cache size
#     if len(local_cache[a])>cache_size:
#         local_cache[a] = prune_cache(local_cache[a], type_limits_car, cache_size,'time','from')
#     if len(local_cache[b])>cache_size:
#         local_cache[b] = prune_cache(local_cache[b], type_limits_car, cache_size,'time','from')

def update_model_cache_car_to_taxi(local_cache, model_car,car,taxi,round_index,cache_size_car,cache_size_taxi, kick_out):
    
    old_local_cache_car = copy.deepcopy(local_cache[car])
    old_local_cache_taxi = copy.deepcopy(local_cache[taxi])
    temp_model_car = copy.deepcopy(model_car)
    
    #update other's model into cache
    local_cache[taxi][car] = {'model' : temp_model_car,'time' : round_index}
    
    
    #update cache by fetching other's cache
    for key in old_local_cache_car:
        if key == taxi:
            continue
        if key not in local_cache[taxi]:
            local_cache[taxi][key] = old_local_cache_car[key].copy()
        elif local_cache[taxi][key]['time']<old_local_cache_car[key]['time']:
            local_cache[taxi][key] = old_local_cache_car[key].copy()
            
    for key in old_local_cache_taxi:
        if key == car:
            continue
        if key not in local_cache[car]:
            local_cache[car][key] = old_local_cache_taxi[key].copy()
        elif local_cache[car][key]['time']<old_local_cache_taxi[key]['time']:
            local_cache[car][key] = old_local_cache_taxi[key].copy()

    #kick out time-out model
    # if kick_out == True:
    #     local_cache[a] = kick_out_timeout_model(local_cache[a],round_index-cache_size)
    #     local_cache[b] = kick_out_timeout_model(local_cache[b],round_index-cache_size)
    
    
    #keep satisfying the cache size
    while len(local_cache[car])>cache_size_car:
        local_cache[car] = delete_smallest_value(local_cache[car],'time')
    while len(local_cache[taxi])>cache_size_taxi:
        local_cache[taxi] = delete_smallest_value(local_cache[taxi],'time')


def update_model_cache_car_to_taxi_p(local_cache, model_car,car,taxi,round_index,cache_size_car,cache_size_taxi, kick_out,car_type_list,type_limits_car,type_limits_taxi):
    
    old_local_cache_car = copy.deepcopy(local_cache[car])
    old_local_cache_taxi = copy.deepcopy(local_cache[taxi])
    temp_model_car = copy.deepcopy(model_car)
    
    #update other's model into cache
    local_cache[taxi][car] = {'model' : temp_model_car,'time' : round_index, 'car_type':car_type_list[car],'from':'car'}
    
    
    #update cache by fetching other's cache
    for key in old_local_cache_car:
        if key == taxi:
            continue
        if key not in local_cache[taxi]:
            local_cache[taxi][key] = old_local_cache_car[key].copy()
        elif local_cache[taxi][key]['time']<old_local_cache_car[key]['time']:
            local_cache[taxi][key] = old_local_cache_car[key].copy()
        local_cache[taxi][key]['from'] = 'car'

    for key in old_local_cache_taxi:
        if key == car:
            continue
        if key not in local_cache[car]:
            local_cache[car][key] = old_local_cache_taxi[key].copy()
        elif local_cache[car][key]['time']<old_local_cache_taxi[key]['time']:
            local_cache[car][key] = old_local_cache_taxi[key].copy()
        if car_type_list[key] != car_type_list[car]:
            local_cache[car][key]['from'] = 'taxi'
    #kick out time-out model
    # if kick_out == True:
    #     local_cache[a] = kick_out_timeout_model(local_cache[a],round_index-cache_size)
    #     local_cache[b] = kick_out_timeout_model(local_cache[b],round_index-cache_size)
    
    #keep satisfying the cache size
    if len(local_cache[car])>cache_size_car:
        # local_cache[car] = prune_cache(local_cache[car], type_limits_car, cache_size_car,'time','from')
        local_cache[car] = prune_cache(local_cache[car], type_limits_car, cache_size_car,'time','car_type')
    if len(local_cache[taxi])>cache_size_taxi:
        local_cache[taxi] = prune_cache(local_cache[taxi], type_limits_taxi, cache_size_taxi,'time','car_type')

def update_model_cache_taxi_to_taxi(local_cache, a,b,round_index,cache_size, kick_out):
    
    old_local_cache_a = copy.deepcopy(local_cache[a])
    old_local_cache_b = copy.deepcopy(local_cache[b])
    # temp_model_a = copy.deepcopy(model_a)
    # temp_model_b = copy.deepcopy(model_b)
    
    #update other's model into cache
    # local_cache[a][b] = {'model' : temp_model_b,'time' : round_index}
    # local_cache[b][a] = {'model' : temp_model_a,'time' : round_index}
    
    
    #update cache by fetching other's cache
    for key in old_local_cache_a:
        if key == b:
            continue
        if key not in local_cache[b]:
            local_cache[b][key] = old_local_cache_a[key].copy()
        elif local_cache[b][key]['time']<old_local_cache_a[key]['time']:
            local_cache[b][key] = old_local_cache_a[key].copy()
            
    for key in old_local_cache_b:
        if key == a:
            continue
        if key not in local_cache[a]:
            local_cache[a][key] = old_local_cache_b[key].copy()
        elif local_cache[a][key]['time']<old_local_cache_b[key]['time']:
            local_cache[a][key] = old_local_cache_b[key].copy()

    #kick out time-out model
    # if kick_out == True:
    #     local_cache[a] = kick_out_timeout_model(local_cache[a],round_index-cache_size)
    #     local_cache[b] = kick_out_timeout_model(local_cache[b],round_index-cache_size)
    
    
    #keep satisfying the cache size
    while len(local_cache[a])>cache_size:
        local_cache[a] = delete_smallest_value(local_cache[a],'time')
    while len(local_cache[b])>cache_size:
        local_cache[b] = delete_smallest_value(local_cache[b],'time')   

def update_model_cache_taxi_to_taxi_p(local_cache, a,b,cache_size, type_limits_taxi):
    
    old_local_cache_a = copy.deepcopy(local_cache[a])
    old_local_cache_b = copy.deepcopy(local_cache[b])
    # temp_model_a = copy.deepcopy(model_a)
    # temp_model_b = copy.deepcopy(model_b)
    
    #update other's model into cache
    # local_cache[a][b] = {'model' : temp_model_b,'time' : round_index}
    # local_cache[b][a] = {'model' : temp_model_a,'time' : round_index}
    
    
    #update cache by fetching other's cache
    for key in old_local_cache_a:
        if key == b:
            continue
        if key not in local_cache[b]:
            local_cache[b][key] = old_local_cache_a[key].copy()
        elif local_cache[b][key]['time']<old_local_cache_a[key]['time']:
            local_cache[b][key] = old_local_cache_a[key].copy()
            
    for key in old_local_cache_b:
        if key == a:
            continue
        if key not in local_cache[a]:
            local_cache[a][key] = old_local_cache_b[key].copy()
        elif local_cache[a][key]['time']<old_local_cache_b[key]['time']:
            local_cache[a][key] = old_local_cache_b[key].copy()

    #kick out time-out model
    # if kick_out == True:
    #     local_cache[a] = kick_out_timeout_model(local_cache[a],round_index-cache_size)
    #     local_cache[b] = kick_out_timeout_model(local_cache[b],round_index-cache_size)
    

    #keep satisfying the cache size
    if len(local_cache[a])>cache_size:
        local_cache[a] = prune_cache(local_cache[a], type_limits_taxi, cache_size,'time','car_type')
    if len(local_cache[b])>cache_size:
        local_cache[b] = prune_cache(local_cache[b], type_limits_taxi, cache_size,'time','car_type')

def update_model_cache_random(local_cache, model_a,model_b,a,b,round_index,cache_size):
    
    old_local_cache_a = copy.deepcopy(local_cache[a])
    old_local_cache_b = copy.deepcopy(local_cache[b])
    temp_model_a = copy.deepcopy(model_a)
    temp_model_b = copy.deepcopy(model_b)
    
    #update other's model into cache
    local_cache[a][b] = {'model' : temp_model_b,'time' : round_index}
    local_cache[b][a] = {'model' : temp_model_a,'time' : round_index}
    
    
    #update cache by fetching other's cache
    for key in old_local_cache_a:
        if key == b:
            continue
        if key not in local_cache[b]:
            local_cache[b][key] = old_local_cache_a[key].copy()
        elif local_cache[b][key]['time']<old_local_cache_a[key]['time']:
            local_cache[b][key] = old_local_cache_a[key].copy()
            
    for key in old_local_cache_b:
        if key == a:
            continue
        if key not in local_cache[a]:
            local_cache[a][key] = old_local_cache_b[key].copy()
        elif local_cache[a][key]['time']<old_local_cache_b[key]['time']:
            local_cache[a][key] = old_local_cache_b[key].copy()

    #kick out time-out model
    # if kick_out == True:
    #     local_cache[a] = kick_out_timeout_model(local_cache[a],round_index-cache_size)
    #     local_cache[b] = kick_out_timeout_model(local_cache[b],round_index-cache_size)
    
    
    #keep satisfying the cache size
    while len(local_cache[a])>cache_size:
        local_cache[a] = delete_random(local_cache[a])
    while len(local_cache[b])>cache_size:
        local_cache[b] = delete_random(local_cache[b])

def update_model_cache_global(local_cache, model_a,model_b,a,b,round_index,cache_size,cache_info,kick_out):
    
    old_local_cache_a = copy.deepcopy(local_cache[a])
    old_local_cache_b = copy.deepcopy(local_cache[b])
    temp_model_a = copy.deepcopy(model_a)
    temp_model_b = copy.deepcopy(model_b)
    
    #update other's model into cache
    if b in local_cache[a]:
        cache_info[b] -= 1
    if a in local_cache[b]:
        cache_info[a] -= 1
    local_cache[a][b] = {'model' : temp_model_b,'time' : round_index}
    cache_info[b] += 1
    local_cache[b][a] = {'model' : temp_model_a,'time' : round_index}
    cache_info[a] += 1
    
    #update cache by fetching other's cache
    for key in old_local_cache_a:
        if key == b:
            continue;
        if key not in local_cache[b]:
            local_cache[b][key] = old_local_cache_a[key].copy()
            cache_info[key] += 1
        elif local_cache[b][key]['time']<old_local_cache_a[key]['time']:
            local_cache[b][key] = old_local_cache_a[key].copy()
            
    for key in old_local_cache_b:
        if key == a:
            continue;
        if key not in local_cache[a]:
            local_cache[a][key] = old_local_cache_b[key].copy()
            cache_info[key] += 1
        elif local_cache[a][key]['time']<old_local_cache_b[key]['time']:
            local_cache[a][key] = old_local_cache_b[key].copy()

    #kick out time-out model
    if kick_out > 0 :
        local_cache[a], cache_info = kick_out_timeout_model_cache_info(local_cache[a],round_index-kick_out, cache_info)
        local_cache[b], cache_info = kick_out_timeout_model_cache_info(local_cache[b],round_index-kick_out, cache_info)
    #keep satisfying the cache size
    while len(local_cache[a])>cache_size:
        local_cache[a], cache_info = delete_cache_global(local_cache[a],cache_info)
    while len(local_cache[b])>cache_size:
        local_cache[b],cache_info = delete_cache_global(local_cache[b],cache_info)
    return cache_info


def update_model_cache_plus(local_cache, model_a,model_b, datapoint_a,datapoint_b,a,b,round_index,cache_size, kick_out):
    
    old_local_cache_a = copy.deepcopy(local_cache[a])
    old_local_cache_b = copy.deepcopy(local_cache[b])
    temp_model_a = copy.deepcopy(model_a)
    temp_model_b = copy.deepcopy(model_b)
    
    #update other's model into cache
    local_cache[a][b] = {'model' : temp_model_b,'time' : round_index,'datapoint':datapoint_b}
    local_cache[b][a] = {'model' : temp_model_a,'time' : round_index,'datapoint':datapoint_a}
    
    
    #update cache by fetching other's cache
    for key in old_local_cache_a:
        if key == b:
            continue;
        if key not in local_cache[b]:
            local_cache[b][key] = old_local_cache_a[key].copy()
        elif local_cache[b][key]['time']<old_local_cache_a[key]['time']:
            local_cache[b][key] = old_local_cache_a[key].copy()
            
    for key in old_local_cache_b:
        if key == a:
            continue;
        if key not in local_cache[a]:
            local_cache[a][key] = old_local_cache_b[key].copy()
        elif local_cache[a][key]['time']<old_local_cache_b[key]['time']:
            local_cache[a][key] = old_local_cache_b[key].copy()

    # #kick out time-out model
    # if kick_out == True:
    #     local_cache[a] = kick_out_timeout_model(local_cache[a],round_index-cache_size)
    #     local_cache[b] = kick_out_timeout_model(local_cache[b],round_index-cache_size)
    
    
    #keep satisfying the cache size
    while len(local_cache[a])>cache_size:
        local_cache[a] = delete_smallest_value(local_cache[a],'time')
    while len(local_cache[b])>cache_size:
        local_cache[b] = delete_smallest_value(local_cache[b],'time')
        
def update_model_cache_only_one_by_duration(local_cache,model_list,a,b,round_index,num_round,num_car,cache_size,expected_duration,duration):
    #update own model into cache
    target = num_round/num_car*cache_size
    # local_cache[a].clear()
    # local_cache[b].clear()
    local_cache[a][a] = {'model' : model_list[a],'time' : round_index}
    local_cache[b][b] = {'model' : model_list[b],'time' : round_index}
    
    local_cache[a][b] = {'model' : model_list[b],'time' : round_index}
    local_cache[b][a] = {'model' : model_list[a],'time' : round_index}
    
    if len(local_cache[a])>cache_size+1:
        max_key = -1
        max_value = 0
        for key in local_cache[a]:
            if key == a or key == b:
                continue;
            if expected_duration[round_index][a][key] + duration[a][key]< target:
                continue;
            if duration[a][key]> max_value:
                max_value = duration[a][key]
                max_key = key
        if max_key == -1:
            delete_smallest_value(local_cache[a],'time')
        else:
            del local_cache[a][max_key]
#         print(max_value,max_key)
    if len(local_cache[b])>cache_size+1:
        max_key = -1
        max_value = 0
        for key in local_cache[b]:
            if key == b or key == a:
                continue;
            if expected_duration[round_index][b][key] + duration[b][key]< target:
                continue;
            if  duration[b][key]> max_value:
                max_value = duration[b][key]
                max_key = key
        if max_key == -1:
            delete_smallest_value(local_cache[b],'time')
        else:
            print(max_key)
            print(local_cache[b])
            del local_cache[b][max_key]
#         print(max_value,max_key)

# def update_model_diag_fisher_cache(local_cache_model,model_list, diag_fisher_list,a,b,model_trace,round_index):
#     #update model_trace
#     model_trace[a][b] = round_index
#     model_trace[b][a] = round_index
#     model_trace[a][a] = round_index
#     model_trace[b][b] = round_index
#     #update cache based on other's model
#     local_cache_model[a][b] = copy.deepcopy(model_list[b])
#     local_cache_model[a][a] = copy.deepcopy(model_list[a])
#     local_cache_model[b][a] = copy.deepcopy(model_list[a])
#     local_cache_model[b][b] = copy.deepcopy(model_list[b])
#     #update diag_fisher cache based on other's model
#     diag_fisher_list[a][b] = get_diag_fisher_matrix(model_list[b],train_loader,b)
#     diag_fisher_list[b][a] = get_diag_fisher_matrix(model_list[a],train_loader,a)
#     diag_fisher_list[a][a] = diag_fisher_list[b][a]
#     diag_fisher_list[b][b] = diag_fisher_list[a][b]
#     #update model and fisher cache by fetching other's cache
#     for i in range(len(model_trace[a])):
#         if model_trace[a][i] > model_trace[b][i]:
#             model_trace[b][i] = model_trace[a][i]
#             local_cache_model[b][i] = copy.deepcopy(local_cache_model[a][i])
#             diag_fisher_list[b][i] = copy.deepcopy(diag_fisher_list[a][i])
            
#         elif model_trace[a][i] < model_trace[b][i]:
#             model_trace[a][i] = model_trace[b][i]ca
#             local_cache_model[a][i] = copy.deepcopy(local_cache_model[b][i])
#             diag_fisher_list[a][i] = copy.deepcopy(diag_fisher_list[b][i])

def solve_LP_alpha(model_time,combination):
    beta = 1 #parameter to control tradeoff between model coverage and freshness
    # Number of arrays and elements
    num_arrays, num_elements = combination.shape
    # Define the weights variable
    w = cp.Variable(num_arrays)
    # Calculate the weighted sum of arrays
    weighted_sums = combination.T @ w
    weighted_time_sum = model_time@w
    # Define S_max and S_min as variables
    S_max = cp.Variable()
    S_min = cp.Variable()
    
    # Objective: minimize S_max - S_min
    objective = cp.Minimize(beta*(S_max - S_min) + (1-beta)*weighted_time_sum)
    # Constraints
    # 1. Weights sum to 1
    # 2. Weights are non-negative
    # 3. For each element in weighted sums, it should be less than S_max and greater than S_min
    constraints = [cp.sum(w) == 1,
                   w >= 0,
                   weighted_sums <= S_max,
                   weighted_sums >= S_min]
    
    # Define and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()
    # Print the results
    print("Status:", prob.status)
    # print(combination.T @ w.value)
    if prob.status == 'optimal':
        alpha = w.value
    else: alpha = np.ones(len(model_time))
    return alpha 


def solve_LP_fresh(model_fresh_time,metric:str):
    # Number of arrays and elements
    num_arrays, num_elements = model_fresh_time.shape
    # Define the weights variable
    w = cp.Variable(num_arrays)
    # Calculate the weighted sum of arrays
    weighted_sums = model_fresh_time.T @ w
    
    if metric == 'min':
        # Define S_max and S_min as variables
        S_min = cp.Variable()
        
        # Objective: Maximize metric
        objective = cp.Maximize(S_min) 
        # Constraints
        # 1. Weights sum to 1
        # 2. Weights are non-negative
        # 3. For each element in weighted sums, it should be less than S_max and greater than S_min
        constraints = [cp.sum(w) == 1,
                       w >= 0,
                       weighted_sums >= S_min]
    elif metric == 'mean':
        
        # Objective: Maximize mean
        objective = cp.Maximize(weighted_sums.T @np.ones(num_elements)) 
        # Constraints
        # 1. Weights sum to 1
        # 2. Weights are non-negative
        # 3. For each element in weighted sums, it should be less than S_max and greater than S_min
        constraints = [cp.sum(w) == 1,
                       w >= 0]
    else: 
        print("Error! Please provide correct metric")
    
    # Define and solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()
    # Print the results
    print("Status:", prob.status)
    # print(combination.T @ w.value)
    if prob.status == 'optimal':
        alpha = w.value
    else: alpha = np.ones(len(model_fresh_time))
    return alpha 

def cache_average_process_fresh(model,i, local_cache, fresh_class_time_table, metric:str, full_weights_list):
    w=[]
    weight = []
    fresh_matrix = np.zeros([len(local_cache)+1,len(fresh_class_time_table)]) 
    # fresh_matrix = np.zeros([len(local_cache)+1,1])
    fresh_matrix[0] = fresh_class_time_table
    # fresh.append( fresh_class_time_table)
    # for key in local_cache:
    #     fresh.append(local_cache[key]['fresh'])
    
    # print(combination)
    # alpha = solve_LP_fresh(np.array(fresh),metric)
    alpha = np.ones(len(local_cache)+1)#/(len(local_cache)+1)
    # print(alpha)
    # alpha *= len(local_cache)+1
    # new_fresh = []
    new_fresh = np.ones(len(fresh_class_time_table))
    index = 0
    # new_fresh.append(fresh_class_time_table*alpha[index])
    w.append(model.state_dict())
    weight.append(full_weights_list[i])
    for key in local_cache:
        index += 1
        w.append(mul_weights(local_cache[key]['model'].state_dict(),alpha[index]))
        # new_fresh.append(local_cache[key]['fresh']*alpha[index])
        fresh_matrix[index] = local_cache[key]['class_fresh']
        weight.append(full_weights_list[key])
    new_w = average_weights(w,np.array(weight))
    # new_fresh = np.average(new_fresh, axis = 0)
    if metric == 'mean':
        for i in range(len(fresh_class_time_table)):
            new_fresh[i] = np.mean(fresh_matrix[:,i])
    elif metric == 'max':
        for i in range(len(fresh_class_time_table)):
            new_fresh[i] = np.max(fresh_matrix[:,i])
    else:
        print('Error! Please provide correct prompt!')
    model.load_state_dict(new_w)
    return model, new_fresh

def cache_average_process_fresh_v3(model, local_cache, fresh_class_time_table, metric:str):
    w=[]
    # fresh_matrix = np.zeros([len(local_cache)+1,len(fresh_class_time_table)]) 
    fresh_matrix = np.zeros(len(local_cache)+1)
    fresh_matrix[0] = fresh_class_time_table
    alpha = np.ones(len(local_cache)+1)#/(len(local_cache)+1)
    index = 0
    # new_fresh.append(fresh_class_time_table*alpha[index])
    w.append(model.state_dict())
    for key in local_cache:
        index += 1
        w.append(mul_weights(local_cache[key]['model'].state_dict(),alpha[index]))
        # new_fresh.append(local_cache[key]['fresh']*alpha[index])
        fresh_matrix[index] = local_cache[key]['fresh']
    
    new_w = average_weights(w)
    new_fresh = np.mean(fresh_matrix)
    model.load_state_dict(new_w)
    return model, new_fresh


def cache_average_process_fresh_without_model( local_cache, fresh_class_time_table, metric:str):
    fresh_matrix = np.zeros([len(local_cache)+1,len(fresh_class_time_table)])
    fresh_matrix[0] = fresh_class_time_table
    # fresh.append( fresh_class_time_table)
    # for key in local_cache:
    #     fresh.append(local_cache[key]['fresh'])
    
    # print(combination)
    # alpha = solve_LP_fresh(np.array(fresh),metric)
    alpha = np.ones(len(local_cache)+1)#/(len(local_cache)+1)
    print(alpha)
    # alpha *= len(local_cache)+1
    # new_fresh = []
    new_fresh = np.ones(len(fresh_class_time_table))
    index = 0
    # new_fresh.append(fresh_class_time_table*alpha[index])
    for key in local_cache:
        index += 1
        # new_fresh.append(local_cache[key]['fresh']*alpha[index])
        fresh_matrix[index] = local_cache[key]['fresh']
    
    # new_fresh = np.average(new_fresh, axis = 0)
    if metric == 'mean':
        for i in range(len(fresh_class_time_table)):
            new_fresh[i] = np.mean(fresh_matrix[:,i])
    elif metric == 'max':
        for i in range(len(fresh_class_time_table)):
            new_fresh[i] = np.max(fresh_matrix[:,i])
    else:
        print('Error! Please provide correct prompt!')
    return  new_fresh
    
def cache_average_process_combination(model,local_cache,current_model_time, current_model_combination):
    w=[]
    model_time = []
    combination = []
    model_time.append( current_model_time)
    combination.append( current_model_combination)
    for key in local_cache:
        model_time.append( local_cache[key]['time'])
        combination.append(local_cache[key]['combination'])
    
    # print(combination)
    alpha = solve_LP_alpha(np.array(model_time), np.array(combination))
    print(alpha)
    alpha *= len(combination)
    new_model_time = []
    new_combination = []
    index = 0
    new_model_time.append( current_model_time*alpha[index])
    new_combination.append( current_model_combination*alpha[index])
    w.append(model.state_dict())
    for key in local_cache:
        index += 1
        w.append(mul_weights(local_cache[key]['model'].state_dict(),alpha[index]))
        new_model_time.append(local_cache[key]['time']*alpha[index])
        new_combination.append(local_cache[key]['combination']*alpha[index])
    new_w = average_weights(w)
    new_model_time = np.average(new_model_time)
    new_combination = np.average(new_combination, axis = 0)
    model.load_state_dict(new_w)
    return model, new_model_time, new_combination

# def cache_average_process(model, i,current_round, local_cache, full_weight_list):
#     # w=[]
#     # weight = []
#     # w.append(model.state_dict())
#     # weight.append(full_weight_list[i])
#     total_weight = sum(full_weight_list[i])
#     normalized_weights = [x/total_weight for x in full_weight_list[i]]
#     avg_model = copy.deepcopy(model)
#     with torch.no_grad():
#         for param_name in avg_model.state_dict().keys():
#             avg_param = torch.zeros_like(avg_model.state_dict()[param_name])
#             for model, weight in zip(models, normalized_weights):
#                 avg_param += weight * model.state_dict()[param_name]
#             avg_model.state_dict()[param_name].copy_(avg_param)
    
#     for key in local_cache:
#         w.append(local_cache[key]['model'].state_dict())
#         weight.append(full_weight_list[key]*get_mixing_weight(current_round,local_cache[key]['time']))
#     new_w = average_weights(w,np.array(weight))
#     model.load_state_dict(new_w)
#     return model


def cache_average_process(model, i,current_round, local_cache, full_weight_list):
    w=[]
    weight = []
    w.append(model.state_dict())
    weight.append(full_weight_list[i])
    for key in local_cache:
        w.append(local_cache[key]['model'].state_dict())
        weight.append(full_weight_list[key]*get_mixing_weight(current_round,local_cache[key]['time']))
    new_w = average_weights(w,np.array(weight))
    model.load_state_dict(new_w)
    return model

def cache_average_process_mixing_old(model,i,local_cache,full_weight_list):
    w = []
    weight = []
    w.append(model.state_dict())
    weight.append(full_weight_list[i])
    for key in  local_cache:
        for cached_model in local_cache[key]['model']:
            w.append(local_cache[key]['model'][cached_model].state_dict())
            weight.append(full_weight_list[cached_model])
    new_w = average_weights(w,np.array(weight))
    model.load_state_dict(new_w)
    return model

def cache_average_process_mixing(model,i,local_cache,full_weight_list):
    w = []
    weight = []
    w.append(model.state_dict())
    weight.append(full_weight_list[i])
    for key in range(len(local_cache)):
        for cached_model in local_cache[key]['models']:
            w.append(local_cache[key]['models'][cached_model]['model'].state_dict())
            weight.append(full_weight_list[cached_model])
    new_w = average_weights(w,np.array(weight))
    model.load_state_dict(new_w)
    return model
# def cache_average_process_distribution(model, i,current_round, local_cache, full_weight_list,statisitc_data):
#     w=[]
#     weight = []
#     model_distribution = statisitc_data*full_weight_list[i]
#     w.append(model.state_dict())
#     weight.append(full_weight_list[i])
#     for key in local_cache:
#         w.append(local_cache[key]['model'].state_dict())
#         weight.append(full_weight_list[key]*get_mixing_weight(current_round,local_cache[key]['time']))
#         model_distribution += local_cache[key]['distribution']*full_weight_list[key]*get_mixing_weight(current_round,local_cache[key]['time'])
#     model_distribution = model_distribution/sum(weight)
#     new_w = average_weights(w,np.array(weight))
#     model.load_state_dict(new_w)
#     return model,model_distribution

def cache_average_process_plus(model, i,local_cache, full_weight_list):
    w=[]
    weight = []
    w.append(model.state_dict())
    weight.append(full_weight_list[i])
    for key in local_cache:
        w.append(local_cache[key]['model'].state_dict())
        weight.append(local_cache[key]['datapoint'])
    new_w = average_weights(w,np.array(weight))
    model.load_state_dict(new_w)
    return model

def weighted_cache_average_process(model,local_cache,round_index,p,beta):
    w=[]
    w.append(model.state_dict())
    for key in local_cache:
        delta_t = round_index-local_cache[key]['time']
        weight = (1-p)**delta_t
        w.append(div_weights(local_cache[key]['model'].state_dict(),weight))
    new_w = average_weights(w)
    model.load_state_dict(new_w)
    return model
    




def update_Subgradient_push_cache(local_cache, x_list,a,b,round_index,cache_size):
    #update own model into cache
    local_cache[a][a] = {'x' : x_list[a],'time' : round_index}
    local_cache[b][b] = {'x' : x_list[b],'time' : round_index}
    
    #update cache by fetching other's cache
    merged_cache =  merge_dictionaries(local_cache[a], local_cache[b])
    local_cache[a].clear()
    local_cache[b].clear()
    
    count=0
    for key in merged_cache:
        count+=1
        local_cache[a][key] = merged_cache[key].copy()
        local_cache[b][key] = merged_cache[key].copy()
        if count>=cache_size+1:
            break



def cache_sum_process_subgradient_push(model, local_cache, y, d, round_index,x):
    w = []
    y_new = 0
    for key in  local_cache:
        # w.append(div_weights(x[key],d[key]))
        w.append(div_weights(local_cache[key]['x'],d[key]))
        y_new += y[key]/d[key]
    tilde_w = sum_weights(w)
    new_w = div_weights(tilde_w,y_new)
    model.load_state_dict(new_w)
    return model, tilde_w, y_new

