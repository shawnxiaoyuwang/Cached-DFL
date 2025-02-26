# trainer_single_thread.py
# -*- coding: utf-8 -*-

import argparse
import random
import torch
import os
import copy
import numpy as np
import datetime
import time
from sklearn.metrics.pairwise import cosine_similarity
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Make sure your environment can handle potential MKL warnings:
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Local imports
from cache_algorithm import (
    kick_out_timeout_model,  
    update_model_cache_car_to_car_p,  update_model_cache_car_to_taxi_p,
    update_model_cache_taxi_to_taxi_p, update_model_cache_car_to_taxi, update_model_cache_taxi_to_taxi,
    update_model_cache_global, kick_out_timeout_model_cache_info,
    cache_average_process, 
    update_model_cache, update_model_cache_only_one,
)
from aggregation import (
    average_weights, normal_training_process, normal_train,
    weighted_average_process
)
from utils_cnn import test
from model import CNNMnist, CNNFashion_Mnist, ResNet18

from data_loader import (
    get_mnist_iid, get_mnist_area, get_mnist_dirichlet, get_mnist_non_iid,
    get_cifar10_iid,  get_cifar10_dirichlet, get_cifar10_non_iid
    get_fashionmnist_area, get_fashionmnist_iid,  get_fashionmnist_dirichlet, get_fashionmnist_non_iid
)
from road_sim import generate_roadNet_pair_area_list
import seed_setter

# Set random seeds for reproducibility
Randomseed = seed_setter.set_seed()

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)
device = torch.device("cuda" if cuda_available else "cpu")

np.set_printoptions(precision=4, suppress=True)

# --------------------------------------------------------------------------------
# ARGUMENT PARSING
# --------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Single-thread trainer script")

# Basic arguments
parser.add_argument("--suffix", type=str, default="", help="Suffix in the folder")
parser.add_argument("--note", type=str, default="N/A", help="Special notes")
parser.add_argument("--task", type=str, choices=['mnist', 'fashionmnist', 'cifar10', 'harbox'],
                    help="Dataset task to run")
parser.add_argument("--local_ep", type=int, default=10, help="Number of local epochs")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for non-iid data")
parser.add_argument("--distribution", type=str, choices=['iid', 'non-iid', 'dirichlet','area'],
                    help="Choose data distribution")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--hidden_size", type=int, default=200, help="Hidden layer size (if applicable)")
parser.add_argument("--num_round", type=int, default=2000, help="Number of rounds")
parser.add_argument("--early_stop_round", type=int, default=20,
                    help="Stop if test accuracy doesn't improve for this many rounds")
parser.add_argument("--speed", type=float, default=13.59, help="Speed in m/s")
parser.add_argument("--communication_distance", type=int, default=100, help="Comm distance in meters")
parser.add_argument("--epoch_time", type=int, default=60, help="Time to finish one epoch in seconds")
parser.add_argument("--num_car", type=int, default=100, help="Number of cars")
parser.add_argument("--lr_factor", type=float, default=0.1, help="ReduceLROnPlateau factor")
parser.add_argument("--lr_patience", type=int, default=20, help="ReduceLROnPlateau patience")
parser.add_argument("--cache_size", type=int, default=3, help="Cache size")
parser.add_argument("--augment", action='store_true', help="Enable data augmentation")
parser.add_argument('--no-augment', dest='augment', action='store_false')
parser.set_defaults(augment=True)
parser.add_argument("--shards_allocation", nargs='+', type=int,
                    default=[3,2,1,3,2,1,1,4,1,2]*10, help="Shards allocation for non-iid data")
parser.add_argument("--County", type=str, default="New York", help="County")
parser.add_argument('--kick_out', type=int, default=3, help='Threshold round to kick out from cache')
parser.add_argument("--weighted_aggregation", action='store_true', help="Enable weighted aggregation")
parser.add_argument('--no-weighted_aggregation', dest='weighted_aggregation', action='store_false')
parser.set_defaults(weighted_aggregation=True)

parser.add_argument("--algorithm", type=str, choices=[
    'ml', 'cfl', 'dfl',  'cache', 'test', 'test_area', 'test_area_GB'
], help="Algorithm to run")

args = parser.parse_args()






# --------------------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------------------

def write_info(write_dir):
    """
    Write run configuration info to a text file in write_dir.
    """
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    with open(os.path.join(write_dir, 'configuration.txt'), 'w') as file:
        file.write('Special suffix = ' + str(args.suffix) + '\n')
        file.write('Special Notes: ' + str(args.note) + '\n')
        file.write('Task: ' + str(args.task) + '\n')
        file.write('Start time: ' + str(date_time.strftime('%Y-%m-%d %H:%M:%S')) + '\n')
        file.write('Random Seed = ' + str(Randomseed) + '\n')
        file.write('local_ep = ' + str(args.local_ep) + '\n')
        file.write('lr = ' + str(args.lr) + '\n')
        file.write('batch_size = ' + str(args.batch_size) + '\n')
        file.write('hidden_size = ' + str(args.hidden_size) + '\n')
        file.write('lr_factor = ' + str(args.lr_factor) + '\n')
        file.write('lr_patience = ' + str(args.lr_patience) + '\n')
        file.write('num_round = ' + str(args.num_round) + '\n')
        file.write('num_car = ' + str(args.num_car) + '\n')
        file.write('epoch time = ' + str(args.epoch_time) + '\n')
        file.write('speed = ' + str(args.speed) + '\n')
        file.write('communication_distance = ' + str(args.communication_distance) + '\n')
        file.write('cache_size = ' + str(args.cache_size) + '\n')
        file.write('shards_allocation = ' + str(args.shards_allocation) + '\n')
        file.write('Aggregation weights = ' + str(weights) + '\n')
        file.write('County = ' + str(args.County) + '\n')
        file.write('kick_out = ' + str(args.kick_out) + '\n')
        file.write('alpha = ' + str(alpha) + '\n')
        file.write('Data distribution among cars:\n')
        file.write(str(statistic_data) + '\n')
        file.write('Data similarity among cars:\n')
        file.write(str(data_similarity) + '\n')
        file.write('Data_points:\n' + str(data_points) + '\n')

    # Generate pair & area from road network simulation
    pair, area = generate_roadNet_pair_area_list(
        write_dir, num_car, num_round, args.communication_distance,
        args.epoch_time, args.speed, args.County, 10, car_type_list
    )
    with open(os.path.join(write_dir, 'pair.txt'), 'w') as file:
        for i in range(num_round):
            file.write('Round:' + str(i) + ': \n')
            for j in range(args.epoch_time):
                file.write('Seconds:' + str(j) + ': \n')
                file.write(str(pair[i*args.epoch_time + j]) + '\n')
    with open(os.path.join(write_dir, 'area.txt'), 'w') as file:
        for i in range(num_car):
            file.write('Car:' + str(i) + ': ')
            file.write(str(area[i]) + '\n')
    return pair, area


def final_test(model, acc_list, class_acc_list):
    """
    Evaluate each model in 'model' on the global test_loader,
    storing accuracy and class-wise accuracy in acc_list & class_acc_list.
    """
    for i in range(len(model)):
        acc, class_acc = test(model[i], test_loader, num_class)
        acc_list[i].append(acc)
        class_acc_list[i].append(class_acc)


def final_test_process(model, process_index, result_list=None):
    """
    If you wanted to do parallel tests, you'd store results in result_list.
    For single-thread, you can ignore or call it inline.
    """
    result_list[process_index] = test(model, test_loader)


# --------------------------------------------------------------------------------
# FL WORKFLOW FUNCTIONS (Single-threaded)
# --------------------------------------------------------------------------------

def ml_process(suffix_dir, train_loader,test_loader, num_round,local_ep):
    """
    Simple single-process "centralized" ML training:
      - Train one model on the entire dataset (train_loader).
      - Evaluate on test_loader after each round.
    """
    model = copy.deepcopy(global_model)
    acc = []
    loss_history = []
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_factor,
                                  patience=args.lr_patience, verbose=False)
    learning_rate = lr
    model_dir = './result/{}_{}_{}_{}_ml{}'.format(
        date_time.strftime('%Y-%m-%d %H_%M_%S'), task, str(Randomseed),
        args.algorithm, suffix_dir
    )
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for i in range(num_round):
        print('Round:', i)
        for param_group in optimizer.param_groups:
            print("Current LR =", param_group['lr'])
        with open(os.path.join(model_dir, 'log.txt'), 'a') as file:
            file.write('Round: {}\n'.format(i))
            for param_group in optimizer.param_groups:
                file.write('Current LR = {}\n'.format(param_group['lr']))

        start_time = time.time()
        # Single-process local training
        for _ in tqdm(range(local_ep), disable=True):
            loss_val = normal_train(model, optimizer, full_loader)
            loss_history.append(loss_val)

        current_acc, class_acc = test(model, test_loader, num_class)
        acc.append(current_acc)

        end_time = time.time()
        print(f'{end_time - start_time:.2f} seconds for this epoch')
        print('Accuracy:', current_acc)
        print('Class accuracy:', class_acc)

        # use_lr_scheduler:
        scheduler.step(current_acc)

        # Log
        with open(os.path.join(model_dir, 'log.txt'), 'a') as file:
            file.write(f'{end_time - start_time:.2f} seconds for this epoch\n')
            file.write('Acc: {}\n'.format(current_acc))
            file.write('Class Acc: {}\n'.format(class_acc))
        with open(os.path.join(model_dir,
                  f'acc_{task}_{Randomseed}_{args.algorithm}{suffix_dir}.txt'),
                  'a') as file:
            file.write(f'{i}:{current_acc}\n')

        # Early stop if last X rounds' accuracy hasn't improved
        if i > early_stop_round and (
           abs(np.array(acc[-early_stop_round:]) - acc[-1]) < 1e-7
        ).all():
            print('Early stop at round:', i)
            with open(os.path.join(model_dir, 'log.txt'), 'a') as file:
                file.write('Early stop at round:{}\n'.format(i))
            break
    return loss_history, acc



def Centralized_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    """
    Standard Centralized FL (cooperative):
      - Each client trains locally -> all models are uploaded -> average -> broadcast back.
    """
    model = []
    test_acc = []
    class_acc_list = []
    loss_list = []
    optimizer = []
    scheduler_list = []
    learning_rate = lr

    model_dir = './result/{}_{}_{}_{}_cfl{}'.format(
        date_time.strftime('%Y-%m-%d %H_%M_%S'), task, distribution,
        Randomseed, suffix_dir
    )
    pair, area = write_info(model_dir)

    # Initialize each car's model/optimizer
    for _ in range(num_car):
        m = copy.deepcopy(global_model).to(device)
        opt = optim.SGD(params=m.parameters(), lr=lr)
        sched = ReduceLROnPlateau(opt, mode='max', factor=args.lr_factor,
                                  patience=args.lr_patience, verbose=False)
        model.append(m)
        optimizer.append(opt)
        scheduler_list.append(sched)
        loss_list.append([])

    for i in range(num_round):
        print('Round:', i)
        for param_group in optimizer[0].param_groups:
            print("Current LR =", param_group['lr'])
        with open(os.path.join(model_dir, 'log.txt'), 'a') as file:
            file.write('Round: {}\n'.format(i))
            for param_group in optimizer[0].param_groups:
                file.write('Current LR = {}\n'.format(param_group['lr']))

        start_time = time.time()

        # Local training on each device
        for idx in range(num_car):
            normal_training_process(model[idx], optimizer[idx],
                                    train_loader[idx], local_ep, loss_list[idx])

        # Aggregate
        w = [copy.deepcopy(m.state_dict()) for m in model]
        avg_w = average_weights(w, np.array(weights))
        for idx in range(num_car):
            model[idx].load_state_dict(copy.deepcopy(avg_w))
            model[idx].to(device)

        # Evaluate on a single "global" model (model[0] after averaging)
        acc, c_acc = test(model[0], test_loader)
        test_acc.append(acc)
        class_acc_list.append(c_acc)
        print('Test acc:', acc)
        print('Class acc:', c_acc)

        end_time = time.time()
        print(f'{end_time - start_time:.2f} seconds for this round')

        # Adjust LR
        for sch in scheduler_list:
            sch.step(acc)
        # Logging
        with open(os.path.join(model_dir, 'log.txt'), 'a') as file:
            file.write(f'{end_time - start_time:.2f} sec for this epoch\n')
            file.write('Test acc: {}\n'.format(acc))
            file.write('Class acc: {}\n'.format(c_acc))
        out_name = f'average_acc_{task}_{distribution}_{Randomseed}_{args.algorithm}{suffix_dir}.txt'
        with open(os.path.join(model_dir, out_name), 'a') as file:
            file.write(f'{i}:{acc}\n')

        # Early stop
        if i > early_stop_round and (
           abs(np.array(test_acc[-early_stop_round:]) - test_acc[-1]) < 1e-7
        ).all():
            print('Early stop at round:', i)
            with open(os.path.join(model_dir, 'log.txt'), 'a') as file:
                file.write('Early stop at round:{}\n'.format(i))
            break

    return loss_list, test_acc, model_dir

def Decentralized_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    """
    Basic Decentralized FL:
      - Each car trains locally
      - Pairwise model exchange/aggregation (graph-based) each round
    """
    model = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss_list = []
    optimizer = {}
    scheduler = {}
    learning_rate = lr

    model_dir = './result/{}_{}_{}_{}_{}'.format(
        date_time.strftime('%Y-%m-%d %H_%M_%S'), task, distribution,
        Randomseed, args.algorithm + suffix_dir
    )
    pair, area = write_info(model_dir)


    # Initialize
    for index in range(num_car):
        m = copy.deepcopy(global_model).to(device)
        model.append(m)
        optimizer[index] = optim.SGD(params=model[index].parameters(), lr=learning_rate)
        scheduler[index] = ReduceLROnPlateau(optimizer[index], mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=False)
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss_list.append([])

    for i in range(num_round):
        print('==================================================================')
        print('Round:', i)

        for param_group in optimizer[0].param_groups:
            print("Current LR =", param_group['lr'])

        with open(os.path.join(model_dir, 'log.txt'), 'a') as file:
            file.write(f'Round: {i}\n')
            for param_group in optimizer[0].param_groups:
                file.write('Current LR = {}\n'.format(param_group['lr']))

        start_time = time.time()

        # Local training
        for idx in range(num_car):
            normal_training_process(model[idx], optimizer[idx],
                                    train_loader[idx], local_ep, loss_list[idx])

        # Evaluate all models BEFORE aggregation
        model_before_aggregation = copy.deepcopy(model)
        final_test(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation)

        # Pairwise model exchange
        # pair[i] is a list of (a, b) pairs that meet in round i
        for (a, b) in pair[i]:
            weighted_average_process(model[a], model[b],
                                     np.array([weights[a], weights[b]]))
            model[a].to(device)
            model[b].to(device)

        end_time = time.time()

        # Evaluate all models AFTER aggregation
        final_test(model, acc_global, class_acc_list)

        print(f'{end_time - start_time:.2f} seconds for this round')
        print('Before/After aggregation acc:')
        for idx in range(num_car):
            bef = acc_global_before_aggregation[idx][-1]
            aft = acc_global[idx][-1]
            print(f'Car {idx} | {bef:.4f} -> {aft:.4f}')
            print('Class accuracy before:', class_acc_list_before_aggregation[idx][-1])
            print('Class accuracy after:', class_acc_list[idx][-1])

        avg_acc = np.average(acc_global, axis=0)[-1]
        print('Average test acc:', avg_acc)
        print('Pairs in this round:', pair[i])

        # update LR
        for index in range(num_car):
            scheduler[index].step(avg_acc)

        # Logging
        with open(os.path.join(model_dir, 'log.txt'), 'a') as file:
            file.write(f'{end_time - start_time:.2f} sec for this round\n')
            file.write('Before/After aggregation acc:\n')
            for idx in range(num_car):
                file.write(f'Car {idx} ----------------------------------\n')
                file.write(f'{acc_global_before_aggregation[idx][-1]} -> {acc_global[idx][-1]}\n')
                file.write(str(class_acc_list_before_aggregation[idx][-1]) + '\n')
                file.write(str(class_acc_list[idx][-1]) + '\n')
            file.write('Average test acc:' + str(avg_acc) + '\n')

        fn_name = f'average_acc_{task}_{distribution}_{Randomseed}_{args.algorithm}{suffix_dir}.txt'
        with open(os.path.join(model_dir, fn_name), 'a') as file:
            file.write(f'{i}:{avg_acc}\n')

        # Early stop
        if i > early_stop_round and (
           abs(np.average(acc_global, axis=0)[-early_stop_round:] - avg_acc) < 1e-7
        ).all():
            print('Early stop at round:', i)
            with open(os.path.join(model_dir, 'log.txt'), 'a') as file:
                file.write('Early stop at round:{}\n'.format(i))
            break

    return loss_list, acc_global, class_acc_list, acc_local, model_dir




def Decentralized_Cache_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    """
    Decentralized FL with a model cache that can hold up to 'cache_size' models for each node.
    """
    model = []
    local_cache = []
    acc_global = []
    class_acc_list = []
    acc_local = []
    loss_list = []
    optimizer = {}
    scheduler = {}
    learning_rate = lr

    # Example table if you need freshness metrics, etc.
    fresh_class_time_table = np.zeros([num_car, num_car])
    current_class_test = np.zeros([num_car, 10])

    model_dir = './result/{}_{}_{}_{}_{}_{}'.format(
        date_time.strftime('%Y-%m-%d %H_%M_%S'), task, distribution,
        Randomseed, args.algorithm, str(cache_size) + suffix_dir
    )
    pair, area = write_info(model_dir)

    # Init
    for i in range(num_car):
        model.append(copy.deepcopy(global_model).to(device))
        local_cache.append({})
        optimizer[i] = optim.SGD(params=model[j].parameters(), lr=learning_rate)
        scheduler[i] = ReduceLROnPlateau(optimizer[j], mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=False)
        acc_global.append([])
        class_acc_list.append([])
        acc_local.append([])
        loss_list.append([])

    for i in range(num_round):
        print('==================================================================')
        print('Round:', i)

        for param_group in optimizer[0].param_groups:
            print("Current LR =", param_group['lr'])
        with open(os.path.join(model_dir, 'log.txt'), 'a') as file:
            file.write(f'Round: {i}\n')
            for param_group in optimizer[0].param_groups:
                file.write('Current LR = {}\n'.format(param_group['lr']))

        start_time = time.time()

        # Local training for each car
        for index in range(num_car):
            normal_training_process(model[index], optimizer[index],
                                    train_loader[index], local_ep, loss_list[index])
            fresh_class_time_table[index][index] = i

        model_before_training = copy.deepcopy(model)

        # Exchange caches over each second in the round
        for seconds in range(args.epoch_time):
            for a, b in pair[i * args.epoch_time + seconds]:
                update_model_cache(local_cache, model_before_training[a],
                                   model_before_training[b], a, b, i,
                                   cache_size, kick_out)

        # After exchanging, do cache-based model aggregation
        for index in range(num_car):
            model[index] = cache_average_process(model[index], index, i,
                                                      local_cache[index], weights)
            model[index].to(device)

        # Evaluate
        final_test(model, acc_global, class_acc_list)
        end_time = time.time()

        # Logging / prints
        avg_acc = np.average(acc_global, axis=0)[-1]
        print(f'{end_time - start_time:.2f} seconds for this round')
        print('Average test acc:', avg_acc)
        with open(os.path.join(model_dir, 'log.txt'), 'a') as file:
            file.write('fresh_class_time_table\n')
            file.write(str(fresh_class_time_table) + '\n')
            for idx in range(num_car):
                file.write(f'{idx}:{class_acc_list[idx][-1]}\n')
            file.write(f'{end_time - start_time:.2f} sec this round\n')
            file.write('Average test acc:' + str(avg_acc) + '\n')

        fn_name = f'average_acc_{task}_{distribution}_{Randomseed}_{args.algorithm}_{cache_size}{suffix_dir}.txt'
        with open(os.path.join(model_dir, fn_name), 'a') as file:
            file.write(f'{i}:{avg_acc}\n')

        # Early stop
        if i > early_stop_round and (
           abs(np.average(acc_global, axis=0)[-early_stop_round:] - avg_acc) < 1e-7
        ).all():
            print('Early stop at round:', i)
            with open(os.path.join(model_dir, 'log.txt'), 'a') as file:
                file.write('Early stop at round:{}\n'.format(i))
            break

    return loss_list, acc_global, class_acc_list, acc_local, model_dir


def Decentralized_Cache_test_area():
    """
    Example test for taxi scenario. Simplified, no real training loop—just demonstrates
    how you might do repeated rounds of local training & cache updates in one thread.
    """
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    model_dir = './result/test/taxi'
    pair,area = write_info(model_dir)

    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=lr))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    cache_info_dynamic = np.ones([num_car])
    for i in range(num_round):
        # print('######################################################################')
        print('This is the round:',i)
        for ep in range(local_ep):
            start_time = time.time()
            
            model_before_training = copy.deepcopy(model)
            if kick_out == True:
                for index in range(num_car):
                    local_cache[index] = kick_out_timeout_model(local_cache[index],i-args.kick_out)
                    
            for seconds in range(args.epoch_time):
                for a,b in pair[i*args.epoch_time+seconds]: 
                    if car_type_list[a] == car_type_list[b] or car_type_list[a] == 0 or car_type_list[b] == 0: 
                        update_model_cache(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, kick_out)
            cache_info = np.zeros([num_car])
            for index in range(num_car):
                # cache_info_by_time[0][index] += 1 
                cache_info[index] += 1
                for key in local_cache[index]:
                    # print(local_cache[index][key]['time'])
                    cache_info[key] += 1
            cache_age = 0
            cache_num = 0
            for index in range(num_car):
                cache_num += len(local_cache[index])
                for key in local_cache[index]:
                    cache_age += i-local_cache[index][key]['time']
            avg_cache_age = cache_age/cache_num
            with open(model_dir+'/cache_age_cache_num_'+str('cache' )+'_'+str(cache_size)+'_'+str(args.epoch_time)+'_'+str(args.kick_out)+'.txt','a') as file:
                file.write(str(i)+':')
                file.write(str(avg_cache_age)+'\t'+str(cache_num/num_car)+'\n')


        end_time = time.time()
        
    
    return loss,[0,0,0],class_acc_list, acc_local 

def Decentralized_Cache_test_area_GB():
    """
    Another scenario test. Single-thread approach (no mp).
    """
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    model_dir = './result/test/taxi'
    pair,area = write_info(model_dir)

    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=lr))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    cache_info_dynamic = np.ones([num_car])
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        # print('######################################################################')
        print('This is the round:',i)
        for ep in range(local_ep):

            model_before_training = copy.deepcopy(model)
            if kick_out == True:
                for index in range(num_car):
                    local_cache[index] = kick_out_timeout_model(local_cache[index],i-args.kick_out)
                    
            for seconds in range(args.epoch_time):
                for a,b in pair[i*args.epoch_time+seconds]: 
                    if car_type_list[a] == car_type_list[b] or car_type_list[a] == 0 or car_type_list[b] == 0: 
                        update_model_cache_car_to_car_p(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, kick_out,car_area_list, type_limits_car)
                    # if car_type_list[a] == car_type_list[b]: 
                    #     if car_type_list[a] != 0:
                    #         update_model_cache_car_to_car_p(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, kick_out,car_type_list, type_limits_car)
                    #     else:
                    #         update_model_cache_taxi_to_taxi_p(local_cache, a,b,i, cache_size, kick_out,car_type_list,type_limits_taxi)
                    # elif car_type_list[a] == 0:
                    #     update_model_cache_car_to_taxi_p(local_cache, model_before_training[b],b,a,i, cache_size, cache_size, kick_out,car_type_list,type_limits_car,type_limits_taxi)
                    # elif car_type_list[b] == 0:
                    #     update_model_cache_car_to_taxi_p(local_cache, model_before_training[a],a,b,i, cache_size, cache_size, kick_out,car_type_list,type_limits_car,type_limits_taxi)
                    # update_model_cache_distribution(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, args.kick_out,statistic_data,max_std,0.9)
            # for key in local_cache[0]:
            #     print(key,local_cache[0][key]['time'],local_cache[0][key]['distribution'],local_cache[0][key]['cache_score'])
            cache_info = np.zeros([num_car])
            for index in range(num_car):
                # cache_info_by_time[0][index] += 1 
                cache_info[index] += 1
                for key in local_cache[index]:
                    # print(local_cache[index][key]['time'])
                    cache_info[key] += 1
            cache_age = 0
            cache_num = 0
            for index in range(num_car):
                cache_num += len(local_cache[index])
                for key in local_cache[index]:
                    cache_age += i-local_cache[index][key]['time']
            avg_cache_age = cache_age/cache_num
            with open(model_dir+'/cache_age_cache_num_'+str('cache' )+'_'+str(cache_size)+'_'+str(args.epoch_time)+'_'+str(args.kick_out)+'.txt','a') as file:
                file.write(str(i)+':')
                file.write(str(avg_cache_age)+'\t'+str(cache_num/num_car)+'\n')
            with open(model_dir+'/log.txt','a') as file:
            # file.write('fresh_class_time_table\n')
            # file.write(str(fresh_class_time_table)+'\n')
                # for index in range(num_car):
                #     file.write(str(index)+':'+str(class_acc_list[index][-1])+'\n')
                # file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
                # file.write('Before/After aggregation acc:'+'\n')
                for index in range(num_car):
                    file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                    # file.write(str(acc_global[index][-1])+'\n')
                    # # file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                    # # file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                    # file.write(str(class_acc_list[index][-1])+'\n')
                    file.write('Local Cache model version:'+'\n')
                    for key in local_cache[index]:
                        file.write(str(key)+':'+str(local_cache[index][key]['time'])+'\t'+str(local_cache[index][key]['car_type'])+'\n')#,local_cache[index][key]['fresh_metric'])

    return loss,[0,0,0],class_acc_list, acc_local 

def Decentralized_Cache_test():
    """
    Another test function that loops but doesn't do real multi-threading.
    """
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = []
    model_dir = './result/test/cache'
    pair,area = write_info(model_dir)

    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        optimizer.append(optim.SGD(params=model[i].parameters(), lr=lr))
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for i in range(10):
        local_cache.append({})
    cache_info_dynamic = np.ones([num_car])
    for i in range(num_round):
        # old_model = copy.deepcopy(model)
        # print('######################################################################')
        print('This is the round:',i)
        for ep in range(local_ep):
            start_time = time.time()
            
            model_before_training = copy.deepcopy(model)
            if kick_out == True:
                for index in range(num_car):
                    local_cache[index] = kick_out_timeout_model(local_cache[index],i-args.kick_out)
                    
            for seconds in range(args.epoch_time):
                for a,b in pair[i*args.epoch_time+seconds]: 
                    update_model_cache(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, kick_out)
            cache_info = np.zeros([num_car])
            for index in range(num_car):
                # cache_info_by_time[0][index] += 1 
                cache_info[index] += 1
                for key in local_cache[index]:
                    # print(local_cache[index][key]['time'])
                    cache_info[key] += 1
            cache_age = 0
            cache_num = 0
            for index in range(num_car):
                cache_num += len(local_cache[index])
                for key in local_cache[index]:
                    cache_age += i-local_cache[index][key]['time']
            avg_cache_age = cache_age/cache_num
            with open(model_dir+'/cache_age_cache_num_'+str('cache' )+'_'+str(cache_size)+'_'+str(args.epoch_time)+'_'+str(args.kick_out)+'.txt','a') as file:
                file.write(str(i)+':')
                file.write(str(avg_cache_age)+'\t'+str(cache_num/num_car)+'\n')
        end_time = time.time()

    return loss,[0,0,0],class_acc_list, acc_local 

# --------------------------------------------------------------------------------
# MAIN SCRIPT
# --------------------------------------------------------------------------------

if __name__ == '__main__':
    # Expose some parsed arguments as simpler variables
    task = args.task
    distribution = args.distribution
    local_ep = args.local_ep
    lr = args.lr
    alpha = args.alpha
    batch_size = args.batch_size
    num_round = args.num_round
    early_stop_round = args.early_stop_round
    num_car = args.num_car
    cache_size = args.cache_size
    kick_out = (args.kick_out is not None and args.kick_out > 0)
    data_distribution = distribution
    date_time = datetime.datetime.now()

    hidden_size = args.hidden_size
    speed = args.speed
    communication_distance = args.communication_distance
    shards_allocation = args.shards_allocation
    County = args.County
    suffix = args.suffix
    special_notes = args.note


    task = 'fashionmnist'
    distribution = 'area'
    args.algorithm = 'test_area_GB'
    # distribution = 'iid'
    # args.algorithm = 'test_mixing'
    # args.algorithm = 'cache'
    cache_size = 3
    kick_out = True
    args.kick_out = 5
    if distribution == 'area':
        kick_out = False
    args.epoch_time = 120
    num_car = 100
    num_round = 1000

    balance = False
    test_ratio = 0.1
    data_distribution = distribution

    # If some tasks require typed lists for car_type_list, etc.:
    car_type_list =  [0]*num_car
    type_limits_taxi = {'1':4,'2':3,'3':3}
    type_limits_car = {'1':4,'2':3,'3':3}
    # target_labels = [[0,1,2,3],[4,5,6],[7,8,9]]
    target_labels = [[9,0,1,2],[3,4,5,6],[6,7,8,9]]
    # target_labels = [[7,8,9,0,1,2,3],[1,2,3,4,5,6],[4,5,6,7,8,9]]
    # type_limits_car = {'taxi':5,'car':5}

    # Basic assumption for test splits
    test_ratio = 0.1

    # Pick or build the global model
    if task == 'mnist':
        global_model = CNNMnist(1, 10)
        num_class = 10
        # Load data
        if distribution == 'iid':
            train_loader, sub_test_loader, test_loader, full_loader = \
                get_mnist_iid(num_car, batch_size, test_ratio)
        elif distribution == 'non-iid':
            train_loader, sub_test_loader, test_loader, full_loader = \
                get_mnist_non_iid(args.shards_allocation, num_car, batch_size, test_ratio)
        elif distribution == 'dirichlet':
            train_loader, sub_test_loader, test_loader, full_loader = \
                get_mnist_dirichlet(alpha, num_car, batch_size, test_ratio)
            data_distribution += f'_{alpha}'
        elif distribution == 'area':
            # Example usage
            car_area_list = [1]*34 + [2]*33 + [3]*33  # total 100
            # Shards or target labels example:
            target_labels = [[9, 0, 1, 2], [3, 4, 5, 6], [6, 7, 8, 9]]
            train_loader, sub_test_loader, test_loader, full_loader = \
                get_mnist_area(args.shards_allocation, batch_size, test_ratio,
                                    car_area_list, target_labels)
        else:
            raise ValueError('Unsupported MNIST distribution')

    elif task == 'cifar10':
        global_model = ResNet18()
        num_class = 10
        if distribution == 'iid':
            train_loader, sub_test_loader, test_loader, full_loader = \
                get_cifar10_iid(num_car, batch_size, test_ratio)
        elif distribution == 'non-iid':
            train_loader, sub_test_loader, test_loader, full_loader = \
                get_cifar10_non_iid(args.shards_allocation, num_car, batch_size, test_ratio)
        elif distribution == 'dirichlet':
            train_loader, sub_test_loader, test_loader, full_loader = \
                get_cifar10_dirichlet(alpha, num_car, batch_size, test_ratio)
            data_distribution += f'_{alpha}'
        else:
            raise ValueError('Unsupported CIFAR-10 distribution')

    elif task == 'fashionmnist':
        global_model = CNNFashion_Mnist()
        num_class = 10
        if distribution == 'iid':
            train_loader, sub_test_loader, test_loader, full_loader = \
                get_fashionmnist_iid(num_car, batch_size, test_ratio)
        elif distribution == 'non-iid':
            train_loader, sub_test_loader, test_loader, full_loader = \
                get_fashionmnist_non_iid(args.shards_allocation, num_car, batch_size, test_ratio)
        elif distribution == 'dirichlet':
            train_loader, sub_test_loader, test_loader, full_loader = \
                get_fashionmnist_dirichlet(alpha, num_car, batch_size, test_ratio)
            data_distribution += f'_{alpha}'
        elif distribution == 'area':
            # Example usage
            car_area_list = [1]*34 + [2]*33 + [3]*33
            target_labels = [[9, 0, 1, 2], [3, 4, 5, 6], [6, 7, 8, 9]]
            train_loader, sub_test_loader, test_loader, full_loader = \
                get_fashionmnist_area(args.shards_allocation, batch_size,
                                           test_ratio, car_area_list, target_labels)
        else:
            raise ValueError('Unsupported FashionMNIST distribution')

    else:
        raise ValueError('Error: task must be mnist, cifar10, fashionmnist, or harbox')

    global_model.to(device)

    # --------------------------------------------------------------------------------
    # Prepare data statistics
    # --------------------------------------------------------------------------------
    statistic_data = np.zeros([num_car, num_class])
    for i in range(num_car):
        for _, target in train_loader[i]:
            for t in target:
                statistic_data[i][t] += 1
    print('Data distribution among cars:')
    print(statistic_data)
    max_std = np.zeros(num_car)
    for i in range(num_car):
        for j in range(num_car):
            var_ij = np.var(statistic_data[i] - statistic_data[j])
            if var_ij > max_std[i]:
                max_std[i] = var_ij
    data_points = np.sum(statistic_data, axis=1)
    print("Total Data Points:", sum(data_points))
    print("Data points fraction per car:", data_points / sum(np.array(data_points)))
    if args.weighted_aggregation:
        weights = data_points
    else:
        weights = [1]*num_car

    data_similarity = cosine_similarity(statistic_data)
    print("Data similarity (cosine):")
    print(data_similarity)


    # --------------------------------------------------------------------------------
    # MAIN EXECUTION: choose the algorithm
    # --------------------------------------------------------------------------------
    start_time = time.time()

    if args.algorithm == 'ml':
        # Single-model approach
        loss_vals, acc_vals = ml_process(args.suffix, train_loader, test_loader, num_round, local_ep)
        print("Final accuracy array:", acc_vals)

    elif args.algorithm == 'cfl':
        fl_loss, fl_test_acc, model_dir = Centralized_process(
            args.suffix, train_loader, test_loader, num_round, local_ep
        )
        print("Final FL test acc array:", fl_test_acc)

    elif args.algorithm == 'dfl':
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir = Decentralized_process(
            args.suffix, train_loader, test_loader, num_round, local_ep
        )
        # Print average accuracy across all cars
        final_avg_acc = np.average(dfl_acc_global, axis=0)
        print("DFL average accuracy:", final_avg_acc)

    elif args.algorithm == 'cache':
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir = Decentralized_Cache_process(
            args.suffix, train_loader, test_loader, num_round, local_ep
        )
        final_avg_acc = np.average(dfl_acc_global, axis=0)
        print("cache average accuracy:", final_avg_acc)

    elif args.algorithm == 'test':
        # Example placeholder
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local = Decentralized_Cache_test()
        print("Test Mode - dfl_acc_global:", dfl_acc_global)

    elif args.algorithm == 'test_area':
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local = Decentralized_Cache_test_area()
        print("Test Taxi Mode - dfl_acc_global:", dfl_acc_global)

    elif args.algorithm == 'test_area_GB':
        dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local = Decentralized_Cache_test_area_GB()
        print("Test Taxi Priority - dfl_acc_global:", dfl_acc_global)

    else:
        raise ValueError("Unknown algorithm choice.")

    end_time = time.time()
    total_training_time = end_time - start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    # If we used a real model_dir in some approaches, you can store final summary:
    # (Below lines show the pattern—uncomment or modify as desired if your algorithm sets model_dir)
    #
    # with open(os.path.join(model_dir, 'summary.txt'), 'w') as file:
    #     file.write('Start time: ' + str(date_time.strftime('%Y-%m-%d %H:%M:%S')) + '\n')
    #     date_time_end = datetime.datetime.fromtimestamp(end_time)
    #     file.write('End time: ' + str(date_time_end.strftime('%Y-%m-%d %H:%M:%S')) + '\n')
    #     file.write(f'Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s\n')
    #     # Optionally log final accuracy across cars for the DFL variants, etc.
     
    
    
