# -*- coding: utf-8 -*-
# import matplotlib.pyplot as plt

# plt.style.use("seaborn-white")
from mpi4py import MPI
import pickle
from mpi4py.util import pkl5
import argparse
import random
import torch
from utils_cnn import variable
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch import optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import copy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import datetime
import math

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

Randomseed = seed_setter.set_seed()
from torch.utils.data import DataLoader, TensorDataset
np.set_printoptions(precision=4, suppress=True)



# Create the parser
parser = argparse.ArgumentParser(description="Configure script parameters and select algorithm")

# Add arguments
parser.add_argument("--suffix", type=str, default="", help="Suffix in the folder")
parser.add_argument("--note", type=str, default="N/A", help="Special_notes")
parser.add_argument("--task", type=str, choices=[
    'mnist', 'fashionmnist', 'cifar10','cifar100','harbox'], help="Choose dataset task to run")
parser.add_argument("--local_ep", type=int, default=1, help="Number of local epochs")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
parser.add_argument("--alpha", type=float, default=0.5, help="Alpha for Dirchlet distribution, lower alpha increases heterogeneity")
parser.add_argument("--distribution", type=str, choices=[
    'iid', 'non-iid','dirichlet','area'], help="Choose data distirbution")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--num_round", type=int, default=1000, help="Number of rounds")
parser.add_argument("--early_stop_round", type=int, default=20, help="Number of rounds test accuracy remain unchanged then stop")
parser.add_argument("--speed", type=float, default=13.59, help="Speed in m/s")
parser.add_argument("--communication_distance", type=int, default=100, help="Communication distance in meters")
parser.add_argument("--epoch_time", type=int, default=120, help="Time to finish on epoch in seconds")
parser.add_argument('--kick_out', type= int, default = 5, help =  'Threshold to kick out in cache')
parser.add_argument("--num_car", type=int, default=100, help="Number of cars")
parser.add_argument("--lr_factor", type=float, default=0.1, help="Learning rate factor for ReduceLROnPlateau")
parser.add_argument("--lr_patience", type=int, default=20, help="Learning rate patience for ReduceLROnPlateau")
parser.add_argument("--cache_size", type=int, default=3, help="Cache size")
parser.add_argument("--overlap", type=int, default=0, help="date overlap in area distribution")
parser.add_argument("--test_ratio", type=float, default=1.0, help="ratio to take the subset of the testset for the testing")
parser.add_argument("--shards_allocation", nargs='+', type=int, default=[3,2,1,3,2,1,1,4,1,2]*10, help="Shards allocation")
parser.add_argument("--County", type=str, default="New York", help="County")
parser.add_argument("--weighted_aggregation", action='store_true', help="Enable weighted aggregation")
parser.add_argument('--no-weighted_aggregation', dest='weighted_aggregation', action='store_false')
parser.set_defaults(weighted_aggregation=True)
parser.add_argument("--algorithm", type=str, choices=[
    'ml', 'cfl', 'dfl', 'cache', 'cache_areas_LRU','cache_areas_GB'
], help="Choose the algorithm to run")
# Parse arguments
args = parser.parse_args()
task = args.task
# Assign values to variables
if args.test_ratio<1.0:
    test_ratio = args.test_ratio
else:
    test_ratio = None
local_ep = args.local_ep
lr = args.lr
alpha = args.alpha
distribution = args.distribution
batch_size = args.batch_size
num_round = args.num_round
speed = args.speed
communication_distance = args.communication_distance
early_stop_round = args.early_stop_round
num_car = args.num_car
cache_size = args.cache_size
shards_allocation = args.shards_allocation
County = args.County
if args.kick_out > 0:
    kick_out = True
else:
    kick_out = False
suffix = args.suffix
special_notes = args.note

def write_info(write_dir):
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    if not os.path.exists(write_dir+'/rank_log'):
        os.makedirs(write_dir+'/rank_log')
    with open(write_dir+'/configuration.txt','w') as file:
        file.write('Special suffix = '+str(suffix)+'\n')
        file.write('Special Notes: '+str(special_notes)+'\n')
        file.write('Task: '+str(task)+'\n')
        file.write('Start time: ' + str(date_time.strftime('%Y-%m-%d %H:%M:%S'))+'\n')
        file.write('Random Seed = '+str(Randomseed)+'\n')
        file.write('local_ep = '+str(local_ep)+'\n')
        file.write('lr = '+str(lr)+'\n')
        file.write('batch_size = '+str(batch_size)+'\n')
        file.write('lr_factor = '+str(args.lr_factor)+'\n')
        file.write('lr_patience = '+str(args.lr_patience)+'\n')
        file.write('num_round = '+str(num_round)+'\n')
        file.write('num_car = '+str(args.num_car)+'\n')
        file.write('speed = '+str(speed)+'\n')
        file.write('communication_distance = '+str(communication_distance)+'\n')
        file.write('epoch_time = '+str(args.epoch_time)+'\n')
        file.write('cache_size = '+str(cache_size)+'\n')
        # file.write('parallel = '+str(parallel)+'\n')
        file.write('shards_allocation = '+str(shards_allocation)+'\n')
        file.write('Aggregation weights = '+str(weights)+'\n')
        file.write('County = '+str(County)+'\n')
        file.write('kick_out = '+str(args.kick_out)+'\n')
        file.write('Test_ratio = '+str(args.test_ratio)+'\n')
        file.write('Test size = '+str(len(test_loader.dataset))+'\n')
        file.write('Use ' +str(torch.cuda.device_count())+ 'GPUs!'+'\n')
        file.write('Rank '+str(rank)+ 'uses GPU' +str(device)+'\n')
        if distribution == 'area':
            file.write('Car type list:\n')
            file.write(str(car_type_list)+'\n')
            file.write('Car area list:\n')
            file.write(str(car_area_list)+'\n')
            file.write('Target labels:\n')
            file.write(str(target_labels)+'\n')
            file.write('Taxi type limits:\n')
            file.write(str(type_limits_taxi)+'\n')
            file.write('Car type limits:\n')
            file.write(str(type_limits_car)+'\n')
            file.write('overlap'+str(args.overlap)+'\n')
        file.write('alpha = '+str(alpha)+'\n')
        file.write('Data distribution among cars:\n')
        file.write(str(statistic_data)+'\n')
        file.write('Data similarity among cars:\n')
        file.write(str(data_similarity)+'\n')
        file.write('Data_points:\n'+str(data_points)+'\n')

    pair, area = generate_roadNet_pair_area_list(write_dir,args.num_car, num_round,communication_distance,args.epoch_time,speed,County,10,car_type_list)
    with open(write_dir+'/pair.txt','w') as file:
        for i in range(num_round):
            file.write('Round:'+str(i)+': \n')
            for j in range(args.epoch_time):
                file.write('Seconds:'+str(j)+': \n')
                file.write(str(pair[i*args.epoch_time+j])+'\n')
    with open(write_dir+'/area.txt','w') as file:
        for i in range(num_car):
            file.write('Car:'+str(i)+': ')
            file.write(str(area[i])+'\n')
    return pair, area
            
# Example: serialize a PyTorch model
def serialize_model(model):
    return pickle.dumps(model)

# Example: deserialize the byte stream back to a PyTorch model
def deserialize_model(byte_stream):
    return pickle.loads(byte_stream)

def final_test(model_list, acc_list,class_acc_list):
    for i in range(len(model_list)):
        acc, class_acc = test(model_list[i], test_loader,num_class)
        acc_list[i].append(acc)
        class_acc_list[i].append(class_acc)

def wait_compute(model,data_loader,num_test):
    for i,(input, target) in enumerate(data_loader):
        if i >= num_test:
            break
        input, target = variable(input), variable(target)
        #target = F.one_hot(target,num_classes = 10).to(torch.float)
        output = model(input)

def mpi_send_model_host(model_list):
    for i in range(1,size):
        for j in client_rank_mapping[i]:
            if async_comm:
                comm.isend(serialize_model(model_list[j].state_dict()), dest=i, tag=11)
            elif high_IO:
                torch.save( model_list[j].state_dict(),high_IO_dir+'/model_'+str(j)+'.pt')
                comm.send(str('MPI_SEND_Model success'), dest=i, tag=11)
            else:
                comm.send(serialize_model(model_list[j].state_dict()), dest=i, tag=11)
        
def mpi_receive_model_host(model_list):
    for i in range(1,size):
        for index in range(len(client_rank_mapping[i])):
            j = client_rank_mapping[i][index]
            if async_comm:
                req = comm.irecv(source = i, tag = 12)
                while not req.Test():
                    #perfom some computation
                    wait_compute(model_group[0],test_loader,num_wait_test)
                    time.sleep(0.1)
                model_list[j].load_state_dict(deserialize_model(req.wait()))
            elif high_IO:
                comm.recv(source=i, tag=12)
                model_list[j].load_state_dict(torch.load(high_IO_dir+'/model_'+str(j)+'.pt'))
            else:
                model_list[j].load_state_dict(deserialize_model(comm.recv(source=i, tag=12)))
            
def mpi_train_host(model_list,optimizer,train_loader,local_ep,loss,model_dir):
    # send model to ranks
    mpi_send_model_host(model_list)
    #train model on host
    time_1 = time.time()
    for j in client_rank_mapping[0]:
        model_list[j].to(device)
        normal_training_process(model_list[j],optimizer[j],train_loader[j],local_ep,loss[j])
    time_2 = time.time()
    print('Host training time:'+str(time_2-time_1))
    with open(model_dir+'/log.txt','a') as file:
        file.write('Host training time:'+str(time_2-time_1)+'\n')
    # receive trained model
    mpi_receive_model_host(model_list)
    time_3 = time.time()
    print('Receive trained model time:'+str(time_3-time_2))
    with open(model_dir+'/log.txt','a') as file:
        file.write('Receive trained model time:'+str(time_3-time_2)+'\n')
    # receive training loss
    for i in range(1,size):
        loss_group = comm.recv(source=i, tag=15)
        for index in range(len(client_rank_mapping[i])):
            j = client_rank_mapping[i][index]
            loss[j] += loss_group[index]
        
def mpi_test_host(model_list, acc_list,class_acc_list,after_aggregation,model_dir):
    time_1 = time.time()
    if after_aggregation:
        mpi_send_model_host(model_list)
    time_2 = time.time()
    #Test the models belong to host
    for j in client_rank_mapping[0]:
        model_list[j].to(device)
        acc, class_acc = test(model_list[j], test_loader)
        acc_list[j].append(acc)
        class_acc_list[j].append(class_acc)
    time_3 = time.time()
    print('Host  test time:'+str(time_3-time_2))
    # Receive models' test result from all workers
    for i in range(1,size):
        acc_group = comm.recv(source=i, tag=13)
        class_acc_group = comm.recv( source=i, tag=14)
        for index in range(len(client_rank_mapping[i])):
            j = client_rank_mapping[i][index]
            acc_list[j].append(acc_group[index])
            class_acc_list[j].append(class_acc_group[index])
    
    time_4 = time.time()
    print('mpi_send model time:'+str(time_2-time_1))
    print('Host  test time:'+str(time_3-time_2))
    print('receive time:'+str(time_4-time_3))
    
    with open(model_dir+'/log.txt','a') as file:
        file.write('mpi_send model time:'+str(time_2-time_1)+'\n')
        file.write('Host  test time:'+str(time_3-time_2)+'\n')
        file.write('receive time:'+str(time_4-time_3)+'\n')
        
        
def mpi_test_rank(model_group,after_aggregation):   
    acc_group = []
    class_group = []
    if after_aggregation:
        # receive model for inference from master processes
        for index in range(len(model_group)):
            if async_comm:
                req = comm.irecv(source = 0, tag = 11)
                while not req.Test():
                    #perfom some computation
                    wait_compute(model_group[0],test_loader,num_wait_test)
                    time.sleep(0.1)
                model_group[index] = deserialize_model(req.wait())
            elif high_IO:
                j = allocated_clients[index]
                comm.recv(source=0, tag=11)
                model_group[index].load_state_dict(torch.load(high_IO_dir+'/model_'+str(j)+'.pt'))
            else:
                model_group[index].load_state_dict(deserialize_model(comm.recv(source=0, tag=11)))
    for index in range(len(model_group)):
        model_group[index].to(device)
        acc, class_acc = test(model_group[index], test_loader)
        acc_group.append(acc)
        class_group.append(class_acc)
    # Send update back to master
    with open(model_dir+'/rank_log/log_'+str(rank)+'.txt','a') as file:
        file.write('The acc groups is:')
        file.write(str(acc_group)+'\n')
        file.write('The class groups is\n')
        file.write(str(class_group)+'\n')
    comm.send(acc_group, dest=0, tag=13)
    comm.send(class_group, dest=0, tag=14)
    
  
def Centralized_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    test_acc = []
    class_acc_list = []
    loss = []
    optimizer = {}
    scheduler = {}
    learning_rate = lr
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir
    pair,area = write_info(model_dir)
    for i in range(1,size):
        comm.send(model_dir, dest=i, tag=7)
        comm.send(area, dest=i, tag=8)
    for i in range(num_car):
        model.append(copy.deepcopy(global_model))
        loss.append([])
    if mpi_train:
        for j in client_rank_mapping[0]:
            optimizer[j] = optim.SGD(params=model[j].parameters(), lr=learning_rate)
            scheduler[j] = ReduceLROnPlateau(optimizer[j], mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=False)
    else:
        for index in range(num_car):
            optimizer[index] = optim.SGD(params=model[index].parameters(), lr=learning_rate)
            scheduler[index] = ReduceLROnPlateau(optimizer[index], mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=False)
    for i in range(num_round):
        print('This is the round:',i)
        
        for param_group in optimizer[client_rank_mapping[0][0]].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[j].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out mpi training
        if mpi_train:
            mpi_train_host(model,optimizer,train_loader,local_ep,loss,model_dir)
        else:
            time_2 = time.time()
            for index in range(num_car):
                normal_training_process(model[index],optimizer[index],train_loader[index],local_ep,loss[index])
            print('Host training time:'+str(time_2-start_time))
            with open(model_dir+'/log.txt','a') as file:
                file.write('Host training time:'+str(time_2-start_time)+'\n')
        
        #upload_central and aggregate
        w = []
        for index in range(num_car):
            w.append(copy.deepcopy(model[index].state_dict()))
        avg_w = average_weights(w,np.array(weights))
        for index in range(num_car):
            model[index].load_state_dict(copy.deepcopy(avg_w))
       
        acc, class_acc = test(model[0], test_loader)
        test_acc.append(acc)
        class_acc_list.append(class_acc)
        print('test acc:',acc)
        print('class acc:', class_acc)
        end_time = time.time()
        print(f'{end_time-start_time} [sec] for this epoch')
        for index in range(1,size):
            comm.send(acc, dest=index, tag=16)
        for j in client_rank_mapping[0]:
            scheduler[j].step(acc)
        with open(model_dir+'/log.txt','a') as file:
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('test acc:'+str(acc)+'\n')
            file.write('class acc:'+str(class_acc)+'\n')
        #test global model accuracy
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(test_acc[-1])+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.array(test_acc[-early_stop_round:])-test_acc[-1])<1e-7).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            comm.send('early stop', dest=0, tag=11)
            break
    return loss, test_acc , model_dir 
    
def Decentralized_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = {}
    scheduler = {}
    learning_rate = lr
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_epoch_time_'+str(args.epoch_time)+suffix_dir
    pair,area = write_info(model_dir)
    for i in range(1,size):
        comm.send(model_dir, dest=i, tag=7)
        comm.send(area, dest=i, tag=8)
    # Distribute model to all clients 
    for i in range(num_car):
        model.append(copy.deepcopy(global_model))
        # optimizer.append([])
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    if mpi_train:
        for j in client_rank_mapping[0]:
            optimizer[j] = optim.SGD(params=model[j].parameters(), lr=learning_rate)
            scheduler[j] = ReduceLROnPlateau(optimizer[j], mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=False)
    else:
        for index in range(num_car):
            optimizer[index] = optim.SGD(params=model[index].parameters(), lr=learning_rate)
            scheduler[index] = ReduceLROnPlateau(optimizer[index], mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=False)
    # mpi_send_model_host(model)
    for i in range(num_round):
        print('######################################################################')
        print('This is the round:',i)
        # print('lr:',learning_rate)
        for param_group in optimizer[client_rank_mapping[0][0]].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[client_rank_mapping[0][0]].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out mpi training
        if mpi_train:
            mpi_train_host(model,optimizer,train_loader,local_ep,loss,model_dir)
        else:
            for index in range(num_car):
                normal_training_process(model[index],optimizer[index],train_loader[index],local_ep,loss[index])
            time_2 = time.time()
            print('Host training time:'+str(time_2-start_time))
            with open(model_dir+'/log.txt','a') as file:
                file.write('Host training time:'+str(time_2-start_time)+'\n')
        
        # do model update
        time_a = time.time()
        receiver_buffer = {}
        model_before_training = copy.deepcopy(model)
        for seconds in range(args.epoch_time):
            for a,b in pair[i*args.epoch_time+seconds]: 
                if distribution == 'area':
                    if car_type_list[a] == car_type_list[b] or car_type_list[a] == 0 or car_type_list[b] == 0: 
                        receiver_buffer[a] = b
                        receiver_buffer[b] = a
                else: 
                    receiver_buffer[a] = b
                    receiver_buffer[b] = a
        # Then check receiver_buffer, if the model is in the buffer, then do aggregation
        for key, buffered_model_id in receiver_buffer.items():  
            model[key].load_state_dict(average_weights([model[key].state_dict(),model_before_training[buffered_model_id].state_dict()],np.array([weights[key],weights[buffered_model_id]])))
            model[key].to(device)


        time_b = time.time()
        print('Aggregation time:',time_b-time_a)
        
        mpi_test_host(model, acc_global, class_acc_list,True,model_dir)
        #use_lr_scheduler:
        if mpi_train:
            for index in range(1,size):
                comm.send(np.average(acc_global,axis=0)[-1], dest=index, tag=16)
            for j in client_rank_mapping[0]:
                scheduler[j].step(np.average(acc_global,axis=0)[-1])
        else:
            for index in range(num_car):
                scheduler[index].step(np.average(acc_global,axis=0)[-1])
        end_time = time.time()
        print('mpi_test time:',end_time-time_b)
        print(f'{end_time-start_time} [sec] for this epoch')
        # print('Before/After aggregation acc:')
        # for index in range(num_car):
        #     print('car:',index,'---------------------------------------------------------------')
        #     print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
        #     print(class_acc_list_before_aggregation[index][-1])
        #     print(class_acc_list[index][-1])
        #     print('Local Cache model version:')
        print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('Variance test acc:',np.var(acc_global,axis=0)[-1])
        # print('pair:',pair[i])
        with open(model_dir+'/log.txt','a') as file:
            file.write('aggregation time:'+str(time_b-time_a)+'\n')
            file.write('mpi_test time:'+str(end_time-time_b)+'\n')
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Before/After aggregation acc:'+'\n')
            for index in range(num_car):
                file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                file.write(str(acc_global[index][-1])+'\n')
                # file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                # file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
            file.write('Variance test acc:'+str(np.var(acc_global,axis=0)[-1])+'\n')

        
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_epoch_time_'+str(args.epoch_time)+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\t'+str(np.var(acc_global,axis=0)[-1])+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-7).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            comm.send('early stop', dest=0, tag=11)
            break
    return loss,acc_global,class_acc_list, acc_local, model_dir
 
def Decentralized_Cache_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = {}
    scheduler = {}
    learning_rate = lr
    last_lr = lr
    fresh_class_time_table = np.zeros([num_car,num_car])
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+'_local_ep_'+str(local_ep)+'_epoch_time_'+str(args.epoch_time)+'_kick_out_'+str(args.kick_out) +suffix_dir
    pair,area = write_info(model_dir)

    for i in range(1,size):
        comm.send(model_dir, dest=i, tag=7)
        comm.send(area, dest=i, tag=8)
    # Distribute model to all clients 
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        # optimizer.append([])
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    if mpi_train:
        for j in client_rank_mapping[0]:
            optimizer[j] = optim.SGD(params=model[j].parameters(), lr=learning_rate)
            scheduler[j] = ReduceLROnPlateau(optimizer[j], mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=False)
    else:
        for index in range(num_car):
            optimizer[index] = optim.SGD(params=model[index].parameters(), lr=learning_rate)
            scheduler[index] = ReduceLROnPlateau(optimizer[index], mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=False)
    # mpi_send_model_host(model)
    for i in range(num_round):
        print('######################################################################')
        print('This is the round:',i)
        # print('lr:',learning_rate)
        for param_group in optimizer[client_rank_mapping[0][0]].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[client_rank_mapping[0][0]].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out mpi training
        if mpi_train:
            mpi_train_host(model,optimizer,train_loader,local_ep,loss,model_dir)
        else:
            time_2 = time.time()
            for index in range(num_car):
                normal_training_process(model[index],optimizer[index],train_loader[index],local_ep,loss[index])
            print('Host training time:'+str(time_2-start_time))
            with open(model_dir+'/log.txt','a') as file:
                file.write('Host training time:'+str(time_2-start_time)+'\n')
        
        # model_before_aggregation = copy.deepcopy(model)
        
        # mpi_test_host(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation,False,model_dir)
        model_before_training = copy.deepcopy(model)
        if kick_out == True:
            for index in range(args.num_car):
                local_cache[index] = kick_out_timeout_model(local_cache[index],i-args.kick_out)
            torch.cuda.empty_cache()
        for seconds in range(args.epoch_time):
            for a,b in pair[i*args.epoch_time+seconds]: 
                update_model_cache(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, kick_out)
                torch.cuda.empty_cache()
        #########################
        #Statistic cache age and cache number
        cache_age = 0
        cache_num = 0
        for index in range(args.num_car):
            cache_num += len(local_cache[index])
            for key in local_cache[index]:
                cache_age += i-local_cache[index][key]['time']
        avg_cache_age = cache_age/cache_num
        with open(model_dir+'/cache_age_cache_num_'+str(args.algorithm )+'_'+str(cache_size)+'_'+str(args.epoch_time)+'_'+str(args.kick_out)+'.txt','a') as file:
            file.write(str(i)+':')
            file.write(str(avg_cache_age)+'\t'+str(cache_num/args.num_car)+'\n')
        #########################
        cache_info = np.zeros([args.num_car])
        for index in range(args.num_car):
            # cache_info_by_time[0][index] += 1 
            cache_info[index] += 1
            for key in local_cache[index]:
                # print(local_cache[index][key]['time'])
                cache_info[key] += 1
        with open(model_dir+'/cache_info.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            file.write(str(cache_info)+'\n')
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        for index in range(num_car):
            model[index] = cache_average_process(model[index],index,i,local_cache[index],weights)
           
            
        

        mpi_test_host(model, acc_global, class_acc_list, True,model_dir)
        #use_lr_scheduler:
        if mpi_train:
            for index in range(1,size):
                comm.send(np.average(acc_global,axis=0)[-1], dest=index, tag=16)
            for j in client_rank_mapping[0]:
                scheduler[j].step(np.average(acc_global,axis=0)[-1])
            new_lr = scheduler[j].get_last_lr()[0]
        else:
            for index in range(num_car):
                scheduler[index].step(np.average(acc_global,axis=0)[-1])
            new_lr = scheduler[index].get_last_lr()[0]
        if last_lr != new_lr and args.kick_out > 1:
            # args.kick_out -=1
            last_lr = new_lr
        
        end_time = time.time()
        

        
        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        # print('Before/After aggregation acc:')
        # for index in range(num_car):
        #     print('car:',index,'---------------------------------------------------------------')
        #     print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
        #     print(class_acc_list_before_aggregation[index][-1])
        #     print(class_acc_list[index][-1])
        #     print('Local Cache model version:')
        #     for key in local_cache[index]:
        #         print(key,local_cache[index][key]['time'])#,local_cache[index][key]['fresh_metric'])
        # print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('Variance test acc:',np.var(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        with open(model_dir+'/log.txt','a') as file:
            file.write('fresh_class_time_table\n')
            file.write(str(fresh_class_time_table)+'\n')
            for index in range(num_car):
                file.write(str(index)+':'+str(class_acc_list[index][-1])+'\n')
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Before/After aggregation acc:'+'\n')
            for index in range(num_car):
                file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                file.write(str(acc_global[index][-1])+'\n')
                # file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                # file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
            file.write('Variance test acc:'+str(np.var(acc_global,axis=0)[-1])+'\n')
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+'_local_ep_'+str(local_ep)+'_epoch_time_'+str(args.epoch_time)+'_kick_out_'+str(args.kick_out)+suffix_dir+'.txt','a') as file:
           file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\t'+str(np.var(acc_global,axis=0)[-1])+'\n')
        
        
        
        
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-7).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            comm.send('early stop', dest=0, tag=11)
            break
    return loss,acc_global,class_acc_list, acc_local, model_dir 

def Decentralized_Cache_areas_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = {}
    scheduler = {}
    learning_rate = lr
    last_lr = lr
    fresh_class_time_table = np.zeros([num_car,num_car])
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+'_local_ep_'+str(local_ep)+'_epoch_time_'+str(args.epoch_time)+'_kick_out_'+str(args.kick_out) +suffix_dir
    pair,area = write_info(model_dir)


    for i in range(1,size):
        comm.send(model_dir, dest=i, tag=7)
        comm.send(area, dest=i, tag=8)
    # Distribute model to all clients 
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        # optimizer.append([])
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for i in range(args.num_car-num_car):
        local_cache.append({})
    if mpi_train:
        for j in client_rank_mapping[0]:
            optimizer[j] = optim.SGD(params=model[j].parameters(), lr=learning_rate)
            scheduler[j] = ReduceLROnPlateau(optimizer[j], mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=False)
    else:
        for index in range(num_car):
            optimizer[index] = optim.SGD(params=model[index].parameters(), lr=learning_rate)
            scheduler[index] = ReduceLROnPlateau(optimizer[index], mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=False)
    # mpi_send_model_host(model)
    for i in range(num_round):
        print('######################################################################')
        print('This is the round:',i)
        # print('lr:',learning_rate)
        for param_group in optimizer[client_rank_mapping[0][0]].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[client_rank_mapping[0][0]].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out mpi training
        if mpi_train:
            mpi_train_host(model,optimizer,train_loader,local_ep,loss,model_dir)
        else:
            time_2 = time.time()
            for index in range(num_car):
                normal_training_process(model[index],optimizer[index],train_loader[index],local_ep,loss[index])
            print('Host training time:'+str(time_2-start_time))
            with open(model_dir+'/log.txt','a') as file:
                file.write('Host training time:'+str(time_2-start_time)+'\n')
        
        # model_before_aggregation = copy.deepcopy(model)
        
        # mpi_test_host(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation,False,model_dir)
        model_before_training = copy.deepcopy(model)
        if kick_out == True:
            for index in range(args.num_car):
                local_cache[index] = kick_out_timeout_model(local_cache[index],i-args.kick_out)
            torch.cuda.empty_cache()
        for seconds in range(args.epoch_time):
            for a,b in pair[i*args.epoch_time+seconds]: 
                if car_type_list[a] == car_type_list[b] or car_type_list[a] == 0 or car_type_list[b] == 0: 
                    update_model_cache(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, kick_out)
                # if car_type_list[a] == car_type_list[b]: 
                #     if car_type_list[a] != 0:
                #         update_model_cache(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, kick_out)
                #     else:
                #         update_model_cache_taxi_to_taxi(local_cache, a,b,i, cache_size, kick_out)
                # elif car_type_list[a] == 0:
                #     update_model_cache_car_to_taxi(local_cache, model_before_training[b],b,a,i, cache_size, cache_size, kick_out)
                # elif car_type_list[b] == 0:
                #     update_model_cache_car_to_taxi(local_cache, model_before_training[a],a,b,i, cache_size, cache_size, kick_out)
                torch.cuda.empty_cache()
        #########################
        #Statistic cache age and cache number
        cache_age = 0
        cache_num = 0
        for index in range(num_car):
            cache_num += len(local_cache[index])
            for key in local_cache[index]:
                cache_age += i-local_cache[index][key]['time']
        avg_cache_age = cache_age/cache_num
        with open(model_dir+'/cache_age_cache_num_'+str(args.algorithm )+'_'+str(cache_size)+'_'+str(args.epoch_time)+'_'+str(args.kick_out)+'.txt','a') as file:
            file.write(str(i)+':')
            file.write(str(avg_cache_age)+'\t'+str(cache_num/num_car)+'\n')
        #########################
        cache_info = np.zeros([num_car])
        for index in range(num_car):
            # cache_info_by_time[0][index] += 1 
            cache_info[index] += 1
            for key in local_cache[index]:
                # print(local_cache[index][key]['time'])
                cache_info[key] += 1
        with open(model_dir+'/cache_info.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            file.write(str(cache_info)+'\n')
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        for index in range(num_car):
            model[index] = cache_average_process(model[index],index,i,local_cache[index],weights)
           
            
        

        mpi_test_host(model, acc_global, class_acc_list, True,model_dir)
        #use_lr_scheduler:
        if mpi_train:
            for index in range(1,size):
                comm.send(np.average(acc_global,axis=0)[-1], dest=index, tag=16)
            for j in client_rank_mapping[0]:
                scheduler[j].step(np.average(acc_global,axis=0)[-1])
            new_lr = scheduler[j].get_last_lr()[0]
        else:
            for index in range(num_car):
                scheduler[index].step(np.average(acc_global,axis=0)[-1])
            new_lr = scheduler[index].get_last_lr()[0]
        if last_lr != new_lr and args.kick_out > 1:
            # args.kick_out -=1
            last_lr = new_lr
        
        end_time = time.time()
        

        
        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        # print('Before/After aggregation acc:')
        # for index in range(num_car):
        #     print('car:',index,'---------------------------------------------------------------')
        #     print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
        #     print(class_acc_list_before_aggregation[index][-1])
        #     print(class_acc_list[index][-1])
        #     print('Local Cache model version:')
        #     for key in local_cache[index]:
        #         print(key,local_cache[index][key]['time'])#,local_cache[index][key]['fresh_metric'])
        # print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('Variance test acc:',np.var(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        with open(model_dir+'/log.txt','a') as file:
            file.write('fresh_class_time_table\n')
            file.write(str(fresh_class_time_table)+'\n')
            for index in range(num_car):
                file.write(str(index)+':'+str(class_acc_list[index][-1])+'\n')
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Before/After aggregation acc:'+'\n')
            for index in range(num_car):
                file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                file.write(str(acc_global[index][-1])+'\n')
                # file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                # file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
            file.write('Variance test acc:'+str(np.var(acc_global,axis=0)[-1])+'\n')
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+'_local_ep_'+str(local_ep)+'_epoch_time_'+str(args.epoch_time)+'_kick_out_'+str(args.kick_out)+suffix_dir+'.txt','a') as file:
           file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\t'+str(np.var(acc_global,axis=0)[-1])+'\n')
        
        
        
        
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-7).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            comm.send('early stop', dest=0, tag=11)
            break
    return loss,acc_global,class_acc_list, acc_local, model_dir 


def Decentralized_Cache_areas_GB_process(suffix_dir,train_loader,test_loader,num_round,local_ep):
    model = []
    local_cache = []
    acc_global = []
    acc_global_before_aggregation = []
    class_acc_list = []
    class_acc_list_before_aggregation = []
    acc_local = []
    loss = []
    optimizer = {}
    scheduler = {}
    learning_rate = lr
    last_lr = lr
    fresh_class_time_table = np.zeros([num_car,num_car])
    # current_model_time_table = np.zeros(num_car)
    # current_model_combination_table = np.eye(num_car)#statistic_data#
    current_class_test = np.zeros([num_car,10])
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+'_local_ep_'+str(local_ep)+'_epoch_time_'+str(args.epoch_time)+'_kick_out_'+str(args.kick_out) +suffix_dir
    pair,area = write_info(model_dir)

    for i in range(1,size):
        comm.send(model_dir, dest=i, tag=7)
        comm.send(area, dest=i, tag=8)
    # Distribute model to all clients 
    for i in range(num_car):
        local_cache.append({})
        model.append(copy.deepcopy(global_model))
        # optimizer.append([])
        acc_global.append([])
        acc_global_before_aggregation.append([])
        class_acc_list.append([])
        class_acc_list_before_aggregation.append([])
        acc_local.append([])
        loss.append([])
    for i in range(args.num_car-num_car):
        local_cache.append({})
    if mpi_train:
        for j in client_rank_mapping[0]:
            optimizer[j] = optim.SGD(params=model[j].parameters(), lr=learning_rate)
            scheduler[j] = ReduceLROnPlateau(optimizer[j], mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=False)
    else:
        for index in range(num_car):
            optimizer[index] = optim.SGD(params=model[index].parameters(), lr=learning_rate)
            scheduler[index] = ReduceLROnPlateau(optimizer[index], mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=False)
    # mpi_send_model_host(model)
    for i in range(num_round):
        print('######################################################################')
        print('This is the round:',i)
        # print('lr:',learning_rate)
        for param_group in optimizer[client_rank_mapping[0][0]].param_groups:
            print(param_group['lr'])
        # if i %300== 0:
        #     learning_rate  = learning_rate/5
        #     for index in range(num_car):
        #         for param_group in optimizer[index].param_groups:
        #             param_group['lr'] = learning_rate
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer[client_rank_mapping[0][0]].param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out mpi training
        if mpi_train:
            mpi_train_host(model,optimizer,train_loader,local_ep,loss,model_dir)
        else:
            time_2 = time.time()
            for index in range(num_car):
                normal_training_process(model[index],optimizer[index],train_loader[index],local_ep,loss[index])
            print('Host training time:'+str(time_2-start_time))
            with open(model_dir+'/log.txt','a') as file:
                file.write('Host training time:'+str(time_2-start_time)+'\n')
        
        # model_before_aggregation = copy.deepcopy(model)
        
        # mpi_test_host(model_before_aggregation, acc_global_before_aggregation, class_acc_list_before_aggregation,False,model_dir)
        model_before_training = copy.deepcopy(model)
        if kick_out == True:
            for index in range(args.num_car):
                if len(local_cache[index])>cache_size:
                        local_cache[index] = prune_cache(local_cache[index], type_limits_taxi, cache_size,'time','car_type')
                # if car_type_list[index] != 0:
                #     if len(local_cache[index])>cache_size:
                #         local_cache[index] = prune_cache(local_cache[index], type_limits_car, cache_size,'time','from')
                # else:
                #     if len(local_cache[index])>cache_size:
                #         local_cache[index] = prune_cache(local_cache[index], type_limits_taxi, cache_size,'time','car_type')
            torch.cuda.empty_cache()
        for seconds in range(args.epoch_time):
            for a,b in pair[i*args.epoch_time+seconds]: 
                if car_type_list[a] == car_type_list[b] or car_type_list[a] == 0 or car_type_list[b] == 0: 
                    update_model_cache_car_to_car_p(local_cache, model_before_training[a], model_before_training[b],a,b,i, cache_size, kick_out,car_area_list, type_limits_car)
                # if car_type_list[a] == car_type_list[b]: 
                #     if car_type_list[a] != 0:
                #         update_model_cache_car_to_car_p(local_cache, model_before_training[a],model_before_training[b],a,b,i, cache_size, kick_out,car_type_list, type_limits_car)
                #     else:
                #         update_model_cache_taxi_to_taxi_p(local_cache, a,b, cache_size,type_limits_taxi)
                # elif car_type_list[a] == 0:
                #     update_model_cache_car_to_taxi_p(local_cache, model_before_training[b],b,a,i, cache_size, cache_size, kick_out,car_type_list,type_limits_car,type_limits_taxi)
                # elif car_type_list[b] == 0:
                #     update_model_cache_car_to_taxi_p(local_cache, model_before_training[a],a,b,i, cache_size, cache_size, kick_out,car_type_list,type_limits_car,type_limits_taxi)
                torch.cuda.empty_cache()
        #########################
        #Statistic cache age and cache number
        cache_age = 0
        cache_num = 0
        for index in range(num_car):
            cache_num += len(local_cache[index])
            for key in local_cache[index]:
                cache_age += i-local_cache[index][key]['time']
        avg_cache_age = cache_age/cache_num
        with open(model_dir+'/cache_age_cache_num_'+str(args.algorithm )+'_'+str(cache_size)+'_'+str(args.epoch_time)+'_'+str(args.kick_out)+'.txt','a') as file:
            file.write(str(i)+':')
            file.write(str(avg_cache_age)+'\t'+str(cache_num/num_car)+'\n')
        #########################
        cache_info = np.zeros([num_car])
        for index in range(num_car):
            # cache_info_by_time[0][index] += 1 
            cache_info[index] += 1
            for key in local_cache[index]:
                # print(local_cache[index][key]['time'])
                cache_info[key] += 1
        with open(model_dir+'/cache_info.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            file.write(str(cache_info)+'\n')
        # do model aggregation
        print('Updated/aggregated model time/combination:')
        for index in range(num_car):
            model[index] = cache_average_process(model[index],index,i,local_cache[index],weights)
        
           
            
        

        mpi_test_host(model, acc_global, class_acc_list, True,model_dir)
        #use_lr_scheduler
        if mpi_train:
            for index in range(1,size):
                comm.send(np.average(acc_global,axis=0)[-1], dest=index, tag=16)
            for j in client_rank_mapping[0]:
                scheduler[j].step(np.average(acc_global,axis=0)[-1])
            new_lr = scheduler[j].get_last_lr()[0]
        else:
            for index in range(num_car):
                scheduler[index].step(np.average(acc_global,axis=0)[-1])
            new_lr = scheduler[index].get_last_lr()[0]
        if last_lr != new_lr and args.kick_out > 1:
            # args.kick_out -=1
            last_lr = new_lr
        
        end_time = time.time()
        

        
        for index in range(num_car):
            current_class_test[index] = class_acc_list[index][-1]
        # print(current_model_combination_table)
        # print(current_class_test)
        print(fresh_class_time_table)
        for index in range(num_car):
            print(class_acc_list[index][-1])
        print(f'{end_time-start_time} [sec] for this epoch')
        # print('Before/After aggregation acc:')
        # for index in range(num_car):
        #     print('car:',index,'---------------------------------------------------------------')
        #     print(acc_global_before_aggregation[index][-1],acc_global[index][-1])
        #     print(class_acc_list_before_aggregation[index][-1])
        #     print(class_acc_list[index][-1])
        #     print('Local Cache model version:')
        #     for key in local_cache[index]:
        #         print(key,local_cache[index][key]['time'])#,local_cache[index][key]['fresh_metric'])
        # print('----------------------------------------------------------------------')
        print('Average test acc:',np.average(acc_global,axis=0)[-1])
        print('Variance test acc:',np.var(acc_global,axis=0)[-1])
        print('pair:',pair[i])
        with open(model_dir+'/log.txt','a') as file:
            # file.write('fresh_class_time_table\n')
            # file.write(str(fresh_class_time_table)+'\n')
            for index in range(num_car):
                file.write(str(index)+':'+str(class_acc_list[index][-1])+'\n')
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Before/After aggregation acc:'+'\n')
            for index in range(num_car):
                file.write('car:'+str(index)+'---------------------------------------------------------------'+'\n')
                file.write(str(acc_global[index][-1])+'\n')
                # file.write(str(acc_global_before_aggregation[index][-1])+','+str(acc_global[index][-1])+'\n')
                # file.write(str(class_acc_list_before_aggregation[index][-1])+'\n')
                file.write(str(class_acc_list[index][-1])+'\n')
                file.write('Local Cache model version:'+'\n')
                for key in local_cache[index]:
                    file.write(str(key)+':'+str(local_cache[index][key]['time'])+'\t'+str(local_cache[index][key]['car_type'])+'\n')#,local_cache[index][key]['fresh_metric'])
            file.write('----------------------------------------------------------------------'+'\n')
            file.write('Average test acc:'+str(np.average(acc_global,axis=0)[-1])+'\n')
            file.write('Variance test acc:'+str(np.var(acc_global,axis=0)[-1])+'\n')
        with open(model_dir+'/average_acc_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+'_'+str(cache_size)+'_local_ep_'+str(local_ep)+'_epoch_time_'+str(args.epoch_time)+'_kick_out_'+str(args.kick_out)+suffix_dir+'.txt','a') as file:
           file.write(str(i)+':'+str(np.average(acc_global,axis=0)[-1])+'\t'+str(np.var(acc_global,axis=0)[-1])+'\n')
        
        
        
        
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.average(acc_global,axis=0)[-early_stop_round:]-np.average(acc_global,axis=0)[-1])<1e-7).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            comm.send('early stop', dest=0, tag=11)
            break
    return loss,acc_global,class_acc_list, acc_local, model_dir 

def ml_process(suffix_dir,train_loader, num_round,local_ep):
    model = copy.deepcopy(global_model)
    acc = []
    loss = []
    optimizer = optim.SGD(params=model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=False)
    learning_rate = lr
    model_dir = './result/'+str(date_time.strftime('%Y-%m-%d %H_%M_%S'))+'_'+task+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir
    
    for i in range(num_round):
        print('This is the round:',i)
        
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        with open(model_dir+'/log.txt','a') as file:
            file.write('This is the round:'+str(i)+'\n')
            for param_group in optimizer.param_groups:
                file.write('current lr is: '+str(param_group['lr'])+'\n')
        start_time = time.time()
        #carry out local training
        for _ in tqdm(range(local_ep),disable=True):
            loss.append(normal_train(model, optimizer, full_loader))
        current_acc, class_acc = test(model, test_loader,num_class)
        acc.append(current_acc) 
        #use_lr_scheduler:
        scheduler.step(current_acc)
        end_time = time.time()
        print(f'{end_time-start_time} [sec] for this epoch')
        print('Acc:',current_acc)
        print('Class acc:',class_acc)
        with open(model_dir+'/log.txt','a') as file:
            file.write(f'{end_time-start_time} [sec] for this epoch'+'\n')
            file.write('Acc:'+str(current_acc)+'\n')
            file.write('Class acc:'+str(class_acc)+'\n')
        with open(model_dir+'/average_acc_'+task+'_'+str(Randomseed)+'_'+args.algorithm+suffix_dir+'.txt','a') as file:
            file.write(str(i)+':'+str(current_acc)+'\n')
        #write a code to early stop, if test acc remain unchanged for 10 rounds, then stop and return
        if i>early_stop_round and (abs(np.array(acc[-early_stop_round:])-acc[-1])<1e-7).all():
            print('early stop at round:',i)
            with open(model_dir+'/log.txt','a') as file:
                file.write('early stop at round:'+str(i)+'\n')
            comm.send('early stop', dest=0, tag=11)
            break
    return loss,acc    




if __name__ == '__main__':
    mpi_train = True
    high_IO = False
    async_comm = False
    num_wait_test = 1
    comm = MPI.COMM_WORLD
    # comm = pkl5.Intracomm(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    size = comm.Get_size()
    status = MPI.Status()
    print('Hello from rank: '+str(rank))
    #rank==0 refers to the Master Process
    car_type_list = []
    car_type_list += [0]*num_car
    type_limits_taxi = {'1':3,'2':3,'3':3}
    type_limits_car = {'1':3,'2':3,'3':3}
    if args.overlap == 0:
        target_labels = [[0,1,2,3],[4,5,6],[7,8,9]]
    if args.overlap == 1:
        target_labels = [[9,0,1,2,3],[3,4,5,6],[6,7,8,9]]
    if args.overlap == 2:
        target_labels = [[8,9,0,1,2,3],[2,3,4,5,6],[5,6,7,8,9]]
    if args.overlap == 3:
        target_labels = [[7,8,9,0,1,2,3],[1,2,3,4,5,6],[4,5,6,7,8,9]]
    # type_limits_car = {'taxi':5,'car':5}

    if rank ==0:
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:0")
        data_distribution = distribution
        if args.algorithm == 'ml':
            distribution = 'iid'
        print('The size is: '+str(size))
        if task == 'mnist':
            hidden_size = 64
            num_class = 10
            global_model = CNNMnist(1,10)
            #MNIST
            if distribution == 'iid':
                train_loader, sub_test_loader, test_loader, full_loader =  get_mnist_iid(num_car,batch_size,test_ratio)
            elif distribution == 'non-iid':
                train_loader, sub_test_loader, test_loader, full_loader =  get_mnist_non_iid(shards_allocation, num_car, batch_size,test_ratio)
            elif distribution == 'dirichlet':
                train_loader, sub_test_loader, test_loader, full_loader =  get_mnist_dirichlet(alpha, num_car,batch_size,test_ratio)
                data_distribution = data_distribution+'_'+str(alpha)
            elif distribution == 'area':
                car_type_list = []
                car_type_list += [1]*30
                car_type_list += [0]*4
                car_type_list += [2]*30
                car_type_list += [0]*3
                car_type_list += [3]*30
                car_type_list += [0]*3
                # num_car -= 10 
                car_area_list = []
                car_area_list += [1]*34
                car_area_list += [2]*33
                car_area_list += [3]*33
                train_loader, sub_test_loader, test_loader, full_loader =  get_mnist_area(shards_allocation,  batch_size,test_ratio,car_area_list,target_labels)
                data_distribution = distribution+'_'+str(args.overlap)+'_overlap'
            else:
                raise ValueError('Error')
        elif task == 'cifar10':
            num_class = 10
            global_model = ResNet18()
            # cifar10
            if distribution == 'iid':
                train_loader, sub_test_loader, test_loader, full_loader =  get_cifar10_iid(num_car,batch_size,test_ratio)
            elif distribution == 'non-iid':
                train_loader, sub_test_loader, test_loader, full_loader =  get_cifar10_non_iid(shards_allocation, num_car, batch_size,test_ratio)
            elif distribution == 'dirichlet':
                train_loader, sub_test_loader, test_loader, full_loader =  get_cifar10_dirichlet(alpha, num_car,batch_size,test_ratio)
                data_distribution = data_distribution+'_'+str(alpha)
            else:
                raise ValueError('Error')
        elif task == 'cifar100':
            num_class = 100
            global_model = ResNet18(num_classes=num_class)
            # cifar100
            if distribution == 'iid':
                train_loader, sub_test_loader, test_loader, full_loader =  get_cifar100_iid(num_car,batch_size,test_ratio)
            elif distribution == 'non-iid':
                train_loader, sub_test_loader, test_loader, full_loader =  get_cifar100_non_iid(shards_allocation, num_car, batch_size,test_ratio)
            elif distribution == 'dirichlet':
                train_loader, sub_test_loader, test_loader, full_loader =  get_cifar100_dirichlet(alpha, num_car,batch_size,test_ratio)
                data_distribution = data_distribution+'_'+str(alpha)
            else:
                raise ValueError('Error')
        elif task == 'fashionmnist':
            global_model = CNNFashion_Mnist()
            # FashionMNIST
            num_class = 10
            if distribution == 'iid':
                train_loader, sub_test_loader, test_loader, full_loader =  get_fashionmnist_iid(num_car,batch_size,test_ratio)
            elif distribution == 'non-iid':
                train_loader, sub_test_loader, test_loader, full_loader =  get_fashionmnist_non_iid(shards_allocation, num_car, batch_size,test_ratio)
            elif distribution == 'dirichlet':
                train_loader, sub_test_loader, test_loader, full_loader =  get_fashionmnist_dirichlet(alpha, num_car,batch_size,test_ratio)
                data_distribution = data_distribution+'_'+str(alpha)
            elif distribution == 'area':
                car_type_list = []
                car_type_list += [1]*30
                car_type_list += [0]*4
                car_type_list += [2]*30
                car_type_list += [0]*3
                car_type_list += [3]*30
                car_type_list += [0]*3
                # num_car -= 10 
                car_area_list = []
                car_area_list += [1]*34
                car_area_list += [2]*33
                car_area_list += [3]*33
                train_loader, sub_test_loader, test_loader, full_loader =  get_fashionmnist_area(shards_allocation,  batch_size,test_ratio,car_area_list,target_labels)
                data_distribution = distribution+'_'+str(args.overlap)+'_overlap'
            else:
                raise ValueError('Error')
        else:
            raise ValueError('Error')
        
        
        if test_ratio:
            test_loader = sub_test_loader
            print('To do subtest, with:',len(test_loader.dataset))
        
        global_model.to(device)
        
        #Allocate clients into different groups for rank, each rank for one group
        #try to split the model into groups
        client_rank_mapping = []
        for i in range(size):
            client_rank_mapping.append([])
        for i in range(num_car):
            client_rank_mapping[i%size].append(i)
        print('client_rank_mapping:')
        print(client_rank_mapping)
        # Distribute the mapping table to all ranks
        for i in range(1,size):
            comm.send(client_rank_mapping[i], dest=i, tag=8)
        if high_IO:
            high_IO_dir = '/vast/xw2597/models/'+str(time.time())
            if not os.path.exists(high_IO_dir):
                os.makedirs(high_IO_dir)
        for i in range(1,size):
            if high_IO:
                comm.send(high_IO_dir, dest=i, tag=3)
                torch.save(global_model,high_IO_dir+'/model_'+str(i)+'.pt')
                comm.send('initial model send success', dest=i, tag=6)
            else:
                comm.send(serialize_model(global_model), dest=i, tag=6)
        
            
        # Distribute loader to all clients
        if mpi_train:
            for i in range(1,size):
                train_loader_group = []
                for j in client_rank_mapping[i]:
                    train_loader_group.append(train_loader[j])
                comm.send(train_loader_group, dest=i, tag=9)
                comm.send(test_loader, dest=i, tag=10)         
        statistic_data = np.zeros([num_car,num_class])
        for i in range(num_car):
            for input, target in train_loader[i]:
                for item in target:
                    statistic_data[i][item] += 1
        print('Data distribution among cars:')
        print(statistic_data)
        max_std = np.zeros(num_car)
        for i in range(num_car):
            for j in range(num_car):
                if max_std[i] < np.var(statistic_data[i]- statistic_data[j]):
                    max_std[i] = np.var(statistic_data[i]- statistic_data[j])
        data_points = np.sum(statistic_data,axis = 1)
        print(sum(data_points))
        print(data_points/sum(np.array(data_points)))
        if args.weighted_aggregation == True:
            weights = data_points
        else:
            weights = [1]*num_car    
        data_similarity = cosine_similarity(statistic_data)
        print(data_similarity)
        
        
        numbers = list(range(num_car))
        random.shuffle(numbers)
        
        numbers = list(range(num_car))
        pair = []
        
        
        # use road sim to generate car pairings.
    
        start_time = time.time()
        date_time = datetime.datetime.fromtimestamp(start_time)
        # Use the algorithm argument to run the corresponding function
        if args.algorithm == 'ml':
            loss, acc = ml_process(suffix,train_loader,num_round,local_ep)  
        elif args.algorithm == 'cfl':
            fl_loss,  fl_test_acc, model_dir = Centralized_process(suffix,train_loader,test_loader,num_round,local_ep)
        elif args.algorithm == 'dfl':
            dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir = Decentralized_process(suffix,train_loader, test_loader,num_round,local_ep)
        elif args.algorithm == 'cache':
            dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir = Decentralized_Cache_process(suffix,train_loader, test_loader,num_round,local_ep)
        elif args.algorithm == 'cache_areas_LRU':
            dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir = Decentralized_Cache_areas_process(suffix,train_loader, test_loader,num_round,local_ep)
        elif args.algorithm == 'cache_areas_GB':
            dfl_loss, dfl_acc_global, df_class_acc, dfl_acc_local, model_dir = Decentralized_Cache_areas_GB_process(suffix,train_loader, test_loader,num_round,local_ep)
        else:
            raise ValueError('Error')
        if args.algorithm == 'ml':
            print(acc)
        elif args.algorithm == 'cfl':
            print(fl_test_acc)
        else:
            print(np.average(dfl_acc_global,axis=0))
            end_time = time.time()
            total_training_time = end_time-start_time
            hours, remainder = divmod(total_training_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            # with open(model_dir+'/summary.txt','w') as file:
            with open(model_dir+'/summary_'+task+'_'+data_distribution+'_'+str(Randomseed)+'_'+args.algorithm+suffix+'.txt','a') as file:
                file.write('Start time: ' + str(date_time.strftime('%Y-%m-%d %H:%M:%S'))+'\n')
                date_time = datetime.datetime.fromtimestamp(end_time)
                file.write('End time: ' + str(date_time.strftime('%Y-%m-%d %H:%M:%S'))+'\n')
                file.write(f'Total training time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds \n')
                for i in range(num_car):
                    file.write('This is car '+str(i)+'\n')
                    file.write(str(dfl_acc_global[i])+'\n')
                file.write('This is the average acc:\n')
                file.write(str(np.average(dfl_acc_global,axis=0))+'\n')
            print(f"Total training time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
     
    elif mpi_train:
        # Worker processes
        # Initialization
        learning_rate = lr
        optimizer = []
        scheduler = []
        model_group = []
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:1")
            # if rank % torch.cuda.device_count() == 0:
            #     device = torch.device("cuda:0")
            # elif rank % torch.cuda.device_count() == 1:
            #     device = torch.device("cuda:1")
            # elif rank % torch.cuda.device_count() == 2:
            #     device = torch.device("cuda:2")
            # else:
            #     device = torch.device("cuda:3")
        #receive the mapping table
        allocated_clients = comm.recv(source = 0, tag = 8)
        num_client = len(allocated_clients)
        if high_IO:
            high_IO_dir = comm.recv(source = 0, tag = 3)
            comm.recv(source = 0, tag = 6)
            global_model = torch.load(high_IO_dir+'/model_'+str(rank)+'.pt')
        else:
            global_model = deserialize_model(comm.recv(source = 0, tag = 6))
        #initialize the model
        for item in allocated_clients:
            model_group.append(copy.deepcopy(global_model))

        for index in range(num_client):
            optimizer.append(optim.SGD(params=model_group[index].parameters(), lr=learning_rate))
            # optimizer[index].state = {k: v.to(device) for k, v in optimizer[index].state.items()}
            scheduler.append(ReduceLROnPlateau(optimizer[index], mode='max', factor=args.lr_factor, patience=args.lr_patience, verbose=False))
        # receive the dataset
        train_loader_group = comm.recv(source = 0, tag = 9)
        test_loader = comm.recv(source = 0, tag = 10)
        model_dir = comm.recv(source = 0, tag = 7)
        area = comm.recv(source = 0, tag = 8)
        # if rank == size-1:
        statistic_data = np.zeros([num_client,10])
        for i in range(num_client):
            for input, target in train_loader_group[i]:
                for item in target:
                    statistic_data[i][item] += 1
        with open(model_dir+'/rank_log/log_'+str(rank)+'.txt','a') as file:
            file.write('Hello from rank: '+str(rank)+'\n')
            file.write('This is the allocated clients: '+str(allocated_clients)+'\n')
            file.write('Use ' +str(torch.cuda.device_count())+ 'GPUs!'+'\n')
            file.write('Rank '+str(rank)+ 'uses GPU:' +str(device)+'\n')
            file.write('Data distribution among cars:\n')
            file.write(str(statistic_data)+'\n')
        if torch.cuda.device_count() > 1:
            for index in range(num_client):
                transferred_data = []
                transferred_targets = []
                # Transfer data to the device and store it in lists
                for batch_idx, (data, target) in enumerate(train_loader_group[index]):
                    data, target = data.to(device), target.to(device)
                    transferred_data.append(data)
                    transferred_targets.append(target)

                # Concatenate all batches into a single tensor
                transferred_data = torch.cat(transferred_data)
                transferred_targets = torch.cat(transferred_targets)

                # Create a new TensorDataset with the transferred data
                transferred_dataset = TensorDataset(transferred_data, transferred_targets)

                # Create a new DataLoader from the transferred dataset
                train_loader_group[index] = DataLoader(transferred_dataset, batch_size=train_loader_group[index].batch_size, shuffle=True)

        #begin training
        for i in range(num_round):
            # if rank == size-1:
            with open(model_dir+'/rank_log/log_'+str(rank)+'.txt','a') as file:
                file.write('This is the round: '+str(i)+'\n')
                file.write('---------------------------------------------------------------------------\n')
            #train current model
            #receive new model

            # Rank 1 probes for incoming message and then receives it
            stop_signal = False
            for index in range(num_client):
                if high_IO:
                    comm.recv(source=0, tag=11)
                    model_group[index] = torch.load(high_IO_dir+'/model_'+str(allocated_clients[index])+'.pt')
                else:
                    recv_item = comm.recv(source=0, tag=11)
                    if recv_item == 'early stop':
                        stop_signal = True
                        break
                    model_group[index].load_state_dict(deserialize_model(recv_item))    
            if stop_signal:
                break

            #train model
            loss = []

            for index in range(num_client):
                loss.append([])
                # if rank == size-1:
                with open(model_dir+'/rank_log/log_'+str(rank)+'.txt','a') as file:
                    model_group[index].to(device)
                    normal_training_process(model_group[index],optimizer[index],train_loader_group[index],local_ep,loss[index])
            # send back model and loss
            for index in range(num_client):
                if high_IO:
                    torch.save(model_group[index],high_IO_dir+'/model_'+str(allocated_clients[index])+'.pt')
                    comm.send('model send success', dest=0, tag=12)
                else:
                    comm.send(serialize_model(model_group[index].state_dict()), dest=0, tag=12)
            comm.send(loss, dest=0, tag=15)

            # test the model before aggregation

            #change the learning rate
            if args.algorithm != 'cfl':
                mpi_test_rank(model_group,True)
            #use_lr_scheduler:
            acc = comm.recv(source = 0, tag = 16)
            for index in range(num_client):
                scheduler[index].step(acc)
    else:
        # Worker processes
        # Initialization
        learning_rate = lr
        model_group = []
        if torch.cuda.device_count() > 1:
            if rank % torch.cuda.device_count() == 0:
                device = torch.device("cuda:0")
            elif rank % torch.cuda.device_count() == 1:
                device = torch.device("cuda:1")
            elif rank % torch.cuda.device_count() == 2:
                device = torch.device("cuda:2")
            else:
                device = torch.device("cuda:3")
        #receive the mapping table
        allocated_clients = comm.recv(source = 0, tag = 8)
        num_client = len(allocated_clients)
        if high_IO:
            high_IO_dir = comm.recv(source = 0, tag = 3)
            comm.recv(source = 0, tag = 6)
            global_model = torch.load(high_IO_dir+'/model_'+str(rank)+'.pt')
        else:
            global_model = deserialize_model(comm.recv(source = 0, tag = 6))
        #initialize the model
        for item in allocated_clients:
            model_group.append(copy.deepcopy(global_model))

        # receive the dataset
        test_loader = comm.recv(source = 0, tag = 10)
        model_dir = comm.recv(source = 0, tag = 7)
        with open(model_dir+'/rank_log/log_'+str(rank)+'.txt','a') as file:
            file.write('Hello from rank: '+str(rank)+'\n')
            file.write('Use ' +str(torch.cuda.device_count())+ 'GPUs!'+'\n')
            file.write('Rank '+str(rank)+ 'uses GPU' +str(device)+'\n')
            file.write('This is the allocated clients: '+str(allocated_clients)+'\n')
                
        #begin training
        for i in range(num_round):
            with open(model_dir+'/rank_log/log_'+str(rank)+'.txt','a') as file:
                file.write('This is the round: '+str(i)+'\n')
                file.write('---------------------------------------------------------------------------\n')
            if args.algorithm == 'cfl':
                continue;
            mpi_test_rank(model_group,True)

    MPI.Finalize()
