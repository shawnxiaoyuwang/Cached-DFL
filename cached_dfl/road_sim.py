# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:46:43 2024

@author:
"""

import pandas as pd
# import matplotlib.animation as animation
import networkx as nx
import numpy as np
import copy
import random
from sklearn.cluster import KMeans

import seed_setter

# Call the set_seed function at the start
Random_SEED  = seed_setter.set_seed()

# from Cached_DFL_main import set_seed

# # Call the set_seed function at the start
# set_seed()


class Car:
    #when initialization, it will assign source and destination nodes to allocate the car
    def __init__(self, source,destination, speed,pos_list,adj_matrix,adj_matrix_area=None,car_type=0):
        self.pos_list = pos_list
        self.adj_matrix = adj_matrix
        self.adj_matrix_area = adj_matrix_area
        self.speed = speed
        self.source = source
        self.destination = destination
        self.speed_vector = get_cordinate_by_node(self.destination, self.pos_list) - get_cordinate_by_node(self.source,self.pos_list)
        self.speed_vector = self.speed_vector/ np.linalg.norm(self.speed_vector)*speed #speed is should be a vector and the Amplitude of the vector should be a fixed value
        self.current_position = get_cordinate_by_node(self.source, self.pos_list) # position is a point along an edge
        self.car_type = car_type # 0 means car without area limitation, else means car with area limitation
        
    def move(self,time):
        #if the car reach the end of current road
        if np.linalg.norm(get_cordinate_by_node(self.destination, self.pos_list) - self.current_position) < self.speed*time:
            previous_source = self.source
            residual_time = time - np.linalg.norm(get_cordinate_by_node(self.destination, self.pos_list) - self.current_position)/self.speed
            self.source = self.destination
            if self.car_type ==0:
                self.destination = get_next_destination(self.destination,previous_source, self.speed_vector, self.pos_list, self.adj_matrix)
            else:
                self.destination = get_next_destination(self.destination,previous_source, self.speed_vector, self.pos_list, self.adj_matrix_area)

            self.speed_vector = get_cordinate_by_node(self.destination,self.pos_list) - get_cordinate_by_node(self.source,self.pos_list) #update speed
            self.speed_vector = self.speed_vector/ np.linalg.norm(self.speed_vector)*self.speed
            self.current_position = get_cordinate_by_node(self.source, self.pos_list) +  self.speed_vector*residual_time
        else:
            self.current_position = self.current_position +  self.speed_vector*time

# def calculate_distance(node1, node2):
#     # Calculate and return the distance between two nodes   
#     pass

def get_cordinate_by_node(node, pos_list):
    return np.array([pos_list[node][0], pos_list[node][1]])

def filter_edges_by_group(adj_matrix, groups):
    num_nodes = len(groups)
    new_adj_matrix = np.zeros_like(adj_matrix)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if groups[i] == groups[j]:  # Only keep edges within the same group
                new_adj_matrix[i][j] = adj_matrix[i][j]
    
    return new_adj_matrix

def get_next_destination(source,previous_source,speed_vector,pos_list, adj_matrix):
    # Select and return a random edge from the given node's edges
    neighbors = np.where(adj_matrix[source] == 1)[0]
    if len(neighbors)>1:# to make sure car dont reverse
        mask = neighbors != previous_source
        neighbors = neighbors[mask]
        if len(neighbors) == 1:
            current_node = neighbors[0]
        else:
            probabilities = get_road_choice_probability(source, neighbors,speed_vector,pos_list)
            current_node = np.random.choice(neighbors,p = probabilities)  # Move to a random neighbor  
    else: 
        current_node = previous_source
    return current_node


def get_road_choice_probability(source, neighbors,speed_vector,pos_list):
    #based on the current direction and the following road direction, to allocate the probabilty:
    direction = []
    for node in neighbors:
        direction.append((get_cordinate_by_node(node,pos_list) - get_cordinate_by_node(source,pos_list)))
    for i in range(len(direction)):
        direction[i] = cosine_similarity(direction[i],speed_vector)
    direction = np.array(direction)
    # Find the maximum element
    max_element = np.max(direction)
    
    # Calculate probabilities
    probabilities = np.ones_like(direction) * (0.5 / (len(direction) - 1))  # Equal share of the remaining probability
    probabilities[direction == max_element] = 0.5  # Assign 0.5 probability to the largest element
    return probabilities



def cosine_similarity(vec1,vec2):
    # Calculate the dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate the norm of each vector
    norm_vector1 = np.linalg.norm(vec1)
    norm_vector2 = np.linalg.norm(vec2)
    # Calculate cosine similarity
    return dot_product / (norm_vector1 * norm_vector2)

def generate_roadNet_pair_area_list( exp_dir, num_car, num_round, circle_radius = 100, step_time = 60, speed = 13.59,  County = 'New York', num_area = 10, car_type_list = [0]*100 ):
    # data = gpd.read_file(file_path)
    data = pd.read_csv("../NY_Data/NewYork.csv")
    gdf = data[data['County']== County ]
    # Initialize a directed graph if the roads have direction, else use nx.Graph()
    G = nx.DiGraph()
    speed = speed *0.00145/100
    circle_radius = circle_radius * 0.00145/100
    for index, row in gdf.iterrows():
        start_node = (row['StartLat'], row['StartLong'])
        end_node = (row['EndLat'], row['EndLong'])
    
        # Add nodes with attributes if you have any specific attributes to add
        G.add_node(start_node)
        G.add_node(end_node)
    
        # Add edge
        G.add_edge(start_node, end_node, length=row['Miles'])
        G.add_edge(end_node, start_node, length=row['Miles'])
    
        # If you have one-way streets, ensure you are adding edges in the correct direction.
        # If the streets are two-way, you'll need to add edges in both directions.
    
    # Now G contains your road network graph
    print('Now generating the road net of '+County)
    scc = list(nx.strongly_connected_components(G))
    largest_scc = max(scc, key=len)
    G_largest_scc = G.subgraph(largest_scc).copy()
    pos = {node: (node[1], node[0]) for node in G_largest_scc.nodes()}  # Create a position map with longitude, latitude
    # Now G contains your road network graph
    print(f"Number of nodes: {G_largest_scc.number_of_nodes()}")
    print(f"Number of edges: {G_largest_scc.number_of_edges()}")
    with open(exp_dir+'/configuration.txt','a') as file:
        file.write('The road net of '+County+'\n')
        file.write(f"Number of nodes: {G_largest_scc.number_of_nodes()}\n")
        file.write(f"Number of edges: {G_largest_scc.number_of_edges()}\n")
    # Generate the binary adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(G_largest_scc).todense()
    pos_list = list(pos.values())
    adj_matrix = np.array(adjacency_matrix)
    
    

    #generate car neighbour list:
    num_nodes = adj_matrix.shape[0]
      # unit/mile  # Set the radius of the circle of communication range
    
    #cluster the nodes into num of areas
    pos_array = np.array(pos_list)
    # Create and fit the KMeans model
    kmeans = KMeans(n_clusters=num_area, random_state=Random_SEED)
    kmeans.fit(pos_array)
    # Get the cluster labels for each position
    area_labels = kmeans.labels_

    # # Get the cluster centers
    # area_centers = kmeans.cluster_centers_

    adj_matrix_area = filter_edges_by_group(adj_matrix,area_labels)
    
    # Draw the car
    # random initial
    
    pair_list = []
    area_list = []
    car_list = []
    car_position = []
    for i in range(num_car):
        car_type = int(car_type_list[i])
        if car_type==0:
            source  = random.randint(0, num_nodes-1)
            neighbors = np.where(adj_matrix[source] == 1)[0]
        else:
            neighbors = np.array([])
            while(len(neighbors)==0):##########keep trying to get a non-empty neighbors
                source = np.random.choice(np.where(area_labels == car_type-1)[0])
                neighbors = np.where(adj_matrix_area[source] == 1)[0]
        #random chose destination from chosen source
        # if len(neighbors) == 0:
        #     print('ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        destination = np.random.choice(neighbors) 
        car_list.append(Car(source,destination,speed,pos_list,adj_matrix,adj_matrix_area,car_type))
        area_list.append([])
        car_position.append(car_list[i].current_position)
    car_position = np.array(car_position)
    
    # # update car_node
    # def update(car_node ,previous_node):
    #     for i in range(num_car):
    #         neighbors = np.where(adj_matrix[car_node[i]] == 1)[0]
    #         temp_node = car_node[i]
    #         if len(neighbors)>1:# to make sure car don't reverse where there are other routine avaliable
    #             node_to_remove = previous_node[i]
    #             mask = neighbors != node_to_remove
    #             neighbors = neighbors[mask]
    #         car_node[i] = random.choice(neighbors)  # Move to a random neighbor
    #         car_position[i] = [pos_list[car_node[i]][0], pos_list[car_node[i]][1]]
    #         previous_node[i] = temp_node
    #     return car_position,car_node, previous_node

    def caculate_pair(car_position,meeting_record):
        distance = np.sqrt(np.sum((car_position[:, np.newaxis, :] - car_position[np.newaxis, :, :]) ** 2, axis=-1))
        pair = []
        i, j = np.where(distance < circle_radius)
        for x, y in zip(i, j):
            if x<y:
                if meeting_record[x][y]==0:
                    pair.append((x,y))
                    meeting_record[x][y] = 1
        # for those not in the record: reset to not meet for the future use
        i, j = np.where(distance > circle_radius)
        for x, y in zip(i, j):
            if x<y:
                meeting_record[x][y] = 0
        random.shuffle(pair)
        return pair,meeting_record
    
    # loop to simulate cars' movement
    # a table to record which car meet with other car, to remove duplicated continuous meeting.
    meeting_record = np.zeros([num_car,num_car])
    for j in range(num_round*step_time):
        if j%step_time == 0:
            #reset the meeting information every step time, because at that time the new model finishes training
            meeting_record = np.zeros([num_car,num_car])
        for i in range(num_car):
            car_list[i].move(1)
            car_position[i] = car_list[i].current_position
    #     print(car_position[0]) 
        for i in range(num_car):
            area_list[i].append(area_labels[car_list[i].source])
        pair_info, meeting_record = caculate_pair(car_position,meeting_record)
        pair_list.append(pair_info)
    return pair_list, area_list

# def generate_roadNet_pair_area_list( exp_dir, num_car, num_round, circle_radius = 100, step_time = 60, speed = 13.59,  County = 'New York', num_area = 10):
#     # data = gpd.read_file(file_path)
#     data = pd.read_csv("../NY_Data/NewYork.csv")
#     gdf = data[data['County']== County ]
#     # Initialize a directed graph if the roads have direction, else use nx.Graph()
#     G = nx.DiGraph()
#     speed = speed *0.00145/100
#     circle_radius = circle_radius * 0.00145/100
#     for index, row in gdf.iterrows():
#         start_node = (row['StartLat'], row['StartLong'])
#         end_node = (row['EndLat'], row['EndLong'])
    
#         # Add nodes with attributes if you have any specific attributes to add
#         G.add_node(start_node)
#         G.add_node(end_node)
    
#         # Add edge
#         G.add_edge(start_node, end_node, length=row['Miles'])
#         G.add_edge(end_node, start_node, length=row['Miles'])
    
#         # If you have one-way streets, ensure you are adding edges in the correct direction.
#         # If the streets are two-way, you'll need to add edges in both directions.
    
#     # Now G contains your road network graph
#     print('Now generating the road net of '+County)
#     scc = list(nx.strongly_connected_components(G))
#     largest_scc = max(scc, key=len)
#     G_largest_scc = G.subgraph(largest_scc).copy()
#     pos = {node: (node[1], node[0]) for node in G_largest_scc.nodes()}  # Create a position map with longitude, latitude
#     # Now G contains your road network graph
#     print(f"Number of nodes: {G_largest_scc.number_of_nodes()}")
#     print(f"Number of edges: {G_largest_scc.number_of_edges()}")
#     with open(exp_dir+'/configuration.txt','a') as file:
#         file.write('The road net of '+County+'\n')
#         file.write(f"Number of nodes: {G_largest_scc.number_of_nodes()}\n")
#         file.write(f"Number of edges: {G_largest_scc.number_of_edges()}\n")
#     # Generate the binary adjacency matrix
#     adjacency_matrix = nx.adjacency_matrix(G_largest_scc).todense()
#     pos_list = list(pos.values())
#     adj_matrix = np.array(adjacency_matrix)
    
#     #generate car neighbour list:
#     num_nodes = adj_matrix.shape[0]
#       # unit/mile  # Set the radius of the circle of communication range
    
#     #cluster the nodes into num of areas
#     pos_array = np.array(pos_list)
#     # Create and fit the KMeans model
#     kmeans = KMeans(n_clusters=num_area, random_state=Random_SEED)
#     kmeans.fit(pos_array)
#     # Get the cluster labels for each position
#     area_labels = kmeans.labels_

#     # # Get the cluster centers
#     # area_centers = kmeans.cluster_centers_

    
#     # Draw the car
#     # random initial
    
#     pair_list = []
#     area_list = []
#     car_list = []
#     car_position = []
#     for i in range(num_car):
#         #random choose road (source and destination)
#         source  = random.randint(0, num_nodes-1)
#         neighbors = np.where(adj_matrix[source] == 1)[0]
#         destination = np.random.choice(neighbors) 
#         car_list.append(Car(source,destination,speed,pos_list,adj_matrix))
#         car_position.append(car_list[i].current_position)
#         area_list.append([])
#     car_position = np.array(car_position)
    
#     # update car_node
#     def update(car_node ,previous_node):
#         for i in range(num_car):
#             neighbors = np.where(adj_matrix[car_node[i]] == 1)[0]
#             temp_node = car_node[i]
#             if len(neighbors)>1:# to make sure car don't reverse where there are other routine avaliable
#                 node_to_remove = previous_node[i]
#                 mask = neighbors != node_to_remove
#                 neighbors = neighbors[mask]
#             car_node[i] = random.choice(neighbors)  # Move to a random neighbor
#             car_position[i] = [pos_list[car_node[i]][0], pos_list[car_node[i]][1]]
#             previous_node[i] = temp_node
#         return car_position,car_node, previous_node

#     def caculate_pair(car_position,meeting_record):
#         distance = np.sqrt(np.sum((car_position[:, np.newaxis, :] - car_position[np.newaxis, :, :]) ** 2, axis=-1))
#         pair = []
#         i, j = np.where(distance < circle_radius)
#         for x, y in zip(i, j):
#             if x<y:
#                 if meeting_record[x][y]==0:
#                     pair.append((x,y))
#                     meeting_record[x][y] = 1
#         # for those not in the record: reset to not meet for the future use
#         i, j = np.where(distance > circle_radius)
#         for x, y in zip(i, j):
#             if x<y:
#                 meeting_record[x][y] = 0
#         random.shuffle(pair)
#         return pair,meeting_record
    
#     # loop to simulate cars' movement
#     # a table to record which car meet with other car, to remove duplicated continuous meeting.
#     meeting_record = np.zeros([num_car,num_car])
#     for j in range(num_round*step_time):
#         if j%step_time == 0:
#             #reset the meeting information every step time, because at that time the new model finishes training
#             meeting_record = np.zeros([num_car,num_car])
#         for i in range(num_car):
#             car_list[i].move(1)
#             car_position[i] = car_list[i].current_position
#     #     print(car_position[0]) 
#         for i in range(num_car):
#             area_list[i].append(area_labels[car_list[i].source])
#         pair_info, meeting_record = caculate_pair(car_position,meeting_record)
#         pair_list.append(pair_info)
#     return pair_list, area_list

def generate_roadNet_pair_list_v2( exp_dir, num_car, num_round, circle_radius = 100, step_time = 60, speed = 13.59,  County = 'New York'):
    # data = gpd.read_file(file_path)
    data = pd.read_csv("./NY_Data/NewYork.csv")
    gdf = data[data['County']== County ]
    # Initialize a directed graph if the roads have direction, else use nx.Graph()
    G = nx.DiGraph()
    speed = speed *0.00145/100
    circle_radius = circle_radius * 0.00145/100
    for index, row in gdf.iterrows():
        start_node = (row['StartLat'], row['StartLong'])
        end_node = (row['EndLat'], row['EndLong'])
    
        # Add nodes with attributes if you have any specific attributes to add
        G.add_node(start_node)
        G.add_node(end_node)
    
        # Add edge
        G.add_edge(start_node, end_node, length=row['Miles'])
        G.add_edge(end_node, start_node, length=row['Miles'])
    
        # If you have one-way streets, ensure you are adding edges in the correct direction.
        # If the streets are two-way, you'll need to add edges in both directions.
    
    # Now G contains your road network graph
    print('Now generating the road net of '+County)
    scc = list(nx.strongly_connected_components(G))
    largest_scc = max(scc, key=len)
    G_largest_scc = G.subgraph(largest_scc).copy()
    pos = {node: (node[1], node[0]) for node in G_largest_scc.nodes()}  # Create a position map with longitude, latitude
    # Now G contains your road network graph
    print(f"Number of nodes: {G_largest_scc.number_of_nodes()}")
    print(f"Number of edges: {G_largest_scc.number_of_edges()}")
    with open(exp_dir+'/configuration.txt','a') as file:
        file.write('The road net of '+County+'\n')
        file.write(f"Number of nodes: {G_largest_scc.number_of_nodes()}\n")
        file.write(f"Number of edges: {G_largest_scc.number_of_edges()}\n")
    # Generate the binary adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(G_largest_scc).todense()
    pos_list = list(pos.values())
    adj_matrix = np.array(adjacency_matrix)
    
    #generate car neighbour list:
    num_nodes = adj_matrix.shape[0]
      # unit/mile  # Set the radius of the circle of communication range
    
    
    
    # Draw the car
    # random initial
    
    pair_list = []
    car_list = []
    car_position = []
    for i in range(num_car):
        #random choose road (source and destination)
        source  = random.randint(0, num_nodes-1)
        neighbors = np.where(adj_matrix[source] == 1)[0]
        destination = np.random.choice(neighbors) 
        car_list.append(Car(source,destination,speed,pos_list,adj_matrix))
        car_position.append(car_list[i].current_position)
    car_position = np.array(car_position)
    
    # update car_node
    def update(car_node ,previous_node):
        for i in range(num_car):
            neighbors = np.where(adj_matrix[car_node[i]] == 1)[0]
            temp_node = car_node[i]
            if len(neighbors)>1:# to make sure car don't reverse where there are other routine avaliable
                node_to_remove = previous_node[i]
                mask = neighbors != node_to_remove
                neighbors = neighbors[mask]
            car_node[i] = random.choice(neighbors)  # Move to a random neighbor
            car_position[i] = [pos_list[car_node[i]][0], pos_list[car_node[i]][1]]
            previous_node[i] = temp_node
        return car_position,car_node, previous_node

    def caculate_pair(car_position):
        distance = np.sqrt(np.sum((car_position[:, np.newaxis, :] - car_position[np.newaxis, :, :]) ** 2, axis=-1))
        pair = []
        i, j = np.where((distance < circle_radius) & (distance!= 0))
        for x, y in zip(i, j):
            if x<y:
                pair.append((x,y))
        random.shuffle(pair)
        return pair
    
    # loop to simulate cars' movement
    for j in range(num_round*step_time):
        for i in range(len(car_list)):
            car_list[i].move(1)
            car_position[i] = car_list[i].current_position
        pair_list.append(caculate_pair(car_position))
    return pair_list



def generate_roadNet_pair_list( exp_dir, num_car, num_round, circle_radius = 0.02, County = 'New York', communication_interval = 1):
    data = pd.read_csv("./NY_Data/NewYork.csv")
    gdf = data[data['County']== County ]
    # Initialize a directed graph if the roads have direction, else use nx.Graph()
    G = nx.DiGraph()
    
    for index, row in gdf.iterrows():
        start_node = (row['StartLat'], row['StartLong'])
        end_node = (row['EndLat'], row['EndLong'])
    
        # Add nodes with attributes if you have any specific attributes to add
        G.add_node(start_node)
        G.add_node(end_node)
    
        # Add edge
        G.add_edge(start_node, end_node, length=row['Miles'])
        G.add_edge(end_node, start_node, length=row['Miles'])
    
        # If you have one-way streets, ensure you are adding edges in the correct direction.
        # If the streets are two-way, you'll need to add edges in both directions.
    
    # Now G contains your road network graph
    print('Now generating the road net of '+County)
    scc = list(nx.strongly_connected_components(G))
    largest_scc = max(scc, key=len)
    G_largest_scc = G.subgraph(largest_scc).copy()
    pos = {node: (node[1], node[0]) for node in G_largest_scc.nodes()}  # Create a position map with longitude, latitude
    # Now G contains your road network graph
    print(f"Number of nodes: {G_largest_scc.number_of_nodes()}")
    print(f"Number of edges: {G_largest_scc.number_of_edges()}")
    with open(exp_dir+'/configuration.txt','a') as file:
        file.write('The road net of '+County+'\n')
        file.write(f"Number of nodes: {G_largest_scc.number_of_nodes()}\n")
        file.write(f"Number of edges: {G_largest_scc.number_of_edges()}\n")
    # Generate the binary adjacency matrix
    adjacency_matrix = nx.adjacency_matrix(G_largest_scc).todense()
    pos_list = list(pos.values())
    adj_matrix = np.array(adjacency_matrix)
    
    #generate car neighbour list:
    num_nodes = adj_matrix.shape[0]
      # unit/mile  # Set the radius of the circle of communication range
    
    
    
    # Draw the car
    # random initial
    car_node = [random.randint(0, num_nodes-1) for _ in range(num_car)]
    previous_node = copy.deepcopy(car_node)
    pair_list = []
    car_position = []
    for i in range(num_car):
        car_position.append([pos_list[car_node[i]][0], pos_list[car_node[i]][1]])
    car_position = np.array(car_position)
    
    # update car_node
    def update(car_node ,previous_node):
        for i in range(num_car):
            neighbors = np.where(adj_matrix[car_node[i]] == 1)[0]
            temp_node = car_node[i]
            if len(neighbors)>1:# to make sure car don't reverse where there are other routine avaliable
                node_to_remove = previous_node[i]
                mask = neighbors != node_to_remove
                neighbors = neighbors[mask]
            car_node[i] = random.choice(neighbors)  # Move to a random neighbor
            car_position[i] = [pos_list[car_node[i]][0], pos_list[car_node[i]][1]]
            previous_node[i] = temp_node
        return car_position,car_node, previous_node

    def caculate_pair(car_position):
        distance = np.sqrt(np.sum((car_position[:, np.newaxis, :] - car_position[np.newaxis, :, :]) ** 2, axis=-1))
        pair = []
        i, j = np.where((distance < circle_radius) & (distance!= 0))
        for x, y in zip(i, j):
            if x<y:
                pair.append((x,y))
        random.shuffle(pair)
        return pair
    
    # loop to simulate cars' movement
    for j in range(num_round):
        for k in range(communication_interval):
            car_position,car_node, previous_node = update(car_node, previous_node)
    #     print(car_position[0]) 
        pair_list.append(caculate_pair(car_position))
    return pair_list