import random
import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from sklearn.model_selection import train_test_split
# Normalize the features
from sklearn.preprocessing import StandardScaler
import seed_setter
import os

# Call the set_seed function at the start
random_seed = seed_setter.set_seed()


class PermutedMNIST(datasets.MNIST):

    def __init__(self, root="~/.torch/data/mnist", train=True, permute_idx=None):
        super(PermutedMNIST, self).__init__(root, train, download=True)
        assert len(permute_idx) == 28 * 28
        if self.train:
            self.train_data = torch.stack([img.float().view(-1)[permute_idx] / 255 for img in self.train_data])
        else:
            self.test_data = torch.stack([img.float().view(-1)[permute_idx] / 255 for img in self.test_data])

    def __getitem__(self, index):

        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img for img in self.train_data[sample_idx]]


def initial_training_subset(train_dataset, target_class, initial_size):
    class_indices = np.where(train_dataset.targets == target_class)[0]
    initial_indices = np.random.choice(class_indices, initial_size, replace=False)
    return initial_indices.tolist()

def get_dataloader_by_indices(train_dataset, indices, batch_size):
    # Create the initial subset and DataLoader
    subset = Subset(train_dataset, indices)
    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return train_loader

def update_training_subset(current_indices, train_dataset, target_class, max_num, add_fraction=0.1):
    # Remove the oldest data points (based on remove_fraction)
    num_remove = max_num*(1-add_fraction) - len(current_indices)
    if num_remove > 0:
        current_subset = current_subset[num_remove:]
    # Add new samples from the target class
    full_labels = train_dataset.targets.numpy()
    class_indices = np.where(full_labels == target_class)[0]
    num_add = int(max_num* add_fraction)
    new_samples = np.random.choice(class_indices, num_add, replace=False)
    # Append new samples to the current indices
    current_indices = np.concatenate((current_indices, new_samples))

    return current_indices.tolist()


def initial_mnist(batch_size, test_ratio):
    data_dir = '../data/'
    apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)
    

    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)
    
    test_idx = [i for i in range(len(test_dataset))]
    y_test = np.array([test_dataset[i][1] for i in test_idx])
    _, test_sampled_idx = train_test_split(test_idx, test_size=test_ratio, stratify=y_test, random_state=random_seed )
    sub_test_loader = DataLoader(DatasetSplit(test_dataset, test_sampled_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(DatasetSplit(test_dataset,test_idx),
                                                     batch_size=batch_size, shuffle = False)
    
    all_idxs = [i for i in range(len(train_dataset))]
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    return train_dataset, sub_test_loader, test_loader, full_loader


def load_all_data(num_user=120):
	# dataset append and split
    NUM_OF_CLASS = 5
    DIMENSION_OF_FEATURE = 900
    class_set = ['Call','Hop','typing','Walk','Wave']
    coll_class = []
    coll_label = []
    for user_id in range(1,num_user+1):
        for class_id in range(NUM_OF_CLASS):
            read_path = '../data/large_scale_HARBox/' +  str(user_id) + '/' + str(class_set[class_id]) + '_train' + '.txt'
            if os.path.exists(read_path):
                temp_original_data = np.loadtxt(read_path)
                temp_reshape = temp_original_data.reshape(-1, 100, 10)
                temp_coll = temp_reshape[:, :, 1:10].reshape(-1, DIMENSION_OF_FEATURE)
                count_img = temp_coll.shape[0]
                temp_label = class_id * np.ones(count_img)
                # print(temp_original_data.shape)
                # print(temp_coll.shape)
                coll_class.extend(temp_coll)
                coll_label.extend(temp_label)

				# total_class += 1
    coll_class = np.array(coll_class)
    coll_label = np.array(coll_label)

    print(coll_class.shape)
    print(coll_label.shape)

    return coll_class, coll_label

class HARBoxDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
def get_harbox_iid( num_car, batch_size,test_ratio):
    #Harbox
    x_coll, y_coll = load_all_data()
    scaler = StandardScaler()
    features = scaler.fit_transform(x_coll)
    labels = torch.tensor(y_coll, dtype=torch.long)
    # Create an instance of the dataset
    dataset = HARBoxDataset(features, labels)

    # Define the sizes for training and test sets
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # x_train,x_test,y_train,y_test = train_test_split(x_coll,y_coll,test_size = 0.1, random_state = random_seed)

    
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)
    #iid distribution
    num_items = int(len(train_dataset)/num_car)
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    #all_idxs = np.array(all_idxs)
    for i in range(num_car):
        dict_tasks[i] = set(np.random.choice(all_idxs, num_items,
                                                 replace=False))
        all_idxs = list(set(all_idxs) - dict_tasks[i])
    all_idxs = [i for i in range(len(train_dataset))]
        
    train_loader, sub_test_loader,full_test_loader = get_permute_dataset(train_dataset,test_dataset,test_idx, dict_tasks, batch_size,test_ratio)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader

def get_harbox_non_iid(shards_allocation_list, num_car,batch_size,test_ratio):
    #Harbox
    x_coll, y_coll = load_all_data()
    scaler = StandardScaler()
    features = scaler.fit_transform(x_coll)
    labels = torch.tensor(y_coll, dtype=torch.long)
    # Create an instance of the dataset
    dataset = HARBoxDataset(features, labels)

    # Define the sizes for training and test sets
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)
    # dict_tasks[0] = all_idxs[idx_0]
    # dict_tasks[1] = all_idxs[idx_1] 
    # dict_tasks[2] = all_idxs[idx_2]
    # dict_tasks[3] = all_idxs[idx_3] 
    
    # balanced non_iid data
    num_shards, num_imgs = sum(shards_allocation_list), int(len(all_idxs)/(sum(shards_allocation_list)))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_car)}
    idxs = all_idxs#np.arange(num_shards*num_imgs)
    # Extract labels from the train_dataset
    train_labels = [label for _, label in train_dataset]

    # Convert the list of labels to a tensor
    labels = torch.tensor(train_labels)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign  shards/client
    for i in range(num_car):
        # shards_per_user = int(num_shards/num_car)
        print(idx_shard)
        rand_set = set(np.random.choice(idx_shard, shards_allocation_list[i], replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    dict_tasks = dict_users    
    
    idxs = [int(i) for i in dict_tasks[0]]
    # sample_idx = random.sample(range(len(train_dataset)), 200)
    # sample_set = []
    # for idx in sample_idx:
    #     batch_set = []
    #     for i in range(10):
    #         image, label = train_dataset[idx]
    #         # batch_set.append(torch.tensor(image))
    #         batch_set.append(image.clone().detach().requires_grad_(True))
    #     sample_set.append(torch.stack(batch_set))
        
    train_loader, sub_test_loader,full_test_loader = get_permute_dataset(train_dataset,test_dataset,test_idx, dict_tasks, batch_size,test_ratio)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader


def get_harbox_dirichlet(alpha, num_car,batch_size,test_ratio):
    #Harbox
    x_coll, y_coll = load_all_data()
    scaler = StandardScaler()
    features = scaler.fit_transform(x_coll)
    labels = torch.tensor(y_coll, dtype=torch.long)
    # Create an instance of the dataset
    dataset = HARBoxDataset(features, labels)

    # Define the sizes for training and test sets
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)
    
    # Extract labels from the train_dataset
    train_labels = [label for _, label in train_dataset]
    data_indices = [np.where(np.array(train_labels) == i)[0] for i in range(5)]
    client_indices = [[] for _ in range(num_car)]
    for indices in data_indices:
        # Draw samples from a Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_car))
        
        # Calculate the number of data points for each client
        # proportions = np.array([p * (len(indices) / sum(proportions)) for p in proportions]).astype(int)
        proportions = (proportions * len(indices)).astype(int)

        # Ensure total sum of indices is equal to the original total
        proportions[-1] = len(indices) - sum(proportions[:-1])
        
        # Assign indices to each client based on the calculated proportions
        np.random.shuffle(indices)
        start = 0
        for client, num_indices in zip(client_indices, proportions):
            end = start + num_indices
            client.extend(indices[start:end])
            start = end

    # Check and redistribute if any client has zero total samples
    min_samples_per_client = 1  # Minimum samples any client should have
    for i, client in enumerate(client_indices):
        if len(client) < min_samples_per_client:
            # Find a client to take samples from
            donor = max(enumerate(client_indices), key=lambda x: len(x[1]))[0]
            client_indices[i].extend(client_indices[donor][:min_samples_per_client])
            client_indices[donor] = client_indices[donor][min_samples_per_client:]
    # Create client datasets
    client_datasets = [torch.utils.data.Subset(train_dataset, indices) for indices in client_indices]

    train_loader = [torch.utils.data.DataLoader(traindataset, batch_size, shuffle=True) for traindataset in client_datasets]
    # Stratified sampling for the test set
    y_test = np.array([test_dataset[i][1] for i in test_idx])
    _, test_sampled_idx = train_test_split(test_idx, test_size=test_ratio, stratify=y_test, random_state=random_seed )
    sub_test_loader = torch.utils.data.DataLoader(DatasetSplit(test_dataset, test_sampled_idx), batch_size=batch_size, shuffle=False)
    full_test_loader = torch.utils.data.DataLoader(DatasetSplit(test_dataset,test_idx),batch_size=batch_size, shuffle = False)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader

def get_fashionmnist_iid( num_car, batch_size,test_ratio):
    #MNIST
    data_dir = '../data/'
    apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
    
    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)
    
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)
    #iid distribution
    num_items = int(len(train_dataset)/num_car)
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    #all_idxs = np.array(all_idxs)
    for i in range(num_car):
        dict_tasks[i] = set(np.random.choice(all_idxs, num_items,
                                                 replace=False))
        all_idxs = list(set(all_idxs) - dict_tasks[i])
    all_idxs = [i for i in range(len(train_dataset))]
        
    train_loader, sub_test_loader,full_test_loader = get_permute_dataset(train_dataset,test_dataset,test_idx, dict_tasks, batch_size,test_ratio)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader

def get_fashionmnist_non_iid(shards_allocation_list, num_car,batch_size,test_ratio):
    #MNIST
    data_dir = '../data/'
    apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
    
    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)
    
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)
    
    
    # idx_0 = train_dataset.targets<3
    # idx_1 = (train_dataset.targets<5)&(train_dataset.targets>=3)
    # idx_2 = (train_dataset.targets<8)&(train_dataset.targets>=5)
    # idx_3 = train_dataset.targets>=8
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)
    # dict_tasks[0] = all_idxs[idx_0]
    # dict_tasks[1] = all_idxs[idx_1] 
    # dict_tasks[2] = all_idxs[idx_2]
    # dict_tasks[3] = all_idxs[idx_3] 
    
    # balanced non_iid data
    num_shards, num_imgs = sum(shards_allocation_list), int(60000/(sum(shards_allocation_list)))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_car)}
    idxs = np.arange(num_shards*num_imgs)
    labels = train_dataset.targets.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign  shards/client
    for i in range(num_car):
        # shards_per_user = int(num_shards/num_car)
        print(idx_shard)
        rand_set = set(np.random.choice(idx_shard, shards_allocation_list[i], replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    dict_tasks = dict_users    
    
    idxs = [int(i) for i in dict_tasks[0]]
    # sample_idx = random.sample(range(len(train_dataset)), 200)
    # sample_set = []
    # for idx in sample_idx:
    #     batch_set = []
    #     for i in range(10):
    #         image, label = train_dataset[idx]
    #         # batch_set.append(torch.tensor(image))
    #         batch_set.append(image.clone().detach().requires_grad_(True))
    #     sample_set.append(torch.stack(batch_set))
        
    train_loader, sub_test_loader,full_test_loader = get_permute_dataset(train_dataset,test_dataset,test_idx, dict_tasks, batch_size,test_ratio)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader

def get_fashionmnist_dirichlet(alpha, num_car,batch_size,test_ratio):
    # Lower alpha increases heterogeneity
    data_dir = '../data/'
    apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
    
    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)
    
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)


    data_indices = [np.where(np.array(train_dataset.targets) == i)[0] for i in range(10)]
    client_indices = [[] for _ in range(num_car)]
    for indices in data_indices:
        # Draw samples from a Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_car))
        
        # Calculate the number of data points for each client
        # proportions = np.array([p * (len(indices) / sum(proportions)) for p in proportions]).astype(int)
        proportions = (proportions * len(indices)).astype(int)

        # Ensure total sum of indices is equal to the original total
        proportions[-1] = len(indices) - sum(proportions[:-1])
        
        # Assign indices to each client based on the calculated proportions
        np.random.shuffle(indices)
        start = 0
        for client, num_indices in zip(client_indices, proportions):
            end = start + num_indices
            client.extend(indices[start:end])
            start = end

    # Check and redistribute if any client has zero total samples
    min_samples_per_client = 1  # Minimum samples any client should have
    for i, client in enumerate(client_indices):
        if len(client) < min_samples_per_client:
            # Find a client to take samples from
            donor = max(enumerate(client_indices), key=lambda x: len(x[1]))[0]
            client_indices[i].extend(client_indices[donor][:min_samples_per_client])
            client_indices[donor] = client_indices[donor][min_samples_per_client:]
    # Create client datasets
    client_datasets = [torch.utils.data.Subset(train_dataset, indices) for indices in client_indices]

    train_loader = [torch.utils.data.DataLoader(traindataset, batch_size, shuffle=True) for traindataset in client_datasets]
    # Stratified sampling for the test set
    y_test = np.array([test_dataset[i][1] for i in test_idx])
    _, test_sampled_idx = train_test_split(test_idx, test_size=test_ratio, stratify=y_test, random_state=random_seed )
    sub_test_loader = torch.utils.data.DataLoader(DatasetSplit(test_dataset, test_sampled_idx), batch_size=batch_size, shuffle=False)
    full_test_loader = torch.utils.data.DataLoader(DatasetSplit(test_dataset,test_idx),batch_size=batch_size, shuffle = False)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader

def get_fashionmnist_area(shards_allocation_list,batch_size,test_ratio,car_type_list,target_labels):
    #MNIST
    data_dir = '../data/'
    apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
    
    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)
    
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)
    car_type_list = np.array(car_type_list)
    count_list = []
    count_list.append(np.sum(car_type_list == 1))
    count_list.append(np.sum(car_type_list == 2))
    count_list.append(np.sum(car_type_list == 3))
    shard_ac_1 = shards_allocation_list[0:count_list[0]]
    shard_ac_2 = shards_allocation_list[count_list[0]:count_list[0]+count_list[1]]
    shard_ac_3 = shards_allocation_list[count_list[0]+count_list[1]:count_list[0]+count_list[1]+count_list[2]]
    shard_ac = [shard_ac_1,shard_ac_2,shard_ac_3]


    subset_targets = []
    subset_train_dataset = []
    subset_indices = []
    for i in range(len(target_labels)):
        subset_indices.append([])

    target_group_mapping = []
    for i in range(10):
        target_group_mapping.append([])
        for j in range(len(target_labels)):
            if i in target_labels[j]:
                target_group_mapping[i].append(j)

    for i in range(10):
        indice = np.where(train_dataset.targets.numpy() == i)[0]
        # Shuffle the indices
        np.random.shuffle(indice)
        n_total = len(indice)
        for j in range(len(target_group_mapping[i])):
            n = int(n_total/len(target_group_mapping[i]))
            subset_indices[target_group_mapping[i][j]] += indice[:n].tolist()
            indice = indice[n:]
        

    for i in range(len(target_labels)):
        # Get the indices of the samples with the target labels
        # indices = np.where(np.isin(train_dataset.targets.numpy(), target_labels[i]))[0]
        indices = np.array(subset_indices[i])
        # # Create a subset of the dataset
        subset_train_dataset.append(Subset(train_dataset, indices))
        subset_targets.append( train_dataset.targets.numpy()[indices])

    # subset_targets = []
    # subset_train_dataset = []
    # for i in range(3):
    #     # Get the indices of the samples with the target labels
    #     indices = np.where(np.isin(train_dataset.targets.numpy(), target_labels[i]))[0]
    #     # # Create a subset of the dataset
    #     subset_train_dataset.append(Subset(train_dataset, indices))
    #     subset_targets.append( train_dataset.targets.numpy()[indices])

    # idx_0 = train_dataset.targets<=3
    # idx_1 = (train_dataset.targets<7)&(train_dataset.targets>=4)
    # idx_2 = train_dataset.targets>=7

    all_idxs = [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)


    train_loader = {}
    j = 0
    for k in range(3):
        dict_tasks = {}
        # balanced non_iid data
        num_shards, num_imgs = sum(shard_ac[k]), int(len(subset_targets[k])/(sum(shard_ac[k])))
        idx_shard = [i for i in range(num_shards)]
        idxs = np.arange(num_shards*num_imgs)
        labels = subset_targets[k][:num_shards*num_imgs]
        dict_users = {i: np.array([]) for i in range(count_list[k])}
        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]
        # divide and assign  shards/client
        for i in range(count_list[k]):
            # shards_per_user = int(num_shards/num_car)
            print(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_ac[k][i], replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        dict_tasks = dict_users
        sub_train_loader, _,_ = get_permute_dataset(subset_train_dataset[k],test_dataset,test_idx, dict_tasks, batch_size,test_ratio)
        for i in range(count_list[k]):
            train_loader[j] = sub_train_loader[i]
            j += 1

    # dict_tasks[0] = all_idxs[idx_0]
    # dict_tasks[1] = all_idxs[idx_1] 
    # dict_tasks[2] = all_idxs[idx_2]


    # j = 0
    # dict_users = {i: np.array([]) for i in range(np.sum(car_type_list != 0))}

    #based on area to distribute the data
    # balanced non_iid data
    # all_labels = train_dataset.targets.numpy()
    
    
    # idxs = [int(i) for i in dict_tasks[0]]
    # sample_idx = random.sample(range(len(train_dataset)), 200)
    # sample_set = []
    # for idx in sample_idx:
    #     batch_set = []
    #     for i in range(10):
    #         image, label = train_dataset[idx]
    #         # batch_set.append(torch.tensor(image))
    #         batch_set.append(image.clone().detach().requires_grad_(True))
    #     sample_set.append(torch.stack(batch_set))
        
    _, sub_test_loader,full_test_loader = get_permute_dataset(train_dataset,test_dataset,test_idx, dict_tasks, batch_size,test_ratio)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader


def get_mnist_iid(num_car,batch_size,test_ratio):
    #MNIST
    data_dir = '../data/'
    apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)
    
    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)
    
    
    
    # idx_0 = train_dataset.targets<3
    # idx_1 = (train_dataset.targets<5)&(train_dataset.targets>=3)
    # idx_2 = (train_dataset.targets<8)&(train_dataset.targets>=5)
    # idx_3 = train_dataset.targets>=8
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)
    # dict_tasks[0] = all_idxs[idx_0]
    # dict_tasks[1] = all_idxs[idx_1] 
    # dict_tasks[2] = all_idxs[idx_2]
    # dict_tasks[3] = all_idxs[idx_3] 
    
    #iid distribution
    num_items = int(len(train_dataset)/num_car)
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    #all_idxs = np.array(all_idxs)
    for i in range(num_car):
        dict_tasks[i] = set(np.random.choice(all_idxs, num_items,
                                                 replace=False))
        all_idxs = list(set(all_idxs) - dict_tasks[i])
    all_idxs = [i for i in range(len(train_dataset))]
    
    # # balanced non_iid data
    # num_shards, num_imgs = shards_per_user*num_car, int(60000/(shards_per_user*num_car))
    # idx_shard = [i for i in range(num_shards)]
    # dict_users = {i: np.array([]) for i in range(num_car)}
    # idxs = np.arange(num_shards*num_imgs)
    # labels = train_dataset.targets.numpy()
    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs = idxs_labels[0, :]
    # # divide and assign  shards/client
    # for i in range(num_car):
    #     # shards_per_user = int(num_shards/num_car)
    #     print(idx_shard)
    #     rand_set = set(np.random.choice(idx_shard, shards_per_user, replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # dict_tasks = dict_users    
    
    # idxs = [int(i) for i in dict_tasks[0]]
    # sample_idx = random.sample(range(len(train_dataset)), 200)
    # sample_set = []
    # for idx in sample_idx:
    #     batch_set = []
    #     for i in range(10):
    #         image, label = train_dataset[idx]
    #         # batch_set.append(torch.tensor(image))
    #         batch_set.append(image.clone().detach().requires_grad_(True))
    #     sample_set.append(torch.stack(batch_set))
        
    train_loader, sub_test_loader,full_test_loader = get_permute_dataset(train_dataset,test_dataset,test_idx, dict_tasks, batch_size,test_ratio)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader

# def get_mnist_dataset(shards_per_user, num_car,batch_size):
    #MNIST
    data_dir = '../data/'
    apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root=data_dir)
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)
    
    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)
    
    
    
    # idx_0 = train_dataset.targets<3
    # idx_1 = (train_dataset.targets<5)&(train_dataset.targets>=3)
    # idx_2 = (train_dataset.targets<8)&(train_dataset.targets>=5)
    # idx_3 = train_dataset.targets>=8
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)
    # dict_tasks[0] = all_idxs[idx_0]
    # dict_tasks[1] = all_idxs[idx_1] 
    # dict_tasks[2] = all_idxs[idx_2]
    # dict_tasks[3] = all_idxs[idx_3] 
    
    # balanced non_iid data
    num_shards, num_imgs = shards_per_user*num_car, int(60000/(shards_per_user*num_car))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_car)}
    idxs = np.arange(num_shards*num_imgs)
    labels = train_dataset.targets.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign  shards/client
    for i in range(num_car):
        # shards_per_user = int(num_shards/num_car)
        print(idx_shard)
        rand_set = set(np.random.choice(idx_shard, shards_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    dict_tasks = dict_users    
    
    idxs = [int(i) for i in dict_tasks[0]]
    # sample_idx = random.sample(range(len(train_dataset)), 200)
    # sample_set = []
    # for idx in sample_idx:
    #     batch_set = []
    #     for i in range(10):
    #         image, label = train_dataset[idx]
    #         # batch_set.append(torch.tensor(image))
    #         batch_set.append(image.clone().detach().requires_grad_(True))
    #     sample_set.append(torch.stack(batch_set))
        
    train_loader, test_loader = get_permute_dataset(train_dataset,test_dataset,test_idx, dict_tasks, batch_size)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, test_loader, full_loader



def get_mnist_non_iid(shards_allocation_list, num_car,batch_size,test_ratio):
    #MNIST
    data_dir = '../data/'
    apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root=data_dir, download=True)
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)
    
    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)
    
    
    
    # idx_0 = train_dataset.targets<3
    # idx_1 = (train_dataset.targets<5)&(train_dataset.targets>=3)
    # idx_2 = (train_dataset.targets<8)&(train_dataset.targets>=5)
    # idx_3 = train_dataset.targets>=8
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)
    # dict_tasks[0] = all_idxs[idx_0]
    # dict_tasks[1] = all_idxs[idx_1] 
    # dict_tasks[2] = all_idxs[idx_2]
    # dict_tasks[3] = all_idxs[idx_3] 
    
    # balanced non_iid data
    num_shards, num_imgs = sum(shards_allocation_list), int(60000/(sum(shards_allocation_list)))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_car)}
    idxs = np.arange(num_shards*num_imgs)
    labels = train_dataset.targets.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign  shards/client
    for i in range(num_car):
        # shards_per_user = int(num_shards/num_car)
        print(idx_shard)
        rand_set = set(np.random.choice(idx_shard, shards_allocation_list[i], replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    dict_tasks = dict_users    
    
    idxs = [int(i) for i in dict_tasks[0]]
    # sample_idx = random.sample(range(len(train_dataset)), 200)
    # sample_set = []
    # for idx in sample_idx:
    #     batch_set = []
    #     for i in range(10):
    #         image, label = train_dataset[idx]
    #         # batch_set.append(torch.tensor(image))
    #         batch_set.append(image.clone().detach().requires_grad_(True))
    #     sample_set.append(torch.stack(batch_set))
        
    train_loader, sub_test_loader,full_test_loader = get_permute_dataset(train_dataset,test_dataset,test_idx, dict_tasks, batch_size,test_ratio)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader



def get_mnist_area(shards_allocation_list,batch_size,test_ratio,car_type_list,target_labels):
    #MNIST
    data_dir = '../data/'
    apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root=data_dir, download=True)
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)
    
    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)
    car_type_list = np.array(car_type_list)
    count_list = []
    count_list.append(np.sum(car_type_list == 1))
    count_list.append(np.sum(car_type_list == 2))
    count_list.append(np.sum(car_type_list == 3))
    shard_ac_1 = shards_allocation_list[0:count_list[0]]
    shard_ac_2 = shards_allocation_list[count_list[0]:count_list[0]+count_list[1]]
    shard_ac_3 = shards_allocation_list[count_list[0]+count_list[1]:count_list[0]+count_list[1]+count_list[2]]
    shard_ac = [shard_ac_1,shard_ac_2,shard_ac_3]

    # target_labels = [[0,1,2,3],[4,5,6],[7,8,9]]
    # target_labels_count = np.zeros(10)
    # for i in range(len(target_labels)):
    #     target_labels_count[target_labels[i]] += 1
    subset_targets = []
    subset_train_dataset = []
    subset_indices = []
    for i in range(len(target_labels)):
        subset_indices.append([])

    target_group_mapping = []
    for i in range(10):
        target_group_mapping.append([])
        for j in range(len(target_labels)):
            if i in target_labels[j]:
                target_group_mapping[i].append(j)
    for i in range(10):
        indice = np.where(train_dataset.targets.numpy() == i)[0]
        # Shuffle the indices
        np.random.shuffle(indice)
        n_total = len(indice)
        for j in range(len(target_group_mapping[i])):
            n = int(n_total/len(target_group_mapping[i]))
            subset_indices[target_group_mapping[i][j]] += indice[:n].tolist()
            indice = indice[n:]
        

    for i in range(len(target_labels)):
        # Get the indices of the samples with the target labels
        # indices = np.where(np.isin(train_dataset.targets.numpy(), target_labels[i]))[0]
        indices = np.array(subset_indices[i])
        # # Create a subset of the dataset
        subset_train_dataset.append(Subset(train_dataset, indices))
        subset_targets.append( train_dataset.targets.numpy()[indices])


    # for i in range(len(target_labels)):
    #     # Get the indices of the samples with the target labels
    #     indices = np.where(np.isin(train_dataset.targets.numpy(), target_labels[i]))[0]
    #     # # Create a subset of the dataset
    #     subset_train_dataset.append(Subset(train_dataset, indices))
    #     subset_targets.append( train_dataset.targets.numpy()[indices])

    # idx_0 = train_dataset.targets<=3
    # idx_1 = (train_dataset.targets<7)&(train_dataset.targets>=4)
    # idx_2 = train_dataset.targets>=7

    all_idxs = [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)


    train_loader = {}
    j = 0
    for k in range(len(target_labels)):
        dict_tasks = {}
        # balanced non_iid data
        num_shards, num_imgs = sum(shard_ac[k]), int(len(subset_targets[k])/(sum(shard_ac[k])))
        idx_shard = [i for i in range(num_shards)]
        idxs = np.arange(num_shards*num_imgs)
        labels = subset_targets[k][:num_shards*num_imgs]
        dict_users = {i: np.array([]) for i in range(count_list[k])}
        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]
        # divide and assign  shards/client
        for i in range(count_list[k]):
            # shards_per_user = int(num_shards/num_car)
            print(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_ac[k][i], replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        dict_tasks = dict_users
        sub_train_loader, _,_ = get_permute_dataset(subset_train_dataset[k],test_dataset,test_idx, dict_tasks, batch_size,test_ratio)
        for i in range(count_list[k]):
            train_loader[j] = sub_train_loader[i]
            j += 1

    # dict_tasks[0] = all_idxs[idx_0]
    # dict_tasks[1] = all_idxs[idx_1] 
    # dict_tasks[2] = all_idxs[idx_2]


    # j = 0
    # dict_users = {i: np.array([]) for i in range(np.sum(car_type_list != 0))}

    #based on area to distribute the data
    # balanced non_iid data
    # all_labels = train_dataset.targets.numpy()
    
    
    # idxs = [int(i) for i in dict_tasks[0]]
    # sample_idx = random.sample(range(len(train_dataset)), 200)
    # sample_set = []
    # for idx in sample_idx:
    #     batch_set = []
    #     for i in range(10):
    #         image, label = train_dataset[idx]
    #         # batch_set.append(torch.tensor(image))
    #         batch_set.append(image.clone().detach().requires_grad_(True))
    #     sample_set.append(torch.stack(batch_set))
        
    _, sub_test_loader,full_test_loader = get_permute_dataset(train_dataset,test_dataset,test_idx, dict_tasks, batch_size,test_ratio)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader



def get_mnist_dirichlet(alpha, num_car,batch_size,test_ratio):
    # Lower alpha increases heterogeneity
    data_dir = '../data/'
    apply_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=apply_transform)
    
    test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=apply_transform)
    
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)


    data_indices = [np.where(np.array(train_dataset.targets) == i)[0] for i in range(10)]
    client_indices = [[] for _ in range(num_car)]
    for indices in data_indices:
        # Draw samples from a Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_car))
        
        # Calculate the number of data points for each client
        # proportions = np.array([p * (len(indices) / sum(proportions)) for p in proportions]).astype(int)
        proportions = (proportions * len(indices)).astype(int)

        # Ensure total sum of indices is equal to the original total
        proportions[-1] = len(indices) - sum(proportions[:-1])
        
        # Assign indices to each client based on the calculated proportions
        np.random.shuffle(indices)
        start = 0
        for client, num_indices in zip(client_indices, proportions):
            end = start + num_indices
            client.extend(indices[start:end])
            start = end

    # Check and redistribute if any client has zero total samples
    min_samples_per_client = 1  # Minimum samples any client should have
    for i, client in enumerate(client_indices):
        if len(client) < min_samples_per_client:
            # Find a client to take samples from
            donor = max(enumerate(client_indices), key=lambda x: len(x[1]))[0]
            client_indices[i].extend(client_indices[donor][:min_samples_per_client])
            client_indices[donor] = client_indices[donor][min_samples_per_client:]
    # Create client datasets
    client_datasets = [torch.utils.data.Subset(train_dataset, indices) for indices in client_indices]

    train_loader = [torch.utils.data.DataLoader(traindataset, batch_size, shuffle=True) for traindataset in client_datasets]
    # Stratified sampling for the test set
    y_test = np.array([test_dataset[i][1] for i in test_idx])
    _, test_sampled_idx = train_test_split(test_idx, test_size=test_ratio, stratify=y_test, random_state=random_seed )
    sub_test_loader = torch.utils.data.DataLoader(DatasetSplit(test_dataset, test_sampled_idx), batch_size=batch_size, shuffle=False)
    full_test_loader = torch.utils.data.DataLoader(DatasetSplit(test_dataset,test_idx),batch_size=batch_size, shuffle = False)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader
# def get_cifar10_dataset(shards_per_user, num_car,augment,batch_size):
    #Cifar10
    data_dir = '../data/'
    # define transforms
    # normalize = transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225],
    #     )
    # valid_transform = transforms.Compose([
    #         # transforms.Resize((227,227)),
    #         transforms.ToTensor(),
    #         normalize,
    # ])
    # if augment:
    #     train_transform = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    # else:
    #     train_transform = transforms.Compose([
    #         transforms.Resize((227,227)),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    
    dataset = datasets.CIFAR10(root=data_dir, download = True)
    
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=transform_train)
    
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                  transform=transform_test)
    

    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)
    
    # balanced non_iid data
    num_shards, num_imgs = shards_per_user*num_car, int(50000/(shards_per_user*num_car))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_car)}
    idxs = np.arange(num_shards*num_imgs)
    labels = train_dataset.targets#.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign  shards/client
    for i in range(num_car):
        # shards_per_user = int(num_shards/num_car)
        print(idx_shard)
        rand_set = set(np.random.choice(idx_shard, shards_per_user, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    dict_tasks = dict_users    
    
    idxs = [int(i) for i in dict_tasks[0]]
    # sample_idx = random.sample(range(len(train_dataset)), 200)
    # sample_set = []
    # for idx in sample_idx:
    #     batch_set = []
    #     for i in range(10):
    #         image, label = train_dataset[idx]
    #         # batch_set.append(torch.tensor(image))
    #         batch_set.append(image.clone().detach().requires_grad_(True))
    #     sample_set.append(torch.stack(batch_set))
        
    train_loader, test_loader = get_permute_dataset(train_dataset,test_dataset,test_idx, dict_tasks, batch_size)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, test_loader, full_loader



def get_cifar10_iid(num_car,batch_size,test_ratio):
    #Cifar10
    data_dir = '../data/'
    # define transforms
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    
    
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=transform_train)
    
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                  transform=transform_test)
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)
    
    #iid distribution
    num_items = int(len(train_dataset)/num_car)
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    #all_idxs = np.array(all_idxs)
    for i in range(num_car):
        dict_tasks[i] = set(np.random.choice(all_idxs, num_items,
                                                 replace=False))
        all_idxs = list(set(all_idxs) - dict_tasks[i])
    all_idxs = [i for i in range(len(train_dataset))]
    
    train_loader, sub_test_loader,full_test_loader = get_permute_dataset(train_dataset,test_dataset,test_idx, dict_tasks, batch_size,test_ratio)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader


def get_cifar10_non_iid(shards_allocation_list, num_car,batch_size,test_ratio):
    #Cifar10
    data_dir = '../data/'
    # define transforms
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    
    
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=transform_train)
    
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                  transform=transform_test)
    
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)
    
    # balanced non_iid data
    num_shards, num_imgs = sum(shards_allocation_list), int(50000/(sum(shards_allocation_list)))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_car)}
    idxs = np.arange(num_shards*num_imgs)
    labels = train_dataset.targets
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign  shards/client
    for i in range(num_car):
        # shards_per_user = int(num_shards/num_car)
        print(idx_shard)
        rand_set = set(np.random.choice(idx_shard, shards_allocation_list[i], replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    dict_tasks = dict_users    
    
    idxs = [int(i) for i in dict_tasks[0]]
        
    train_loader, sub_test_loader,full_test_loader = get_permute_dataset(train_dataset,test_dataset,test_idx, dict_tasks, batch_size,test_ratio)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader

def get_cifar10_dirichlet(alpha, num_car,batch_size,test_ratio):
    # Lower alpha increases heterogeneity
    #Cifar10
    data_dir = '../data/'
    # define transforms
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    
    
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                   transform=transform_train)
    
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                  transform=transform_test)
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)


    data_indices = [np.where(np.array(train_dataset.targets) == i)[0] for i in range(10)]
    client_indices = [[] for _ in range(num_car)]
    for indices in data_indices:
        # Draw samples from a Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_car))
        
        # Calculate the number of data points for each client
        # proportions = np.array([p * (len(indices) / sum(proportions)) for p in proportions]).astype(int)
        proportions = (proportions * len(indices)).astype(int)

        # Ensure total sum of indices is equal to the original total
        proportions[-1] = len(indices) - sum(proportions[:-1])
        
        # Assign indices to each client based on the calculated proportions
        np.random.shuffle(indices)
        start = 0
        for client, num_indices in zip(client_indices, proportions):
            end = start + num_indices
            client.extend(indices[start:end])
            start = end

    # Check and redistribute if any client has zero total samples
    min_samples_per_client = 1  # Minimum samples any client should have
    for i, client in enumerate(client_indices):
        if len(client) < min_samples_per_client:
            # Find a client to take samples from
            donor = max(enumerate(client_indices), key=lambda x: len(x[1]))[0]
            client_indices[i].extend(client_indices[donor][:min_samples_per_client])
            client_indices[donor] = client_indices[donor][min_samples_per_client:]
    # Create client datasets
    client_datasets = [torch.utils.data.Subset(train_dataset, indices) for indices in client_indices]

    train_loader = [torch.utils.data.DataLoader(traindataset, batch_size, shuffle=True) for traindataset in client_datasets]
    # Stratified sampling for the test set
    y_test = np.array([test_dataset[i][1] for i in test_idx])
    _, test_sampled_idx = train_test_split(test_idx, test_size=test_ratio, stratify=y_test, random_state=random_seed )
    sub_test_loader = torch.utils.data.DataLoader(DatasetSplit(test_dataset, test_sampled_idx), batch_size=batch_size, shuffle=False)
    full_test_loader = torch.utils.data.DataLoader(DatasetSplit(test_dataset,test_idx),batch_size=batch_size, shuffle = False)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader

def get_cifar100_iid(num_car,batch_size,test_ratio):
    #Cifar100
    data_dir = '../data/'
    # define transforms
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(value='random'),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    
    dataset = datasets.CIFAR100(root=data_dir, download = True)
    
    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                   transform=transform_train)
    
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                  transform=transform_test)
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)
    
    #iid distribution
    num_items = int(len(train_dataset)/num_car)
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    #all_idxs = np.array(all_idxs)
    for i in range(num_car):
        dict_tasks[i] = set(np.random.choice(all_idxs, num_items,
                                                 replace=False))
        all_idxs = list(set(all_idxs) - dict_tasks[i])
    all_idxs = [i for i in range(len(train_dataset))]
    
    train_loader, sub_test_loader,full_test_loader = get_permute_dataset(train_dataset,test_dataset,test_idx, dict_tasks, batch_size,test_ratio)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader


def get_cifar100_non_iid(shards_allocation_list, num_car,batch_size,test_ratio):
    #Cifar100
    data_dir = '../data/'
    # define transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(value='random'),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    
    dataset = datasets.CIFAR100(root=data_dir, download = True)
    
    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                   transform=transform_train)
    
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                  transform=transform_test)
    
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)
    
    # balanced non_iid data
    num_shards, num_imgs = sum(shards_allocation_list), int(50000/(sum(shards_allocation_list)))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_car)}
    idxs = np.arange(num_shards*num_imgs)
    labels = train_dataset.targets
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign  shards/client
    for i in range(num_car):
        # shards_per_user = int(num_shards/num_car)
        print(idx_shard)
        rand_set = set(np.random.choice(idx_shard, shards_allocation_list[i], replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    dict_tasks = dict_users    
    
    idxs = [int(i) for i in dict_tasks[0]]
        
    train_loader, sub_test_loader,full_test_loader = get_permute_dataset(train_dataset,test_dataset,test_idx, dict_tasks, batch_size,test_ratio)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader

def get_cifar100_dirichlet(alpha, num_car,batch_size,test_ratio):
    # Lower alpha increases heterogeneity
    #Cifar100
    data_dir = '../data/'
    # define transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomErasing(value='random'),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    
    dataset = datasets.CIFAR100(root=data_dir, download = True)
    
    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                   transform=transform_train)
    
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                  transform=transform_test)
    dict_tasks, all_idxs = {}, [i for i in range(len(train_dataset))]
    test_idx = [i for i in range(len(test_dataset))]
    all_idxs = np.array(all_idxs)


    data_indices = [np.where(np.array(dataset.targets) == i)[0] for i in range(10)]
    client_indices = [[] for _ in range(num_car)]
    for indices in data_indices:
        # Draw samples from a Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_car))
        
        # Calculate the number of data points for each client
        # proportions = np.array([p * (len(indices) / sum(proportions)) for p in proportions]).astype(int)
        proportions = (proportions * len(indices)).astype(int)

        # Ensure total sum of indices is equal to the original total
        proportions[-1] = len(indices) - sum(proportions[:-1])
        
        # Assign indices to each client based on the calculated proportions
        np.random.shuffle(indices)
        start = 0
        for client, num_indices in zip(client_indices, proportions):
            end = start + num_indices
            client.extend(indices[start:end])
            start = end

    # Check and redistribute if any client has zero total samples
    min_samples_per_client = 1  # Minimum samples any client should have
    for i, client in enumerate(client_indices):
        if len(client) < min_samples_per_client:
            # Find a client to take samples from
            donor = max(enumerate(client_indices), key=lambda x: len(x[1]))[0]
            client_indices[i].extend(client_indices[donor][:min_samples_per_client])
            client_indices[donor] = client_indices[donor][min_samples_per_client:]
    # Create client datasets
    client_datasets = [torch.utils.data.Subset(train_dataset, indices) for indices in client_indices]

    train_loader = [torch.utils.data.DataLoader(traindataset, batch_size, shuffle=True) for traindataset in client_datasets]
    # Stratified sampling for the test set
    y_test = np.array([test_dataset[i][1] for i in test_idx])
    _, test_sampled_idx = train_test_split(test_idx, test_size=test_ratio, stratify=y_test, random_state=random_seed )
    sub_test_loader = torch.utils.data.DataLoader(DatasetSplit(test_dataset, test_sampled_idx), batch_size=batch_size, shuffle=False)
    full_test_loader = torch.utils.data.DataLoader(DatasetSplit(test_dataset,test_idx),batch_size=batch_size, shuffle = False)
    full_loader = torch.utils.data.DataLoader(DatasetSplit(train_dataset, all_idxs),batch_size=batch_size, shuffle = True)  
    
    return train_loader, sub_test_loader,full_test_loader, full_loader

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return image.clone().detach().requires_grad_(True), torch.tensor(label)
    
    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        sample_set = []
        for idx in sample_idx:
            batch_set = []
            for i in range(10):
                image, label = self.dataset[idx]
                batch_set.append(image.clone().detach().requires_grad_(True))
            sample_set.append(torch.stack(batch_set))
        return sample_set
    
def get_permute_dataset(traindataset,testdataset,test_idx,dict_tasks,batch_size,test_ratio):
    train_loader = {}
    #test_loader = {}
    for i in range(len(dict_tasks)):
        idxs = dict_tasks[i]
        idxs_train = idxs#[:int(0.8*len(idxs))]
        #idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        #idxs_test = idxs[int(0.9*len(idxs)):]
        train_loader[i] = torch.utils.data.DataLoader(DatasetSplit(traindataset, idxs_train),
                                                      batch_size=batch_size, shuffle=True)
    # Stratified sampling for the test set
    y_test = np.array([testdataset[i][1] for i in test_idx])
    _, test_sampled_idx = train_test_split(test_idx, test_size=test_ratio, stratify=y_test, random_state=random_seed )
    sub_test_loader = torch.utils.data.DataLoader(DatasetSplit(testdataset, test_sampled_idx), batch_size=batch_size, shuffle=False)
    full_test_loader = torch.utils.data.DataLoader(DatasetSplit(testdataset,test_idx),
                                                     batch_size=batch_size, shuffle = False)
        #random.shuffle(idx)
    return train_loader, sub_test_loader,full_test_loader