from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
import numpy as np

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, dataset: list):

        self.model = model
        self.dataset = dataset
        #print('estimate fisher matrix!')
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()
        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input in self.dataset:
            self.model.zero_grad()
#             print(input)
#             print(input.shape)
            input = variable(input)
            output = self.model(input)#.view(1, -1)
            label = output.max(1)[1]#.view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss


def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader):
    model.train()
    epoch_loss = 0
    for i,(input, target) in enumerate(data_loader):
        input, target = variable(input), variable(target)
        optimizer.zero_grad()
        #print(target)
        #target = F.one_hot(target,num_classes = 10).to(torch.float)
#         print(input.shape)
        output = model(input)
        #print(output.shape)
        loss = F.cross_entropy(output, target)
        #print(loss)
        #print(loss.item())
        epoch_loss += loss.item()
        #epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float):
    model.train()
    epoch_loss = 0
    for i,(input, target) in enumerate(data_loader):
        input, target = variable(input), variable(target)
        #target = F.one_hot(target,num_classes = 10).to(torch.float)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target) + importance * ewc.penalty(model)
        epoch_loss += loss.item()
        #epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)

def multiple_ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              diag_fisher_list, model_list, index, importance: float):
    model.train()
    epoch_loss = 0
    mean_list = []
    
    for i in range(len(model_list)):
        means = {}
        if type(model_list[i]) == list:
            mean_list.append(means)
        else:
            params = {n: p for n, p in model_list[i].named_parameters() if p.requires_grad}
            for n, p in deepcopy(params).items():
                means[n] = variable(p.data)
            mean_list.append(means)
    # print('model_list',len(model_list))
    # print('mean list',len(mean_list))
    # print('fisher list',len(diag_fisher_list))
    
    for i,(input, target) in enumerate(data_loader):
        input, target = variable(input), variable(target)
        #target = F.one_hot(target,num_classes = 10).to(torch.float)
        optimizer.zero_grad()
        output = model(input)
        loss = F.cross_entropy(output, target)
        for j in range(len(diag_fisher_list)):
            if type(diag_fisher_list[j]) == list:
                continue;
            if j == index:
                continue;
            fisher_loss = 0
            for n, p in model.named_parameters():
                _loss = diag_fisher_list[j][n] * (p - mean_list[j][n]) ** 2
                fisher_loss += _loss.sum()
                loss += importance*fisher_loss
        epoch_loss += loss.item()
        #epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)

def test(model: nn.Module, data_loader: torch.utils.data.DataLoader,num_classes = 10):
    model.eval()
    # correct = 0
    class_correct = np.zeros([num_classes])
    class_count = np.zeros([num_classes])
    for i,(input, target) in enumerate(data_loader):
        input, target = variable(input), variable(target)
        #target = F.one_hot(target,num_classes = 10).to(torch.float)
        output = model(input)
        correct_list = (F.softmax(output, dim=1).max(dim=1)[1] == target).data
        # correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
        for index in range(len(target)):
            item = target[index]
            class_correct[item] += correct_list[index]
            class_count[item] += 1
    return sum(class_correct) / len(data_loader.dataset), class_correct/class_count

# def test(model: nn.Module, data_loader: torch.utils.data.DataLoader):
#     model.eval()
#     correct = 0
#     # class_correct = np.zeros([10])
#     # class_count = np.zeros([10])
#     for i,(input, target) in enumerate(data_loader):
#         input, target = variable(input), variable(target)
#         #target = F.one_hot(target,num_classes = 10).to(torch.float)
#         output = model(input)
#         # correct_list = (F.softmax(output, dim=1).max(dim=1)[1] == target).data
#         correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
#     return np.zeros([10]), correct / len(data_loader.dataset)


def test_parallel(model: nn.Module, data_loader: torch.utils.data.DataLoader, result_queue = None):
    model.eval()
    correct = 0
    for input, target in data_loader:
        input, target = variable(input), variable(target)
        #target = F.one_hot(target,num_classes = 10).to(torch.float)
        output = model(input)
        correct += (F.softmax(output, dim=1).max(dim=1)[1] == target).data.sum()
    if result_queue is not None:
        result_queue.put(correct / len(data_loader.dataset))

def get_subgradient_push(model1: nn.Module, model2: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, lr):
    model1.train()
    model2.train()  # Set both models to training mode
    epoch_loss = 0
    data_size = len(data_loader)
    # print(data_size)
    for input, target in data_loader:
        optimizer.zero_grad()
        output1 = model1(input)
        loss = F.cross_entropy(output1, target)
        epoch_loss += loss.item()
        loss.backward()
    ## Perform parameter update after processing all batches
        with torch.no_grad():
            for param1, param2 in zip(model1.parameters(), model2.parameters()):
                param2.data -= lr * param1.grad.data
    #     # Transfer gradients from model1 to model2 and update model2
    # with torch.no_grad():
    #     for param1, param2 in zip(model1.parameters(), model2.parameters()):
    #         if param2.grad is None:
    #             param2.grad = torch.zeros_like(param2)
    #         param2.grad += param1.grad.clone()

    x = deepcopy(model2.state_dict())

    return epoch_loss / len(data_loader), x
