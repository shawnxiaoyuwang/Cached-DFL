import torch
from utils_cnn import normal_train, test, variable, multiple_ewc_train, get_subgradient_push
from torch.nn import functional as F
from tqdm import tqdm
from copy import deepcopy
   

# def average_weights(w,weights):
#     """
#     Returns the average of the weights.
#     """
#     weights = weights/sum(weights)
#     print(weights)
#     w_avg = deepcopy(w[0])
#     with torch.no_grad():
#         for param_name in w_avg.keys():
#             avg_param = torch.zeros_like(w_avg[param_name],dtype=torch.float32)
#             for model_para, weight in zip(w, weights):
#                 param = model_para[param_name].float()
#                 avg_param += weight * param
#             w_avg[param_name].copy_(avg_param.dtype)
#     return w_avg

def average_weights(w,weights):
    """
    Returns the average of the weights.
    """
    weights = weights/sum(weights)
    print(weights)
    w_avg = deepcopy(w[0])
    for key in w_avg.keys():
            w_avg[key] = torch.mul(w[0][key],weights[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += torch.mul(w[i][key],weights[i])
        # w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def sum_weights(w):
    """
    Returns the sum of the weights.
    """
    w_sum = deepcopy(w[0])
    for key in w_sum.keys():
        for i in range(1, len(w)):
            w_sum[key] += w[i][key]
    return w_sum

def mul_weights(w,mul_value):
    """
    Returns the mul_value*weights.
    """
    w_mul = deepcopy(w)
    for key in w_mul.keys():
        w_mul[key] = torch.mul(w[key],mul_value)
    return w_mul

def div_weights(w, div_value):
    """
    Returns the result of the weights/div_value.
    """
    w_div = deepcopy(w)
    for key in w_div.keys():
        w_div[key] = torch.div(w_div[key], div_value)
    return w_div

# def get_diag_fisher_matrix(model,train_loader,index,sample_size):
#     dataset = train_loader[index].dataset.get_sample(sample_size)
#     params = {n: p for n, p in model.named_parameters() if p.requires_grad}
#     precision_matrices = {}
#     for n, p in copy.deepcopy(params).items():
#         p.data.zero_()
#         precision_matrices[n] = variable(p.data)
        
#     model.eval()
#     for input in dataset:
#         model.zero_grad()
# #             print(input)
# #             print(input.shape)
#         input = variable(input)
#         output = model(input)#.view(1, -1)
#         label = output.max(1)[1]#.view(-1)
#         loss = F.nll_loss(F.log_softmax(output, dim=1), label)
#         loss.backward()

#         for n, p in model.named_parameters():
#             precision_matrices[n].data += p.grad.data ** 2 / len(dataset)

#     precision_matrices = {n: p for n, p in precision_matrices.items()}
#     return precision_matrices


    
def ewc_training_process(model,optimizer,train_loader,test_loader,diag_fisher_list, model_list, importance, index, local_ep, loss,acc_global,acc_local,model_dir):
    # acc_global.append(test(model, test_loader))
    # acc_local.append(test(model, train_loader[task]))
    
    for _ in tqdm(range(local_ep),disable=True):
        loss.append(multiple_ewc_train(model, optimizer, train_loader[index], diag_fisher_list, model_list, index, importance))
        # acc_global.append(test(model, test_loader))
        # acc_local.append(test(model, train_loader[task]))
    # torch.save(model.state_dict(), model_dir+'/model_'+str(index)+'.pt')
    #return loss,acc
    
def normal_training_process(model,optimizer,train_loader,local_ep,loss):
    # acc_global.append(test(model, test_loader))
    # acc_local.append(test(model, train_loader[task]))
    for _ in tqdm(range(local_ep),disable=True):
        loss.append(normal_train(model, optimizer, train_loader))
        # acc_global.append(test(model, test_loader))
        # acc_local.append(test(model, train_loader[task]))
    # torch.save(model.state_dict(), model_dir+'/model_'+str(index)+'.pt')
#         print(acc_global[-1])
    #return loss, acc
    
def average_process(model_1,model_2):
    w_1 = model_1.state_dict()
    w_2 = model_2.state_dict()
    w = average_weights([w_1,w_2])
    model_1.load_state_dict(w)
    model_2.load_state_dict(w)



def weighted_average_process(model_1,model_2,weights):
    w_1 = model_1.state_dict()
    w_2 = model_2.state_dict()
    w = average_weights([w_1,w_2],weights)
    model_1.load_state_dict(w)
    model_2.load_state_dict(w)




def exchange_process(model_1,model_2):
    w_1 = deepcopy(model_1.state_dict())
    w_2 = deepcopy(model_2.state_dict())
    model_1.load_state_dict(w_2)
    model_2.load_state_dict(w_1)
    
#     model_inter = copy.deepcopy(model_1)
    #optimizer_inter = copy.deepcopy(optimizer_1)
#     model_1 = copy.deepcopy(model_2)
#     model_2 = copy.deepcopy(model_inter)
    #optimizer_1 = copy.deepcopy(optimizer_2)
    #optimizer_2 = copy.deepcopy(optimizer_inter)

    return model_1,model_2#,optimizer_1,optimizer_2

def nothing_process(model_1,model_2):
    return model_1,model_2  






def subgradient_push_process(model,tilde_w,optimizer,train_loader,car_id,loss,model_dir,lr):
    model_w = deepcopy(model)
    model_w.load_state_dict(tilde_w)
    iteration_loss, x = get_subgradient_push(model,model_w, optimizer, train_loader[car_id],lr)
    loss.append(iteration_loss)
    # print(x['conv1.bias'])
    # torch.save(x, model_dir+'/x_'+str(car_id)+'.pt')



