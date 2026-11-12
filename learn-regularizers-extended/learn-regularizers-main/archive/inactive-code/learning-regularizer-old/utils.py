
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from meta_optimizer import *
from model import *
from torch.autograd import Variable
import gc
import os
from sam import SAM, smooth_regularize
from hessian_eigenthings import compute_hessian_eigenthings
import copy


def train_opt_func(
        truncated_bptt_step,
        num_unroll_train,
        train_period,
        train_data_loader,
        dataset,
        model_name,
        device,
        meta_optimizer
):
    '''
    general training function for a fixed value of
     - num_unroll_train :number of unrolling steps (inner training steps)
     - train_period : maximum epochs of training the optimizer(args.max_epochs)
     - data_loader : for training the optimizee
     - model_train : the optimizee model used for training
     - save_root : where the intermediate/best model while training, goes to

    '''


    # Create a meta optimizer that wraps a model into a meta model
    # to keep track of the meta updates.
    print(f"N_train : {num_unroll_train}, epochs = {train_period}")

    meta_model = initialize_model(dataset, model_name)


    meta_model = meta_model.to(device)


    optimizer = optim.Adam(meta_optimizer.parameters())

    #t_period many epochs with the same value of N_train
    for epoch in range(train_period):

        train_iter = iter(train_data_loader)

        print(f"optimizee initialization no. {epoch}")
        # Sample a new model
        model = initialize_model(dataset, model_name)

        model = model.to(device)

        # number of unrolls
        for k in (range(num_unroll_train // truncated_bptt_step)):

            
            meta_optimizer.reset_lstm(keep_states=k > 0, model=model)

            loss_sum = torch.zeros(1, requires_grad = True)
            loss_sum = loss_sum.to(device)

            # prev_loss = torch.zeros(1, requires_grad = True)
            # prev_loss = prev_loss.to(device)

            #one unrolling step
            for j in range(truncated_bptt_step):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_data_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)

                # First we need to compute the gradients wrt model parameters at a point
                f_x = model(x)
                loss = F.nll_loss(f_x, y)

                model.zero_grad()
                loss.backward()

                #Update the values of the parameters using the meta optimizer, meta_model is the new model params
                meta_model = meta_optimizer.meta_update(model)

                # Compute a loss for a step the meta optimizer
                #need to find it again since cannot backward throught the same graph twice using the same loss
                # f_x = meta_model(x)
                # loss = F.nll_loss(f_x, y)
                # print(loss.item())
                # loss_sum += (j * truncated_bptt_step) * (loss - Variable(prev_loss))
                # loss_sum += loss
                # print(loss_sum.item(), loss.item())
                # prev_loss.data = loss.data

                # t = torch.cuda.get_device_properties(0).total_memory
                # r = torch.cuda.memory_reserved(0)
                # a = torch.cuda.memory_allocated(0)
                # print(f"epoch {epoch} iter {k}, {j} : {t}, {r}, {a}, {r-a}")
            #finding the loss for optimizer using the last model params
                if j == truncated_bptt_step-1:
                        print(f"epoch>> {epoch}, unrolling>>> {k}, loss on last iteration : {loss.item()} ", flush= True)
            
            for j in range(len(train_data_loader)):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_data_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                f_x = meta_model(x)
                loss = F.nll_loss(f_x, y)
                loss_sum += loss

            loss_sum= loss_sum/len(train_data_loader)
            # Update the parameters of the meta optimizer
            #truncated BPTT
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()
            # for param in meta_optimizer.parameters():
            #     # print(param.grad)
            #     param.grad.data.clamp_(-1, 1)
            #     # print('yes')
            
        
            
    # del meta_model, model, x, y, prev_loss, loss, loss_sum
    gc.collect()


def reg_train_opt_func(
        truncated_bptt_step,
        num_unroll_train,
        train_period,
        train_data_loader,
        dataset,
        model_name,
        device,
        meta_optimizer,
        weight_decay,
        adaptive,
        if_smooth,
        smooth_reg_object
):
    '''
    general training function for a fixed value of
     - num_unroll_train :number of unrolling steps (inner training steps)
     - train_period : maximum epochs of training the optimizer(args.max_epochs)
     - data_loader : for training the optimizee
     - model_train : the optimizee model used for training
     - save_root : where the intermediate/best model while training, goes to

    '''


    # Create a meta optimizer that wraps a model into a meta model
    # to keep track of the meta updates.
    print(f"N_train : {num_unroll_train}, epochs = {train_period}, ")
 
    


    base_optimizer = optim.Adam
    optimizer = SAM(meta_optimizer.parameters(), base_optimizer, weight_decay, adaptive=adaptive, lr=0.01)


    #t_period many epochs with the same value of N_train
    for epoch in range(train_period):

        train_iter = iter(train_data_loader)

        print(f"optimizee initialization no. {epoch}")

        # Sample a new model
        model = initialize_model(dataset, model_name)
        model = model.to(device)
        model2 = initialize_model(dataset, model_name)
        model2 = model2.to(device)

        # number of unrolls
        for k in (range(num_unroll_train // truncated_bptt_step)):

            meta_optimizer.reset_lstm(keep_states=k > 0, model=model)   #resets the initialized model parameters to that of the model 

            #copying the paramters of the model to meta_model (required in SAM)
            for modelA, modelB in zip(model2.parameters(), model.parameters()):
                modelA.data.copy_(modelB.data)

            smooth_loss = torch.zeros(1, requires_grad = True)
            smooth_loss = smooth_loss.to(device)

            #one unrolling step
            for j in range(truncated_bptt_step):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_data_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)

                # First we need to compute the gradients wrt model parameters at a point
                f_x = model(x)
                loss = F.nll_loss(f_x, y)
                model.zero_grad()
                loss.backward() #meta_optimizer params dont change

                #Update the values of the parameters using the meta optimizer, meta_model is the new model params
                model_temp = meta_optimizer.meta_update(model)  #model_temp points to the model used in initializing Meta_optimizer 
                                                                #which will only update the meta_optimizer
                
                ################################
                #smoothness loss

                if if_smooth:
                    smooth_loss += smooth_reg_object.run(model)

                ################################


                if j == truncated_bptt_step-1:
                    print(f"epoch>> {epoch}, unrolling>>> {k}, loss on last iteration : {loss.item()} ", flush= True)
            
            
            #find the meta loss of the meta optimizer
            loss_sum = torch.zeros(1, requires_grad = True)
            loss_sum = loss_sum.to(device)
            
            for j in range(len(train_data_loader)):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_data_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                f_x = model_temp(x)
                loss = F.nll_loss(f_x, y)
                loss_sum += loss
            loss_sum = loss_sum/len(train_data_loader)
            
            loss_sum = loss_sum + smooth_reg_object.eps * smooth_loss
            # # Update the parameters of the meta optimizer
            # #truncated BPTT
            meta_optimizer.zero_grad()
            loss_sum.backward() #model_temp grad dont change

            #the previous step gives us the gradient on the optimizer model to be used for finding e^(w)


#################################################################################################################
            #using the method of Sharpness Aware Minim. to obtain better params for L. Optimizer
            # meta_optimizer.reset_lstm(keep_states=k > 0, model=model)
            
            #1.
            ##find e^(w) and change the value of the paramerter of meta_optimizer by w + e^(w)
            optimizer.first_step(zero_grad=True)


            #2.
            #find the loss due to the w+e^(w)
            
            meta_optimizer.reset_lstm(keep_states=k > 0, model=model2)  #important step to reset initialized optimizee inside Meta_optimizer
            
            
            smooth_loss2 = torch.zeros(1, requires_grad = True)
            smooth_loss2 = smooth_loss2.to(device)

            #finding the optimizee parameter at the last iteration for BPTT
            for j in range(truncated_bptt_step):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_data_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                f_x = model2(x)                 # or, model(x)
                loss = F.nll_loss(f_x, y)
                model2.zero_grad()
                loss.backward()
                model2_temp = meta_optimizer.meta_update(model2)

                #####################
                #smoothness loss
                if if_smooth:
                    smooth_loss2 += smooth_reg_object.run(model2)
                #####################
                    

            loss_sum2 = torch.zeros(1, requires_grad = True)
            loss_sum2 = loss_sum2.to(device)

        
            #finding the optimizer's loss using the optimizee's state
            for j in range(len(train_data_loader)):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_data_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                f_x = model2_temp(x)
                loss = F.nll_loss(f_x, y)
                loss_sum2 += loss
            loss_sum2 = loss_sum2/len(train_data_loader)
            loss_sum2 = loss_sum2 + smooth_reg_object.eps * smooth_loss2
            loss_sum2.backward()

            
            optimizer.second_step(zero_grad=False)

#################################################################################################################            
            
            # for param in meta_optimizer.parameters():
            #     param.grad.data.clamp_(-1, 1)
    gc.collect()



def eval_opt_func(
        num_unroll_eval,
        train_data_loader,
        dataset,
        model_name,
        device,
        meta_optimizer,
        eval_epochs):

    eval_loss = 0
    print("Evaluating optimizer:\n")
    for ep in range(eval_epochs):
        model = initialize_model(dataset, model_name)
        train_iter = iter(train_data_loader)
        model = model.to(device)

        loss_sum = 0
        for j in range(num_unroll_eval):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_data_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            # First we need to compute the gradients wrt model parameters at a point
            f_x = model(x)
            loss = F.nll_loss(f_x, y)
            model.zero_grad()
            loss.backward()
            loss_sum += loss.item()
            if j%20 == 0:
                print(f"loss: {loss.item()}\n")
            # print(loss.item()-prev_loss.item(), loss.item())

            with torch.no_grad():
                meta_optimizer.meta_update(model)
        del model, loss
        gc.collect()

        eval_loss += loss_sum/num_unroll_eval
        print(f"Evaluation Epoch {ep}| loss : {loss_sum/num_unroll_eval}")
        
    return eval_loss/eval_epochs







#############################################################################################

#  Hessian Eigen Value Computation

def compute_hessian_spectrum(num_eigenvals, model, train_data_loader):
    num_eigenthings = num_eigenvals  # compute top 100 eigenvalues/eigenvectors
    loss_e = nn.NLLLoss()
    eigenvals, eigenvecs = compute_hessian_eigenthings(model, train_data_loader,loss_e, num_eigenthings)
    return eigenvals.tolist(),eigenvecs.tolist()




#############################################################################################

# accuracy

def accuracy(outputs, labels):

    _, predicted = torch.max(outputs, 1)
    # print(outputs.size(), predicted.size())
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    return acc




#############################################################################################



def compute_val_loss(test_data_loader, model):
    v_loss = 0
    test_iter = iter(test_data_loader)
    with torch.no_grad():
        for e in range(len(test_data_loader)):
            try:
                x, y = next(test_iter)
                x, y = x.to(device), y.to(device)
                f_x = model(x)
                v_loss += F.nll_loss(f_x, y)
                if e == 0:
                    pred_y = f_x
                    act_y = y
                else:
                    torch.cat([pred_y, f_x], dim = 0)
                    torch.cat([act_y, y], dim = 0)

            except StopIteration:
                test_iter = iter(test_data_loader)

    return v_loss/len(test_data_loader)
    




#############################################################################################
#get flat parameters

def get_flat_params(model):
    params = []
    for name, module in model.named_modules():
        if len(module._parameters) != 0:
            params.append(module._parameters['weight'].view(-1))
            try:
                params.append(module._parameters['bias'].view(-1))
            except:
                pass
    return torch.cat(params)




#set flat params

def set_flat_params(model, flat_params):
    # Restore original shapes
    offset = 0
    for module in model.modules():
        if len(module._parameters) != 0:
            weight_shape = module._parameters['weight'].size()
            weight_flat_size = reduce(mul, weight_shape, 1)
            module._parameters['weight'] = flat_params[
                                            offset:offset + weight_flat_size].view(*weight_shape)
            try:
                bias_shape = module._parameters['bias'].size()
                bias_flat_size = reduce(mul, bias_shape, 1)
                module._parameters['bias'] = flat_params[
                                                offset + weight_flat_size:offset + weight_flat_size + bias_flat_size].view(
                    *bias_shape)
            except:
                bias_flat_size = 0
            offset += weight_flat_size + bias_flat_size
    return model




#############################################################################################

# Max loss in the neighbourhood
# using projected gradient ascent

 
def max_loss_PGD(model, eps, test_data_loader, device):

    #deep copy of the model, will be used for perturbation
    model_p = copy.deepcopy(model)
    step_size = eps/10
    #perturbation
    with torch.no_grad():
        for params in model_p.parameters():
            params.add_(torch.randn(params.size()).to(device))


    # for param_orig, param_copy in zip(model.parameters(), model_p.parameters()):
    #     assert torch.ne(param_orig.data, param_copy.data), "Parameters are not equal"

    actual_loss = compute_val_loss(test_data_loader, model)
    print(f"The actual test loss here is : {actual_loss}")

    flat_params = get_flat_params(model_p)
    perturbed_loss = []
    for _ in range(10):

        flat_params.requires_grad_()
        with torch.enable_grad():
            v_loss = 0
            test_iter = iter(test_data_loader)
            model_p = set_flat_params(model_p, flat_params)
            for e in range(len(test_data_loader)):
                try:
                    x, y = next(test_iter)
                    x, y = x.to(device), y.to(device)
                    f_x = model_p(x)
                    v_loss += F.nll_loss(f_x, y)
                    if e == 0:
                        pred_y = f_x
                        act_y = y
                    else:
                        torch.cat([pred_y, f_x], dim = 0)
                        torch.cat([act_y, y], dim = 0)

                except StopIteration:
                    test_iter = iter(test_data_loader)
            
            p_loss = v_loss/len(test_data_loader)
            perturbed_loss.append(p_loss.item())

            print(f"perturbed loss {_} : {perturbed_loss}")

            diff = p_loss - actual_loss

            grads = torch.autograd.grad(diff, [flat_params],allow_unused=True)[0]

            # for param_p, param, grad in zip(model_p.parameters(), model.parameters(), grads):
            flat_params = flat_params.detach() + step_size * torch.sign(grads.detach())
            flat_params = torch.min(torch.max(flat_params, flat_params - eps), flat_params + eps)

    perturbed_params = flat_params.detach()
    model_p = set_flat_params(model_p, perturbed_params)

    perturbed_loss.append(compute_val_loss(test_data_loader, model_p).detach().cpu().item())

    return perturbed_loss, actual_loss.item()



# def smooth_regularize():
