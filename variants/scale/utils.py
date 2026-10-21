
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from meta_optimizer import *
from model import *
from torch.autograd import Variable
import gc
import os
from sam import SAM



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
    print(f"N_train : {num_unroll_train}, epochs = {train_period}, ")

    meta_model = initialize_model(dataset, model_name)


    meta_model = meta_model.to(device)


    optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-4)

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

            prev_loss = torch.zeros(1, requires_grad = True)
            prev_loss = prev_loss.to(device)

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
                f_x = meta_model(x)
                loss = F.nll_loss(f_x, y)
                # print(loss.item())
                loss_sum += (j * truncated_bptt_step) * (loss - Variable(prev_loss))

                # print(loss_sum.item(), loss.item())
                prev_loss.data = loss.data

                # t = torch.cuda.get_device_properties(0).total_memory
                # r = torch.cuda.memory_reserved(0)
                # a = torch.cuda.memory_allocated(0)
                # print(f"epoch {epoch} iter {k}, {j} : {t}, {r}, {a}, {r-a}")
                
                if j == truncated_bptt_step-1:
                    print(f"epoch>> {epoch}, unrolling>>> {k}, loss on last iteration : {loss.item()} ", flush= True)
            
            
            # Update the parameters of the meta optimizer
            #truncated BPTT
            meta_optimizer.zero_grad()
            loss_sum.backward()
            for param in meta_optimizer.parameters():
                # print(param.grad)
                param.grad.data.clamp_(-1, 1)
                # print('yes')
            optimizer.step()
        
            
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
    print(f"N_train : {num_unroll_train}, epochs = {train_period}, ")

    meta_model = initialize_model(dataset, model_name)


    meta_model = meta_model.to(device)


    base_optimizer = optim.Adam
    optimizer = SAM(meta_optimizer.parameters(), base_optimizer, lr=1e-4)

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

            prev_loss = torch.zeros(1, requires_grad = True)
            prev_loss = prev_loss.to(device)

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
                f_x = meta_model(x)
                loss = F.nll_loss(f_x, y)
                loss_sum += (j * truncated_bptt_step) * (loss - Variable(prev_loss))

                prev_loss.data = loss.data

                if j == truncated_bptt_step-1:
                    print(f"epoch>> {epoch}, unrolling>>> {k}, loss on last iteration : {loss.item()} ", flush= True)
            
            
            # Update the parameters of the meta optimizer
            #truncated BPTT
            meta_optimizer.zero_grad()
            loss_sum.backward()


#################################################################################################################
            #using the method of Sharpness Aware Minim. to obtain better params for L. Optimizer
            meta_optimizer.reset_lstm(keep_states=k > 0, model=model)
              
            optimizer.first_step(zero_grad=True)
            
            #calculating second forward-backward pass

            # print("first step\n")
            model_temp = initialize_model(dataset, model_name)
            model_temp = model_temp.to(device)
            for modelA, modelB in zip(model_temp.parameters(), model.parameters()):
                modelA.data.copy_(modelB.data)

            for x, y in zip(model_temp.parameters(), model.parameters()):
                assert torch.equal(x.data,y.data)==True

            
            loss_sum = torch.zeros(1, requires_grad = True).to(device)
            prev_loss = torch.zeros(1, requires_grad = True).to(device)
            for j in range(truncated_bptt_step):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_data_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                f_x = model_temp(x)
                loss = F.nll_loss(f_x, y)
                model_temp.zero_grad()
                loss.backward()
                # with torch.no_grad():
                meta_model = meta_optimizer.meta_update(model_temp)

                f_x = meta_model(x)
                loss = F.nll_loss(f_x, y)
                # print(loss.item())
                loss_sum += (j * truncated_bptt_step) * (loss - Variable(prev_loss))

                prev_loss.data = loss.data
            # print('end first_step')
            # meta_optimizer.zero_grad()
            loss_sum.backward()

            optimizer.second_step(zero_grad=False)

#################################################################################################################            
            #avoiding the following steps since second_step makes the gradient_zero
            for param in meta_optimizer.parameters():
                param.grad.data.clamp_(-1, 1)
    gc.collect()






def eval_opt_func(
        num_unroll_eval,
        train_data_loader,
        dataset,
        model_name,
        device,
        meta_optimizer,
        eval_epochs
):

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
            
            # print(loss.item()-prev_loss.item(), loss.item())

            with torch.no_grad():
                meta_optimizer.meta_update(model)

            torch.cuda.empty_cache()

        del model, loss
        gc.collect()

        eval_loss += loss_sum
        print(f"Evaluation Epoch {ep}| loss : {loss_sum/num_unroll_eval}")
        
    return eval_loss/eval_epochs
