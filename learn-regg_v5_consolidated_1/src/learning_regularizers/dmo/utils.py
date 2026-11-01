import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from meta_optimizer import *
from model import *
from torch.autograd import Variable
import gc
import os
from regularizers import *
import copy
from regularized_optimizer_helper import *

try:
    from hessian_eigenthings import compute_hessian_eigenthings
except ImportError:
    compute_hessian_eigenthings = None



def train_opt_func(
        num_unroll_train,
        train_data_loader,
        device,
        meta_optimizer,
        args):
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
    print(f"N_train : {num_unroll_train}, epochs = {args.train_period}")

    meta_model = initialize_model(args.dataset, args.optimizee_model_name)
    meta_model = meta_model.to(device)

    if args.base_optimizer.lower() == 'sgd':
        optimizer = optim.SGD(meta_optimizer.parameters(), lr = args.lr, momentum= args.momentum, 
                                            weight_decay=args.weight_decay)
    elif args.base_optimizer.lower() == 'adam':
        optimizer = optim.Adam(meta_optimizer.parameters(), lr = args.lr, 
                                  weight_decay=args.weight_decay)
    
    elif args.base_optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(meta_optimizer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #t_period many epochs with the same value of N_train
    for epoch in range(args.train_period):

        train_iter = iter(train_data_loader)

        print(f"optimizee initialization no. {epoch}")
        # Sample a new model
        model = initialize_model(args.dataset, args.optimizee_model_name)

        model = model.to(device)

        # number of unrolls
        for k in (range(num_unroll_train // args.truncated_bptt_step)):

            
            meta_optimizer.reset_lstm(keep_states=k > 0, model=model)

            loss_sum = torch.zeros(1, device=device)

            # prev_loss = torch.zeros(1, requires_grad = True)
            # prev_loss = prev_loss.to(device)

            #one unrolling step
            for j in range(args.truncated_bptt_step):
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
               
            #finding the loss for optimizer using the last model params
                if j == args.truncated_bptt_step-1:
                        print(f"epoch>> {epoch}, unrolling>>> {k}, loss on last iteration : {loss.item()} ", flush= True)
            
            # for j in range(len(train_data_loader)):
            #     try:
            #         x, y = next(train_iter)
            #     except StopIteration:
            #         train_iter = iter(train_data_loader)
            #         x, y = next(train_iter)
            #     x, y = x.to(device), y.to(device)
            f_x = meta_model(x)
            loss = F.nll_loss(f_x, y)
            loss_sum = loss_sum + loss

            loss_sum= loss_sum/len(train_data_loader)
            # Update the parameters of the meta optimizer
            #truncated BPTT
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

    # del meta_model, model, x, y, prev_loss, loss, loss_sum
    gc.collect()





def SAMoptimizee_reg_train_opt_func(
        num_unroll_train,
        train_data_loader,
        device,
        meta_optimizer,
        smooth_reg_object,
        args):
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
    print(f"N_train : {num_unroll_train}, epochs = {args.train_period}, ")
    
    # model_temp = meta_optimizer.meta_model.model
    

    if args.base_optimizer.lower() == 'sgd':
        LOs_optimizer = optim.SGD(meta_optimizer.parameters(), lr = args.lr, momentum= args.momentum, 
                                            weight_decay=args.weight_decay)
    elif args.base_optimizer.lower() == 'adam':
        LOs_optimizer = optim.Adam(meta_optimizer.parameters(), lr = args.lr, 
                                  weight_decay=args.weight_decay)
    
    elif args.base_optimizer.lower() == 'adamw':
        LOs_optimizer = optim.AdamW(meta_optimizer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # for params in model_temp.parameters():
    #     print("begining:", params.size(), params)

    #t_period many epochs with the same value of N_train
    for epoch in range(args.train_period):

        train_iter = iter(train_data_loader)

        print(f"optimizee initialization no. {epoch}")

        # Sample a new model
        model = initialize_model(args.dataset, args.optimizee_model_name)
        model = model.to(device)
        model.train()
        
        optimizer = SAM(model.parameters(), args= args)
        # model2 = initialize_model(args.dataset, args.optimizee_model_name)
        # model2 = model2.to(device)

        # number of unrolls
        for k in (range(num_unroll_train // args.truncated_bptt_step)):
            
            meta_optimizer.train()
            meta_optimizer.reset_lstm(keep_states=k > 0, model=model)   #resets the initialized model parameters to that of the model 

            smooth_loss = torch.zeros(1, device=device)
            
            #one unrolling step
            for j in range(args.truncated_bptt_step):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_data_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                

                enable_running_stats(model)
                # First we need to compute the gradients wrt model parameters at a point
                f_x = model(x)
                loss = F.nll_loss(f_x, y)
                model.zero_grad()
                loss.backward() #meta_optimizer params dont change

                #Update the values of the parameters using the meta optimizer, meta_model is the new model params
                model_temp = meta_optimizer.meta_update(model)  #model_temp points to the model used in initializing Meta_optimizer 
                                                                #which will only update the meta_optimizer
                

                if j == args.truncated_bptt_step-1:
                    print(f"epoch>> {epoch}, unrolling>>> {k}, loss on last iteration : {loss.item()} ", flush= True)
            
            #################################################################################################################
            #using the method of Sharpness Aware Minim. to obtain the highest value of loss in the nghb            

            #1.
            ##find e^(w) and change the value of the paramerter of optimizee by w + e^(w)
            

            # --- graph-preserving SAM-on-optimizee (Change.MD item 23) ---
            # The old code called reset_lstm() here, which wiped the BPTT
            # graph BEFORE the perturbed loss: the meta-gradient was exactly
            # zero and the LSTM never trained under sam_opzee. Instead:
            # perturb the caller model with SAM's first_step, transfer the
            # (detached) epsilon onto the graph-connected flat params,
            # evaluate, and restore afterwards.
            last_flat_grads = get_flat_grads(model).detach()   # before first_step nulls .grad
            optimizer.first_step(zero_grad=True)          # model <- theta + e(w)
            flat_theta = meta_optimizer.meta_model.get_flat_params()      # graph tensor
            eps_flat = (get_flat_params(model) - flat_theta.detach()).detach()
            meta_optimizer.meta_model.set_flat_params(flat_theta + eps_flat)
            model_temp = meta_optimizer.meta_model.model


            loss_sum = torch.zeros(1, device=device)
            
            # for j in range(len(train_data_loader)):
            #     try:
            #         x, y = next(train_iter)
            #     except StopIteration:
            #         train_iter = iter(train_data_loader)
            #         x, y = next(train_iter)
            #     x, y = x.to(device), y.to(device)

            f_x = model_temp(x)
            loss = F.nll_loss(f_x, y)
            loss_sum = loss_sum + loss


            # /len(train_data_loader) scaling removed (Change.MD item 26)
            
            
            
            # print('added loss', loss_sum)
            if args.if_smooth:
                smooth_loss = smooth_loss + smooth_reg_object.run(last_flat_grads)
                loss_sum = loss_sum + smooth_reg_object.lamb * smooth_loss

            wd_loss = torch.zeros(1, device=device)
            if args.optzee_wd:
                wd_loss = wd_loss + torch.norm(get_flat_params(model_temp).flatten(), p=2)
                loss_sum = loss_sum + args.optzee_wd * wd_loss
            

            
            # restore the unperturbed graph params before continuing
            meta_optimizer.meta_model.set_flat_params(flat_theta)

            disable_running_stats(model)
            #step to change the parameters of the optimizee back to what it was before purturbation
            optimizer.second_step(zero_grad=False)

            # print("change back")
            # for p in model.parameters():
            #     print("change back model_temp", p)


        
            # # Update the parameters of the meta optimizer
            # #truncated BPTT
            meta_optimizer.zero_grad()
            loss_sum.backward() #model_temp grad dont change
            LOs_optimizer.step()
            #the previous step gives us the gradient on the optimizer model to be used for finding e^(w)




    #################################################################################################################            
            
            # for param in meta_optimizer.parameters():
            #     param.grad.data.clamp_(-1, 1)
    gc.collect()




#########################################################################################################
#####  L2 regularization on the optimizee (NO SAM)                                                   ####
#####  Meta-loss = NLL(optimizee) + args.optzee_wd * ||optimizee_params||_2                          ####
#####  Used as the clean L2 baseline requested by the reviewer: 3-way comparison                     ####
#####  (No Reg vs SAM-on-LO vs L2-on-optimizee).                                                     ####
#########################################################################################################
def L2_reg_train_opt_func(
        num_unroll_train,
        train_data_loader,
        device,
        meta_optimizer,
        smooth_reg_object,
        args):
    '''
    L2-regularized meta-training of the learned optimizer.
    Same structure as `train_opt_func` (no SAM perturbation) but appends
    args.optzee_wd * ||optimizee_params||_2 to the meta-loss before backprop.
    Signature matches the SAM* training functions so train.py can call it
    uniformly inside the `if args.if_avreg:` branch.
    '''
    print(f"[L2] N_train : {num_unroll_train}, epochs = {args.train_period}, optzee_wd = {args.optzee_wd}")

    if args.base_optimizer.lower() == 'sgd':
        LOs_optimizer = optim.SGD(meta_optimizer.parameters(), lr=args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.base_optimizer.lower() == 'adam':
        LOs_optimizer = optim.Adam(meta_optimizer.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    elif args.base_optimizer.lower() == 'adamw':
        LOs_optimizer = optim.AdamW(meta_optimizer.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"Unknown base_optimizer: {args.base_optimizer}")

    for epoch in range(args.train_period):
        train_iter = iter(train_data_loader)
        print(f"[L2] optimizee initialization no. {epoch}")

        # Sample a fresh optimizee for this epoch
        model = initialize_model(args.dataset, args.optimizee_model_name)
        model = model.to(device)
        model.train()

        for k in range(num_unroll_train // args.truncated_bptt_step):

            meta_optimizer.train()
            meta_optimizer.reset_lstm(keep_states=k > 0, model=model)

            # ---------- one truncated-BPTT unroll ----------
            for j in range(args.truncated_bptt_step):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_data_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)

                # Gradient of optimizee NLL wrt optimizee params (used by LO to propose update)
                f_x = model(x)
                loss = F.nll_loss(f_x, y)
                model.zero_grad()
                loss.backward()

                # Apply LO update to obtain the new (functional) optimizee
                model_temp = meta_optimizer.meta_update(model)

                if j == args.truncated_bptt_step - 1:
                    print(f"[L2] epoch>> {epoch}, unrolling>>> {k}, loss on last iteration : {loss.item()}", flush=True)

            # ---------- meta-loss on the post-update optimizee ----------
          # ---------- meta-loss on the post-update optimizee ----------
            loss_sum = torch.zeros(1, requires_grad=True).to(device)

            f_x = model_temp(x)
            loss = F.nll_loss(f_x, y)
            loss_sum = loss_sum + loss

            # ---------- L2 PENALTY ON OPTIMIZEE PARAMS ----------
            # Standard L2 = sum(theta^2). Added BEFORE the /len(loader) scaling
            # so the penalty is scaled consistently with the NLL term.
            if args.optzee_wd:
                wd_loss = (get_flat_params(model_temp) ** 2).sum()
                loss_sum = loss_sum + args.optzee_wd * wd_loss

            loss_sum = loss_sum / len(train_data_loader)

            # Optional smoothness term (re-uses existing infra; off by default in l2.yaml)
            if args.if_smooth:
                smooth_loss = smooth_reg_object.run(model_temp)
                loss_sum = loss_sum + smooth_reg_object.lamb * smooth_loss

            # Backprop through the (functional) unroll into LO params
            meta_optimizer.zero_grad()
            loss_sum.backward()
            LOs_optimizer.step()

    gc.collect()





def SAMoptimizer_reg_train_opt_func(
        num_unroll_train,
        train_data_loader,
        device,
        meta_optimizer,
        smooth_reg_object,
        args):
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
    print(f"N_train : {num_unroll_train}, epochs = {args.train_period}, ")

    optimizer = SAM(meta_optimizer.parameters(), args=args)


    #t_period many epochs with the same value of N_train
    for epoch in range(args.train_period):

        train_iter = iter(train_data_loader)

        print(f"optimizee initialization no. {epoch}")

        # Sample a new model
        model = initialize_model(args.dataset, args.optimizee_model_name)
        model = model.to(device)
        model2 = initialize_model(args.dataset, args.optimizee_model_name)
        model2 = model2.to(device)

        # number of unrolls
        for k in (range(num_unroll_train // args.truncated_bptt_step)):

            meta_optimizer.reset_lstm(keep_states=k > 0, model=model)   #resets the initialized model parameters to that of the model
            window_state = meta_optimizer.snapshot_state()   # window-start LSTM state
            window_batches = []                               # same-batch replay cache 

            #copying the paramters of the model to meta_model (required in SAM)
            for modelA, modelB in zip(model2.parameters(), model.parameters()):
                modelA.data.copy_(modelB.data)

            smooth_loss = torch.zeros(1, device=device)
            


            #one unrolling step
            for j in range(args.truncated_bptt_step):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_data_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                window_batches.append((x, y))

                # First we need to compute the gradients wrt model parameters at a point
                f_x = model(x)
                loss = F.nll_loss(f_x, y)
                model.zero_grad()
                loss.backward() #meta_optimizer params dont change

                #Update the values of the parameters using the meta optimizer, meta_model is the new model params
                model_temp = meta_optimizer.meta_update(model)  #model_temp points to the model used in initializing Meta_optimizer 
                                                                #which will only update the meta_optimizer
                


                if j == args.truncated_bptt_step-1:
                    print(f"epoch>> {epoch}, unrolling>>> {k}, loss on last iteration : {loss.item()} ", flush= True)
            
            
            #find the meta loss of the meta optimizer
            loss_sum = torch.zeros(1, device=device)
            
            # for j in range(len(train_data_loader)):
            #     try:
            #         x, y = next(train_iter)
            #     except StopIteration:
            #         train_iter = iter(train_data_loader)
            #         x, y = next(train_iter)
            #     x, y = x.to(device), y.to(device)
            f_x = model_temp(x)
            loss = F.nll_loss(f_x, y)
            loss_sum = loss_sum + loss


            # /len(train_data_loader) scaling removed (Change.MD item 26)
            
            if args.if_smooth:
                smooth_loss = smooth_loss + smooth_reg_object.run(get_flat_grads(model))
                loss_sum = loss_sum + smooth_reg_object.lamb * smooth_loss
            
            
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
            
            # pass 2 starts from the SAME window-start state as pass 1
            # (Change.MD item 30b)
            meta_optimizer.reset_lstm(keep_states=False, model=model2)
            meta_optimizer.restore_state(window_state)
            state_after_pass1 = None
            
            
            smooth_loss2 = torch.zeros(1, device=device)

            #finding the optimizee parameter at the last iteration for BPTT
            # SAME batches as pass 1 (Change.MD item 30a)
            for j in range(args.truncated_bptt_step):
                x, y = window_batches[j]
                f_x = model2(x)                 # or, model(x)
                loss = F.nll_loss(f_x, y)
                model2.zero_grad()
                loss.backward()
                model2_temp = meta_optimizer.meta_update(model2)

            #####################
            #smoothness loss
            if args.if_smooth:
                smooth_loss2 = smooth_loss2 + smooth_reg_object.run(get_flat_grads(model2))
            #####################

            wd_loss = torch.zeros(1, device=device)
            if args.optzee_wd:
                wd_loss = wd_loss + torch.norm(get_flat_params(model2_temp).flatten(), p=2)

            loss_sum2 = torch.zeros(1, device=device)

        
            #finding the optimizer's loss using the optimizee's state
            # for j in range(len(train_data_loader)):
            #     try:
            #         x, y = next(train_iter)
            #     except StopIteration:
            #         train_iter = iter(train_data_loader)
            #         x, y = next(train_iter)
            #     x, y = x.to(device), y.to(device)
            f_x = model2_temp(x)
            loss = F.nll_loss(f_x, y)
            loss_sum2 = loss_sum2 + loss
            # /len(train_data_loader) scaling removed (Change.MD item 26)


            if args.if_smooth:
                loss_sum2 = loss_sum2 + smooth_reg_object.lamb * smooth_loss2
            if args.optzee_wd:
                loss_sum2 = loss_sum2 + args.optzee_wd * wd_loss

            loss_sum2.backward()

            
            optimizer.second_step(zero_grad=False)

#################################################################################################################            
            
            # for param in meta_optimizer.parameters():
            #     param.grad.data.clamp_(-1, 1)
    gc.collect()





def GAM_reg_train_opt_func(
        num_unroll_train,
        train_data_loader,
        device,
        meta_optimizer,
        smooth_reg_object,
        args):
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
    print(f"N_train : {num_unroll_train}, epochs = {args.train_period}, ")


    



    # dict_ = {'model_optimizee' : model2, 'data_loder' : train_data_loader}
    gam_optimizer = get_optim_and_schedulers(meta_optimizer, args = args) #, optzee_model = model2, data_loader = train_data_loader


    #t_period many epochs with the same value of N_train
    for epoch in range(args.train_period):

        train_iter = iter(train_data_loader)
        gam_optimizer.update_rho_t()   # per-epoch rho schedule (item 32)
        print(f"optimizee initialization no. {epoch}")

        # Sample a new model
        model = initialize_model(args.dataset, args.optimizee_model_name)
        model = model.to(device)
        model2 = initialize_model(args.dataset, args.optimizee_model_name)
        model2 = model2.to(device)

        # number of unrolls
        for k in (range(num_unroll_train // args.truncated_bptt_step)):

            meta_optimizer.reset_lstm(keep_states=k > 0, model=model)   #resets the initialized model parameters to that of the model
            window_state = meta_optimizer.snapshot_state()   # window-start LSTM state
            window_batches = []                               # same-batch replay cache

            #copying the paramters of the model to meta_model (required in SAM)
            for modelA, modelB in zip(model2.parameters(), model.parameters()):
                modelA.data.copy_(modelB.data)

            smooth_loss = torch.zeros(1, device=device)



            #one unrolling step
            for j in range(args.truncated_bptt_step):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_data_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                window_batches.append((x, y))

                # First we need to compute the gradients wrt model parameters at a point
                f_x = model(x)
                loss = F.nll_loss(f_x, y)
                model.zero_grad()
                loss.backward() #meta_optimizer params dont change

                #Update the values of the parameters using the meta optimizer, meta_model is the new model params
                model_temp = meta_optimizer.meta_update(model)  #model_temp points to the model used in initializing Meta_optimizer
                                                                #which will only update the meta_optimizer


                if j == args.truncated_bptt_step-1:
                    print(f"epoch>> {epoch}, unrolling>>> {k}, loss on last iteration : {loss.item()} ", flush= True)


            #find the meta loss of the meta optimizer
            loss_sum = torch.zeros(1, device=device)

            # for j in range(len(train_data_loader)):
            #     try:
            #         x, y = next(train_iter)
            #     except StopIteration:
            #         train_iter = iter(train_data_loader)
            #         x, y = next(train_iter)
            #     x, y = x.to(device), y.to(device)
            f_x = model_temp(x)
            loss = F.nll_loss(f_x, y)
            loss_sum = loss_sum + loss
            # /len(train_data_loader) scaling removed (Change.MD item 26)

            if args.if_smooth:
                smooth_loss = smooth_loss + smooth_reg_object.run(get_flat_grads(model))
                loss_sum = loss_sum + smooth_reg_object.lamb * smooth_loss


            # (pre-step backward removed: GAM/GSAM's get_grad() starts with
            #  zero_grad() and recomputes the task gradient itself, so the
            #  old backward here was discarded work — Change.MD item 32c.
            #  NOTE: legacy ascent mode therefore does NOT include L_smooth
            #  in the GAM/GSAM meta-objective; use reg_mode=penalty for the
            #  full Eq. 7 objective.)


            #################################################################################################################
            # employing gradient norm aware minimization


            loss_fn = F.nll_loss
            dummy_model = initialize_model(args.dataset, args.optimizee_model_name)
            dummy_model = dummy_model.to(device)
            gam_optimizer.LO_set_closure(opzee_model = model2, dummy_model=dummy_model, iter_num = k, loss_fn = loss_fn, train_data_loader = train_data_loader, batches=window_batches, start_state=window_state)
            # rho scheduler stepped once per epoch (below), not per window
            # (Change.MD item 32a/b)
            gam_optimizer.step()
            # torch.cuda.empty_cache()
            # gc.collect()
    ##########################################################################################################

            # for param in meta_optimizer.parameters():
            #     param.grad.data.clamp_(-1, 1)
    gc.collect()




def GSAM_reg_train_opt_func(
        num_unroll_train,
        train_data_loader,
        device,
        meta_optimizer,
        smooth_reg_object,
        args):
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
    print(f"N_train : {num_unroll_train}, epochs = {args.train_period}, ")





    # dict_ = {'model_optimizee' : model2, 'data_loder' : train_data_loader}
    gsam_optimizer = get_optim_and_schedulers(meta_optimizer, args = args) #, optzee_model = model2, data_loader = train_data_loader


    #t_period many epochs with the same value of N_train
    for epoch in range(args.train_period):

        train_iter = iter(train_data_loader)
        gsam_optimizer.update_rho_t()   # per-epoch rho schedule (item 32)

        print(f"optimizee initialization no. {epoch}")

        # Sample a new model
        model = initialize_model(args.dataset, args.optimizee_model_name)
        model = model.to(device)
        model2 = initialize_model(args.dataset, args.optimizee_model_name)
        model2 = model2.to(device)

        # number of unrolls
        for k in (range(num_unroll_train // args.truncated_bptt_step)):

            meta_optimizer.reset_lstm(keep_states=k > 0, model=model)   #resets the initialized model parameters to that of the model
            window_state = meta_optimizer.snapshot_state()   # window-start LSTM state
            window_batches = []                               # same-batch replay cache

            #copying the paramters of the model to meta_model (required in SAM)
            for modelA, modelB in zip(model2.parameters(), model.parameters()):
                modelA.data.copy_(modelB.data)

            if args.if_smooth:
                smooth_loss = torch.zeros(1, device=device)



            #one unrolling step
            for j in range(args.truncated_bptt_step):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_data_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                window_batches.append((x, y))

                # First we need to compute the gradients wrt model parameters at a point
                f_x = model(x)
                loss = F.nll_loss(f_x, y)
                model.zero_grad()
                loss.backward() #meta_optimizer params dont change

                #Update the values of the parameters using the meta optimizer, meta_model is the new model params
                model_temp = meta_optimizer.meta_update(model)  #model_temp points to the model used in initializing Meta_optimizer
                                                                #which will only update the meta_optimizer



                if j == args.truncated_bptt_step-1:
                    print(f"epoch>> {epoch}, unrolling>>> {k}, loss on last iteration : {loss.item()} ", flush= True)


            #find the meta loss of the meta optimizer
            loss_sum = torch.zeros(1, device=device)

            # for j in range(len(train_data_loader)):
            #     try:
            #         x, y = next(train_iter)
            #     except StopIteration:
            #         train_iter = iter(train_data_loader)
            #         x, y = next(train_iter)
            #     x, y = x.to(device), y.to(device)
            f_x = model_temp(x)
            loss = F.nll_loss(f_x, y)
            loss_sum = loss_sum + loss
            # /len(train_data_loader) scaling removed (Change.MD item 26)

            if args.if_smooth:
                smooth_loss = smooth_loss + smooth_reg_object.run(get_flat_grads(model))
                loss_sum = loss_sum + smooth_reg_object.lamb * smooth_loss


            # (pre-step backward removed: GAM/GSAM's get_grad() starts with
            #  zero_grad() and recomputes the task gradient itself, so the
            #  old backward here was discarded work — Change.MD item 32c.
            #  NOTE: legacy ascent mode therefore does NOT include L_smooth
            #  in the GAM/GSAM meta-objective; use reg_mode=penalty for the
            #  full Eq. 7 objective.)


            #################################################################################################################
            # employing gradient norm aware minimization


            loss_fn = F.nll_loss
            dummy_model = initialize_model(args.dataset, args.optimizee_model_name)
            dummy_model = dummy_model.to(device)
            gsam_optimizer.LO_set_closure(opzee_model = model2, dummy_model=dummy_model, iter_num = k, loss_fn = loss_fn, train_data_loader = train_data_loader, batches=window_batches, start_state=window_state)
            # rho scheduler stepped once per epoch (below), not per window
            gsam_optimizer.step()
            torch.cuda.empty_cache()
            gc.collect()
    ##########################################################################################################

            # for param in meta_optimizer.parameters():
            #     param.grad.data.clamp_(-1, 1)
    gc.collect()




def eval_opt_func(
        num_unroll_eval,
        train_data_loader,
        device,
        meta_optimizer,
        args,
        val_loader=None):
    """Checkpoint-selection / curriculum metric.
    val_loader: pass the held-out half from data_loader() so selection is
    on meta-validation data (Change.MD item 28); falls back to the train
    stream when None."""
    loader = val_loader if val_loader is not None else train_data_loader
    eval_loss = 0
    print("Evaluating optimizer:\n")
    for ep in range(args.eval_epochs):
        model = initialize_model(args.dataset, args.optimizee_model_name)
        model = model.to(device)
        # fresh hidden state per eval trajectory (Change.MD item 21) —
        # this was the 'has no attribute h0' crash + cross-epoch leak.
        meta_optimizer.reset_lstm(keep_states=False, model=model)
        train_iter = iter(loader)

        loss_sum = 0
        for j in range(num_unroll_eval):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            # First we need to compute the gradients wrt model parameters at a point
            f_x = model(x)
            loss = F.nll_loss(f_x, y)
            model.zero_grad()
            loss.backward()
            loss_sum = loss_sum + loss.item()
            if j%20 == 0:
                print(f"loss: {loss.item()}\n")
            # print(loss.item()-prev_loss.item(), loss.item())

            with torch.no_grad():
                meta_optimizer.meta_update(model)
        del model, loss
        gc.collect()

        eval_loss += loss_sum/num_unroll_eval
        print(f"Evaluation Epoch {ep}| loss : {loss_sum/num_unroll_eval}")
        
    return eval_loss/args.eval_epochs





#############################################################################################

#  Hessian Eigen Value Computation

def compute_hessian_spectrum(num_eigenvals, model, train_data_loader):
    if compute_hessian_eigenthings is None:
        raise ImportError("hessian_eigenthings is required for Hessian spectrum computation.")
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

 
def max_loss_PGD(model, eps, data_loader, device, r_norm= "L_inf"):

    #deep copy of the model, will be used for perturbation
    model_p = copy.deepcopy(model)
    # copy_gradients(model, model_p)
    step_size = eps/10
    
    actual_loss = compute_val_loss(data_loader, model_p)

    #perturbation
    with torch.no_grad():
        for params in model_p.parameters():
            params.add_(torch.randn(params.size()).to(device))
    # for param_orig, param_copy in zip(model.parameters(), model_p.parameters()):
    #     assert torch.ne(param_orig.data, param_copy.data), "Parameters are not equal"

    
    print(f"The actual test loss here is : {actual_loss}")

    flat_params = get_flat_params(model_p)
    center = flat_params.clone()
    perturbed_loss = []
    perturbed_loss.append(actual_loss.item())
    for _ in range(10):

        flat_params.requires_grad_()
        with torch.enable_grad():
            v_loss = 0
            test_iter = iter(data_loader)
            model_p = set_flat_params(model_p, flat_params)
            for e in range(len(data_loader)):
                try:
                    x, y = next(test_iter)
                    x, y = x.to(device), y.to(device)
                    f_x = model_p(x)
                    loss = F.nll_loss(f_x, y)
                    v_loss += loss
                    # if e == 0:
                    #     pred_y = f_x
                    #     act_y = y
                    # else:
                    #     torch.cat([pred_y, f_x], dim = 0)
                    #     torch.cat([act_y, y], dim = 0)

                except StopIteration:
                    test_iter = iter(data_loader)
            
            p_loss = v_loss/len(data_loader)
            perturbed_loss.append(p_loss.item())

            print(f"perturbed loss {_} : {perturbed_loss}")

            diff = p_loss - actual_loss

            grads = torch.autograd.grad(diff, [flat_params],allow_unused=True)[0]
            

            if r_norm == "L_inf":
            #for L inf norm
                flat_params = flat_params.detach() + step_size * torch.sign(grads.detach())
                flat_params = torch.clamp(flat_params, center - eps, center + eps)

            elif r_norm == "euc":
                direction = grads.detach().clone()
                direction /= torch.norm(direction) #frobenius norm by default/ eucliean norm for a vector

                # Update parameters using Euclidean distance constraint
                flat_params = flat_params.detach() + step_size * direction
                norm = torch.norm(flat_params - center)
                if norm > eps:
                    flat_params = center + (flat_params - center) * eps / norm



    perturbed_params = flat_params.detach()
    model_p = set_flat_params(model_p, perturbed_params)

    perturbed_loss.append(compute_val_loss(data_loader, model_p).detach().cpu().item())

    return perturbed_loss, actual_loss.item()



################################################################################################




def copy_gradients(from_model, to_model):
    for from_param, to_param in zip(from_model.parameters(), to_model.parameters()):
        if from_param.grad is not None:
            if to_param.grad is None:
                to_param.grad = torch.zeros_like(to_param)
                to_param.grad.data.copy_(from_param.grad.detach().data)
                # print('in function:', to_param.grad)

def get_flat_grads(model):
    flat = torch.tensor([]).to(device)
    for param in model.parameters():
        # print('get flat params')
        # print(param.grad)
        flat = torch.cat((flat,param.grad.view(-1).to(device)))
    
    return flat.requires_grad_()

def set_flat_grads(model, flat_grads):
    # Restore original shapes
    offset = 0
    for param in model.parameters():
        param_shape = param.size()
        param_flat_size = param.numel()
        param.grad = flat_grads[offset:offset + param_flat_size].view(*param_shape)
        offset += param_flat_size

    return model



################################################################################
# for gam ----- finding the maximum gradient in a neighbourhood
################################################################################








def max_grad_norm_PGD(model, eps, data_loader, device, r_norm= "L_inf"):

    #deep copy of the model, will be used for perturbation
    model_p = copy.deepcopy(model)
    copy_gradients(model, model_p)
    step_size = eps
    
    # for param in model_p.parameters():
    #     param.retain_grad()
    #     print(param)

    
    actual_grad_norm = torch.norm(get_flat_grads(model_p), p=2)

    #perturbation
    with torch.no_grad():
        for params in model_p.parameters():
            params.add_(torch.randn(params.size()).to(device))

    
    print(f"The actual parameter gradient norm is : {actual_grad_norm}")

    flat_params = get_flat_params(model_p)  #gradfn = catbackwards

    # print('flat params:', flat_params)
    perturb_grad_norm_list = []
    perturb_grad_norm_list.append(actual_grad_norm.item())
    
    flat_params2 = flat_params.clone().detach()    
    center = flat_params.clone()

    for _ in range(10):

        flat_params2.requires_grad_()

        #here the gradient_function of flat_param2 gets in model_p
        
        model_p = set_flat_params(model_p, flat_params2) #gradfn = ViewBackward
        
        #making the require_grads of the model as True
        
        
        # print(flat_params2)
        with torch.enable_grad():
            test_iter = iter(data_loader)
            # print('\n1\n', torch.norm(get_flat_grads(model_p), p=2))
            loss = 0
            for e in range(len(data_loader)):
                try:
                    x, y = next(test_iter)
                    x, y = x.to(device), y.to(device)
                    f_x = model_p(x)
                    loss_fn = F.nll_loss
                    loss += loss_fn(f_x, y)

                except StopIteration:
                    test_iter = iter(data_loader)

            loss = loss/len(data_loader)
            model_p.zero_grad()
            for param in model_p.parameters():
                param.retain_grad()
            grad = torch.autograd.grad(loss, [flat_params2], create_graph=True)[0]
            set_flat_grads(model_p, grad)
            # print('\ngrad:\n',grad)
            # loss.backward()
            # print("\nloss :", loss)
            # print('model param grad',[param.grad for param in model_p.parameters()])
            # print('flat_apram2 grad', flat_params2.grad)
            # print("model grad norm", torch.norm(get_flat_grads(model_p), p=2) )
            
            perturb_grad_norm = torch.norm(get_flat_grads(model_p), p=2) 
            # perturb_grad_norm = torch.norm(flat_params2.grad, p=2)

            perturb_grad_norm_list.append(perturb_grad_norm.item())

            diff = perturb_grad_norm - actual_grad_norm
            print(perturb_grad_norm, actual_grad_norm)
            # print("\ndiff :",diff)
            grads = torch.autograd.grad(diff, [flat_params2],allow_unused=True)[0]
            # print("grads:", grads)
            if r_norm == "L_inf":
            #for L inf norm
                flat_params2 = flat_params2.detach() + step_size * torch.sign(grads.detach())
                flat_params2 = torch.clamp(flat_params2, center - eps, center + eps)

            elif r_norm == "euc":
                direction = grads.detach().clone()
                direction /= torch.norm(direction) #frobenius norm by default/ eucliean norm for a vector

                # Update parameters using Euclidean distance constraint
                flat_params2 = flat_params2.detach() + step_size * direction
                norm = torch.norm(flat_params2 - center)
                if norm > eps:
                    flat_params2 = center + (flat_params2 - center) * eps / norm

    print("grad_list: ", perturb_grad_norm_list)


    # perturbed_params = flat_params2.detach()
    flat_params2.requires_grad_()
    model_p = set_flat_params(model_p, flat_params2)


    ########################### Finding the max grad norm ###############

    with torch.enable_grad():
        loss = 0
        test_iter = iter(data_loader)
        for e in range(len(data_loader)):
            try:
                x, y = next(test_iter)
                x, y = x.to(device), y.to(device)
                f_x = model_p(x)
                loss_fn = F.nll_loss
                loss += loss_fn(f_x, y)

            except StopIteration:
                test_iter = iter(data_loader)

        loss = loss/len(data_loader)
        model_p.zero_grad()
        for param in model_p.parameters():
            param.retain_grad()
        grad = torch.autograd.grad(loss, flat_params2)[0]
        set_flat_grads(model_p, grad)

        perturb_grad_norm = torch.norm(get_flat_grads(model_p), p=2)


    perturb_grad_norm_list.append(perturb_grad_norm.item())
    return perturb_grad_norm_list, actual_grad_norm.item()


# Keep legacy imports working while using the cleaned, fixed analysis implementations.
from analysis import (
    accuracy,
    compute_hessian_spectrum,
    compute_val_loss,
    copy_gradients,
    get_flat_grads,
    get_flat_params,
    max_grad_norm_PGD,
    max_loss_PGD,
    set_flat_grads,
    set_flat_params,
)

#########################################################################################################
#####  CONSOLIDATED ALGORITHM 1 — penalty-mode meta-training (reg_mode='penalty')                   #####
#####  L(phi) = L_task + lambda_smooth * L_smooth + lambda_reg * L_reg,   L_reg in {SAM,GSAM,GAM}   #####
#####  Implements the paper's Eq. 7 / Consolidated Algorithm 1 literally (Change.MD item 70).      #####
#########################################################################################################

def consolidated_reg_train_opt_func(
        num_unroll_train,
        train_data_loader,
        device,
        meta_optimizer,
        smooth_reg_object,
        args,
        val_loader=None):
    """One meta-training call (train.py invokes this once per outer epoch).

    Per TBPTT window:
      unroll T steps                              -> L_task = L_T (final step)
      PGA on the final gradient state             -> L_smooth        (Alg.1 b)
      shared ascent step on phi (rho, xi)         -> phi_adv         (Alg.1 c)
      re-unroll SAME window / SAME batches under phi_adv:
          SAM : L_reg = L_task(phi_adv)
          GSAM: L_reg = L_task(phi_adv) - L_task(phi)   (surrogate gap)
          GAM : L_reg = rho * || grad_phi L_task(phi_adv) ||
      total = L_task + ls*L_smooth + lr*L_reg     -> meta-update     (Alg.1 d)

    epsilon-hat is treated as constant when differentiating (the standard
    SAM convention): grads of L_task+ls*L_smooth are taken at phi, grads of
    lr*L_reg at phi_adv, summed into .grad, then one optimizer step.
    """
    if args.base_optimizer == 'sgd':
        LOs_optimizer = optim.SGD(meta_optimizer.parameters(), lr=args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.base_optimizer in ('adamw', 'AdamW'):
        LOs_optimizer = optim.AdamW(meta_optimizer.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    else:
        LOs_optimizer = optim.Adam(meta_optimizer.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    lam_s = float(getattr(args, 'smooth_lamb', 0.0)) if args.if_smooth else 0.0
    lam_r = float(getattr(args, 'lambda_reg', 0.0))
    xi = float(getattr(args, 'xi', 1e-12))
    reg = args.regularizer.split('_')[0] if args.regularizer.startswith(('sam', 'gsam', 'gam')) else args.regularizer
    if reg not in ('sam', 'gsam', 'gam'):
        raise ValueError(f"penalty mode supports sam/gsam/gam, got '{args.regularizer}'")

    phi_params = [p for p in meta_optimizer.parameters()]

    for epoch in range(args.train_period):
        train_iter = iter(train_data_loader)
        print(f"optimizee initialization no. {epoch}")
        model = initialize_model(args.dataset, args.optimizee_model_name).to(device)
        model2 = initialize_model(args.dataset, args.optimizee_model_name).to(device)

        for k in range(num_unroll_train // args.truncated_bptt_step):
            meta_optimizer.reset_lstm(keep_states=k > 0, model=model)
            window_state = meta_optimizer.snapshot_state()
            window_batches = []
            # window-start optimizee copy for the phi_adv replay
            for pA, pB in zip(model2.parameters(), model.parameters()):
                pA.data.copy_(pB.data)

            # ---- pass 1: unroll under phi -------------------------------
            for j in range(args.truncated_bptt_step):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_data_loader)
                    x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                window_batches.append((x, y))
                loss = F.nll_loss(model(x), y)
                model.zero_grad()
                loss.backward()
                model_temp = meta_optimizer.meta_update(model)
                if j == args.truncated_bptt_step - 1:
                    print(f"epoch>> {epoch}, unrolling>>> {k}, loss on last iteration : {loss.item()} ", flush=True)

            x, y = window_batches[-1]
            L_task = F.nll_loss(model_temp(x), y)                       # (a) final-step loss

            L_smooth = torch.zeros(1, device=device)
            if lam_s > 0.0:
                L_smooth = smooth_reg_object.run(get_flat_grads(model)) # (b)

            state_after_pass1 = meta_optimizer.snapshot_state()

            # pure task gradient for the ascent direction (Alg.1 line 16)
            g_task = torch.autograd.grad(L_task, phi_params, retain_graph=True,
                                         allow_unused=True)
            g_task = [torch.zeros_like(p) if g is None else g
                      for p, g in zip(phi_params, g_task)]
            g_norm = torch.sqrt(sum((g ** 2).sum() for g in g_task))

            # grads of (L_task + ls*L_smooth) at phi
            meta_optimizer.zero_grad()
            (L_task + lam_s * L_smooth).backward()
            L_task_val = float(L_task.detach())

            if lam_r > 0.0:
                # ---- (c) shared ascent step: phi_adv = phi + eps_hat ----
                scale = float(getattr(args, 'rho', 0.05)) / (float(g_norm) + xi)
                with torch.no_grad():
                    for p, g in zip(phi_params, g_task):
                        p.add_(g, alpha=scale)

                # re-unroll SAME window, SAME batches, SAME start state
                meta_optimizer.reset_lstm(keep_states=False, model=model2)
                meta_optimizer.restore_state(window_state)
                for j in range(args.truncated_bptt_step):
                    x, y = window_batches[j]
                    loss = F.nll_loss(model2(x), y)
                    model2.zero_grad()
                    loss.backward()
                    model2_temp = meta_optimizer.meta_update(model2)
                x, y = window_batches[-1]
                L_adv = F.nll_loss(model2_temp(x), y)

                if reg == 'sam':
                    L_reg = L_adv                                       # Alg.1 line 17, SAM
                elif reg == 'gsam':
                    L_reg = L_adv - L_task_val                          # surrogate gap
                else:  # gam — rho * ||grad_phi L_task(phi_adv)|| (HVP)
                    g_adv = torch.autograd.grad(L_adv, phi_params, create_graph=True,
                                                allow_unused=True)
                    g_adv = [torch.zeros_like(p) if g is None else g
                             for p, g in zip(phi_params, g_adv)]
                    L_reg = float(getattr(args, 'rho', 0.05)) * torch.sqrt(
                        sum((g ** 2).sum() for g in g_adv) + 1e-24)

                (lam_r * L_reg).backward()                              # accumulate at phi_adv

                # restore phi and the pass-1 hidden-state lineage
                with torch.no_grad():
                    for p, g in zip(phi_params, g_task):
                        p.sub_(g, alpha=scale)
                meta_optimizer.restore_state(state_after_pass1)
                # window k+1 continues from model (pass-1 trajectory).
                # BUGFIX (2026-07-31): this line used copy_params_TO(model),
                # which overwrote `model` (the pass-1 trajectory) with the
                # meta_model's CURRENT contents — i.e. the pass-2 phi_adv
                # endpoint — so every subsequent window silently continued
                # from the ADVERSARIAL unroll instead of the pass-1 one.
                # Resync must go the other way: meta_model <- model.
                # (The next window's reset_lstm(copy_params_from(model))
                # would resync anyway; doing it here keeps the invariant
                # explicit and window-local.)
                meta_optimizer.meta_model.copy_params_from(model)

            LOs_optimizer.step()                                        # (d) meta-update
        gc.collect()
