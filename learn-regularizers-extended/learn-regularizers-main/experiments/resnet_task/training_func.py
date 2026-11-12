
import functools
from tqdm import tqdm
import torch
import numpy as np
from torch.autograd import Variable

from meta_module import *

# LOSS FUNCTION!!!
criterion = nn.CrossEntropyLoss()


def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    return acc


# training function for Resnet

def detach_var(v):
    var = w(Variable(v.data, requires_grad=True))
    var.retain_grad()
    return var

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)




def fit_val_loss_acc(opt_net, target_data, val_data_loader, target_optimizee, optim_it, out_mul=1):

    #use the should_train parameter to evaluate the task and not train the

    opt_net.eval()   #switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc.

    target = target_data  #it is the dataset, optimizee(target) yields loss, output
    optimizee = w(target_optimizee)
    n_params = 0
    for name, p in optimizee.named_parameters():
        n_params += int(np.prod(p.size()))

    #there are two outputs of LSTM (long term= Cell State C_t, and Hidden State = h_t)
    hidden_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    cell_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]



    ###############  keep track of the gradient values at the inference time  ########################

    # grad_list_test = []
    ##############################

    train_loss = []
    test_loss= []
    train_acc = []
    test_acc = []

    for iteration in tqdm(range(1, optim_it + 1), "iteration: "):  #optim_it = number of iterations the optimizee is trained using LO(inner training steps)

        #the loss is always found using all the datapoints (not single or batch loss)
        # difference between consecutive losses will decrease quadratically


        loss, pred, real  = optimizee(target)
        train_loss_iter = loss.item()
        train_loss.append(train_loss_iter)
        tr_acc = accuracy(pred, real)
        train_acc.append(tr_acc)


        # iteration_mod_unroll = iteration % unroll

        loss.backward(retain_graph=False)

        #gradients wrt all the parameters are updated


        offset = 0
        result_params = {}
        hidden_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        cell_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]

        grad_list_each_step = []



        for name, p in optimizee.named_parameters():
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            gradients = detach_var(p.grad.view(cur_sz, 1))


            ###############  keep track of the gradient values at the inference time  ########################

            grad_list_each_step.append(gradients.tolist())

            ############################################


            updates, new_hidden, new_cell = opt_net(
                gradients,
                [h[offset:offset+cur_sz] for h in hidden_states],
                [c[offset:offset+cur_sz] for c in cell_states]
            )
            #len(new_hidden) = 2

            for i in range(len(new_hidden)):
                hidden_states2[i][offset:offset+cur_sz] = new_hidden[i]
                cell_states2[i][offset:offset+cur_sz] = new_cell[i]
            result_params[name] = p + updates.view(*p.size()) * out_mul
            result_params[name].retain_grad()

            offset += cur_sz


        #####finding the average of hidden states after each iteration of optim_it


        # previous_loss = loss

#######   unroll when necessary to update the params of the LO   ############
        # if iteration_mod_unroll == 0:


            #need to update the grads for optimizee parameters too since they also are connected to the RNN in the computational graph
            #reducing them to zero is necessary
        optimizee = w(target_optimizee)
        optimizee.load_state_dict(result_params)
        optimizee.zero_grad()
        hidden_states = [detach_var(v) for v in hidden_states2]
        cell_states = [detach_var(v) for v in cell_states2]


#######  steps when no unrolling happens ##########################

        # for name, p in optimizee.all_named_parameters():
        #     rsetattr(optimizee, name, result_params[name])
        # assert len(list(optimizee.all_named_parameters()))
        # hidden_states = hidden_states2
        # cell_states = cell_states2


################################################################################

########       VALIDATION: ##########
        loss_val_iter = 0
        pred_y = []
        act_y = []
        for batch in val_data_loader:
            val_x, val_y = batch

            loss_1, valout_pred, val_y = optimizee(batch)
            loss_val_iter += loss_1

            pred_y.extend(valout_pred)
            act_y.extend(val_y)

        acc_val_iter = accuracy(pred_y,act_y)
        test_loss.append(loss_val_iter)
        test_acc.append(acc_val_iter)


        print(f"Training loss:{train_loss_iter} Training Accuracy: {tr_acc} Validation Loss: {loss_val_iter} Validation Accuracy: {acc_val_iter}")
################################################################################


    #all_losses_ever is the loss that optimizee gives after each iteration
    return {"optimizee": optimizee, "train_loss" :train_loss, "train_acc": train_acc, "test_loss": test_loss, "test_acc": test_acc }