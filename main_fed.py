import matplotlib
import matplotlib.pyplot as plt
import copy
import torch

from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FedAvg_enc
from models.test import test_img

import DataBalance
import DataProcessor

import numpy as np

import datetime
import crypten

matplotlib.use('Agg')


def train(net_glob, db, w_glob, args):
    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # originally assign clients and Fed Avg -> mediator Fed Avg
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(len(db.dp.mediator))]
    # 3 : for each synchronization round r=1; 2; . . . ; R do
    for iter in range(args.epochs):
        # 4 : for each mediator m in 1; 2; . . . ; M parallelly do
        for i, mdt in enumerate(db.mediator):
            # 5- :
            loss_locals = []
            if not args.all_clients:
                w_locals = []
            need_index = [db.dp.local_train_index[k] for k in mdt]
            local = LocalUpdate(args=args, dataset=dp, idxs=np.hstack(need_index))
            w, loss = local.train(
                net=copy.deepcopy(net_glob).to(args.device))  # for lEpoch in range(E): 在local.train完成
            if args.all_clients:
                w_locals[i] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    return net_glob


def test(net_glob, dp, args, is_self_balanced, imbalanced_way):
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dp, args, is_self_balanced, imbalanced_way)
    dp.type = 'test'
    acc_test, loss_test = test_img(net_glob, dp, args, is_self_balanced, imbalanced_way)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))

def train_enc(net_glob, db, w_glob, args):
    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # originally assign clients and Fed Avg -> mediator Fed Avg
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(len(db.dp.mediator))]
    # 3 : for each synchronization round r=1; 2; . . . ; R do
    for iter in range(args.epochs):
        # 4 : for each mediator m in 1; 2; . . . ; M parallelly do
        for i, mdt in enumerate(db.mediator):
            # 5- :
            loss_locals = []
            if not args.all_clients:
                w_locals = []
            for k in mdt:
                need_index = db.dp.local_train_index[k]
                local = LocalUpdate(args=args, dataset=dp, idxs=np.hstack(need_index))
                w, loss = local.train_enc(
                    net=copy.deepcopy(net_glob).to(args.device), w_enc=copy.deepcopy(w_glob))
                if args.all_clients:
                    w_locals[i] = copy.deepcopy(w)
                else:
                    w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # update global weights
        w_glob = FedAvg_enc(w_locals)

        # copy weight to net_glob
        w_glob_plaintext = dict()
        for key in w_glob_enc.keys():
            w_glob_plaintext[key] = w_glob[key].get_plain_text()
        net_glob.load_state_dict(w_glob_plaintext)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    return net_glob


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # new instances for DataProcessor and DataBalance
    dp = DataProcessor.DataProcessor()
    dp.get_input('mnist')
    imbalanced_way = ""
    if args.size_balance:
        dp.gen_size_imbalance([5000, 2000, 1000])
        imbalanced_way = "size"
    elif args.local_balance:
        dp.gen_local_imbalance(10, 5000, 0.8)
        imbalanced_way = "local"
    elif args.global_balance:
        dp.gen_global_imbalance(5, 2000, [500, 500, 1000, 1000, 1500, 1500, 3000, 1000, 0, 0])
        imbalanced_way = "global"
    # without self-balanced
    db = DataBalance.DataBalance(dp)
    db.assign_clients(balance=False)
    # load dataset and split users
    img_size = dp[0][0].shape

    # build original model
    net_glob = None
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    starttime = datetime.datetime.now()
    train(net_glob, db, w_glob, args)
    endtime = datetime.datetime.now()
    normal_time = endtime - starttime
    test(net_glob, dp, args, "non-self_balanced", imbalanced_way)

    print("normal time: {n}ms".format(n=normal_time.microseconds))

    print('-'*80)

    # build new model
    net_glob = None
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()
    # copy weights
    w_glob = net_glob.state_dict()
    # self balanced
    db = DataBalance.DataBalance(dp)

    starttime = datetime.datetime.now()
    db.z_score()
    endtime = datetime.datetime.now()
    merging_time = endtime - starttime

    starttime = datetime.datetime.now()
    db.assign_clients()
    endtime = datetime.datetime.now()
    resheduling_time = endtime - starttime

    dp.type = "train"
    starttime = datetime.datetime.now()
    train(net_glob, db, w_glob, args)
    endtime = datetime.datetime.now()
    training_time = endtime - starttime

    test(net_glob, dp, args,  "self_balanced", imbalanced_way)

    print("merging time: {m}ms; resheduling time: {r}ms; training time: {t}ms"
          .format(m=merging_time.microseconds,
                  r=resheduling_time.microseconds,
                  t=training_time.microseconds))

    print('-'*80)

    # 半诚实
    crypten.init()
    net_glob = None
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()
    # copy weights
    w_glob = net_glob.state_dict()
    # self balanced
    db = DataBalance.DataBalance(dp)

    starttime = datetime.datetime.now()
    db.z_score_enc()
    endtime = datetime.datetime.now()
    merging_time = endtime - starttime

    starttime = datetime.datetime.now()
    db.assign_clients_enc()
    endtime = datetime.datetime.now()
    resheduling_time = endtime - starttime

    dp.type = "train"
    starttime = datetime.datetime.now()
    w_glob_enc = dict()
    for key in w_glob.keys():
        w_glob_enc[key] = crypten.cryptensor(w_glob[key])
    train_enc(net_glob, db, w_glob_enc, args)
    endtime = datetime.datetime.now()
    training_time = endtime - starttime

    test(net_glob, dp, args, "self_balanced", imbalanced_way)

    print("merging time: {m}ms; resheduling time: {r}ms; training time: {t}ms"
          .format(m=merging_time.microseconds, r=resheduling_time.microseconds, t=training_time.microseconds))

    print('-'*80)

    # 合谋
    crypten.init()
    net_glob = None
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()
    # copy weights
    w_glob = net_glob.state_dict()
    # self balanced
    db = DataBalance.DataBalance(dp)

    starttime = datetime.datetime.now()
    db.z_score_col()
    endtime = datetime.datetime.now()
    merging_time = endtime - starttime

    starttime = datetime.datetime.now()
    db.assign_clients_col()
    endtime = datetime.datetime.now()
    resheduling_time = endtime - starttime

    dp.type = "train"
    starttime = datetime.datetime.now()
    w_glob_enc = dict()
    for key in w_glob.keys():
        w_glob_enc[key] = crypten.cryptensor(w_glob[key])
    train_enc(net_glob, db, w_glob_enc, args)
    endtime = datetime.datetime.now()
    training_time = endtime - starttime

    test(net_glob, dp, args, "self_balanced", imbalanced_way)

    print("merging time: {m}ms; resheduling time: {r}ms; training time: {t}ms"
          .format(m=merging_time.microseconds, r=resheduling_time.microseconds, t=training_time.microseconds))
