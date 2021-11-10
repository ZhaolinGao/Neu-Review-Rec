# -*- encoding: utf-8 -*-
import time
import random
import math
import fire

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import ReviewData
from framework import Model
import models
import config
import scipy.sparse as sp
from collections import defaultdict


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def train(**kwargs):

    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    # 3 data
    train_data = ReviewData(opt.data_root, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    val_data = ReviewData(opt.data_root, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f'train data: {len(train_data)}; test data: {len(val_data)}')

    # train test set
    train_set = sp.dok_matrix((opt.user_num - 2, opt.item_num - 2), dtype=np.float32)
    for i in train_data.data:
        train_set[i[0], i[1]] = 1.0
    test_set = defaultdict(set)
    for i in range(len(val_data.data)):
        if val_data.scores[i] >= 3:
            test_set[val_data.data[i][0]].add(val_data.data[i][1])

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # training
    print("start training....")
    # min_loss = 1e+10
    # best_res = 1e+10
    best_res = [0]
    mse_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    smooth_mae_func = nn.SmoothL1Loss()

    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        total_maeloss = 0.0
        model.train()
        print(f"{now()}  Epoch {epoch}...")
        for idx, (train_datas, scores) in enumerate(train_data_loader):

            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            train_datas = unpack_input(opt, train_datas)

            optimizer.zero_grad()
            output = model(train_datas)
            mse_loss = mse_func(output, scores)
            total_loss += mse_loss.item() * len(scores)

            mae_loss = mae_func(output, scores)
            total_maeloss += mae_loss.item()
            smooth_mae_loss = smooth_mae_func(output, scores)

            if opt.loss_method == 'mse':
                loss = mse_loss
            if opt.loss_method == 'rmse':
                loss = torch.sqrt(mse_loss) / 2.0
            if opt.loss_method == 'mae':
                loss = mae_loss
            if opt.loss_method == 'smooth_mae':
                loss = smooth_mae_loss
            loss.backward()
            optimizer.step()

            if opt.fine_step:
                if idx % opt.print_step == 0 and idx > 0:
                    print("\t{}, {} step finised;".format(now(), idx))
                    recall, ndcg = predict_r_n(model, opt, train_set, test_set)
                    print(recall, ndcg)
                    val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
                    print(val_loss, val_mse, val_mae)
                    # if val_loss < min_loss:
                    #     model.save(name=opt.dataset, opt=opt.print_opt)
                    #     min_loss = val_loss
                    #     print("\tmodel save")
                    # if val_loss > min_loss:
                    #     best_res = min_loss

        scheduler.step()
        mse = total_loss * 1.0 / len(train_data)
        print(f"\ttrain data: loss:{total_loss:.4f}, mse: {mse:.4f};")
        recall, ndcg = predict_r_n(model, opt, train_set, test_set)
        print(recall, ndcg)
        val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
        print(val_loss, val_mse, val_mae)
        # if val_loss < min_loss:
        #     model.save(name=opt.dataset, opt=opt.print_opt)
        #     min_loss = val_loss
        #     print("model save")
        # if val_mse < best_res:
        #     best_res = val_mse
        print("*"*30)

    print("----"*20)
    print(f"{now()} {opt.dataset} {opt.print_opt} best_res:  {best_res}")
    print("----"*20)


def test(**kwargs):

    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    assert(len(opt.pth_path) > 0)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)
    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    model.load(opt.pth_path)
    print(f"load model: {opt.pth_path}")
    test_data = ReviewData(opt.data_root, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"{now()}: test in the test datset")
    predict_loss, test_mse, test_mae = predict(model, test_data_loader, opt)


def predict(model, data_loader, opt):
    total_loss = 0.0
    total_maeloss = 0.0
    model.eval()
    with torch.no_grad():
        for idx, (test_data, scores) in enumerate(data_loader):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            test_data = unpack_input(opt, test_data)

            output = model(test_data)
            mse_loss = torch.sum((output-scores)**2)
            total_loss += mse_loss.item()

            mae_loss = torch.sum(abs(output-scores))
            total_maeloss += mae_loss.item()

    data_len = len(data_loader.dataset)
    mse = total_loss * 1.0 / data_len
    mae = total_maeloss * 1.0 / data_len
    print(f"\tevaluation reslut: mse: {mse:.4f}; rmse: {math.sqrt(mse):.4f}; mae: {mae:.4f};")
    model.train()
    return total_loss, mse, mae


def predict_r_n(model, opt, train_set, test_set):
    batch_size = opt.batch_size
    current_user = 0
    num_user = opt.user_num - 2
    num_item = opt.item_num - 2

    model.eval()
    with torch.no_grad():

        # get all the embeddings
        user_embeddings = []
        item_embeddings = []

        if num_user <= num_item:
            num_batch = math.ceil(num_user / batch_size)
            for i in range(num_batch):
                if i == (num_batch - 1):
                    users = np.arange(i*batch_size, num_user)
                    items = np.arange(i*batch_size, num_user)
                else:
                    users = np.arange(i*batch_size, (i+1)*batch_size)
                    items = np.arange(i*batch_size, (i+1)*batch_size)
                test_data = unpack_input(opt, (users, items), evaluation=True)
                user_feature, item_feature = model.net(test_data)
                user_embeddings.append(user_feature)
                item_embeddings.append(item_feature)
            num_batch = math.ceil((num_item - num_user) / batch_size)
            for i in range(num_batch):
                if i == (num_batch - 1):
                    users = np.repeat(0, num_item - num_user - i*batch_size)
                    items = np.arange(i*batch_size + num_user, num_item)
                else:
                    users = np.repeat(0, batch_size)
                    items = np.arange(i*batch_size + num_user, (i+1)*batch_size + num_user)
                test_data = unpack_input(opt, (users, items), evaluation=True)
                user_feature, item_feature = model.net(test_data)
                item_embeddings.append(item_feature)

        else:
            num_batch = math.ceil(num_item / batch_size)
            for i in range(num_batch):
                if i == (num_batch - 1):
                    users = np.arange(i*batch_size, num_item)
                    items = np.arange(i*batch_size, num_item)
                else:
                    users = np.arange(i*batch_size, (i+1)*batch_size)
                    items = np.arange(i*batch_size, (i+1)*batch_size)
                test_data = unpack_input(opt, (users, items), evaluation=True)
                user_feature, item_feature = model.net(test_data)
                user_embeddings.append(user_feature)
                item_embeddings.append(item_feature)
            num_batch = math.ceil((num_user - num_item) / batch_size)
            for i in range(num_batch):
                if i == (num_batch - 1):
                    users = np.arange(i*batch_size + num_item, num_user)
                    items = np.repeat(0, num_user - num_item - i*batch_size)
                else:
                    users = np.arange(i*batch_size + num_item, (i+1)*batch_size + num_item)
                    items = np.repeat(0, batch_size)
                test_data = unpack_input(opt, (users, items), evaluation=True)
                user_feature, item_feature = model.net(test_data)
                user_embeddings.append(user_feature)

        user_embeddings = torch.cat(user_embeddings, 0)
        item_embeddings = torch.cat(item_embeddings, 0)

        # generate perdictions
        predictions = []
        for i in range(num_user):
            prediction = []
            num_batch = math.ceil(num_item / batch_size)
            for j in range(num_batch):
                if j == (num_batch - 1):
                    users = np.repeat(i, num_item-j*batch_size)
                    items = np.arange(j*batch_size, num_item)
                else:
                    users = np.repeat(i, batch_size)
                    items = np.arange(j*batch_size, (j+1)*batch_size)
                ui_feature = model.fusion_net(user_embeddings[users], item_embeddings[items])
                ui_feature = model.dropout(ui_feature)
                output = model.predict_net(ui_feature, users, items).squeeze(1).detach().cpu().numpy()
                prediction.append(output)
            predictions.append(np.concatenate(prediction))
        predictions = np.stack(predictions, axis=0)
        
    topk = 20
    predictions[train_set.nonzero()] = np.NINF

    ind = np.argpartition(predictions, -topk)
    ind = ind[:, -topk:]
    arr_ind = predictions[np.arange(len(predictions))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(predictions)), ::-1]
    pred_list = ind[np.arange(len(predictions))[:, None], arr_ind_argsort]

    recall = []
    for k in [5, 10, 20]:
        recall.append(recall_at_k(test_set, pred_list, k))

    all_ndcg = ndcg_func([*test_set.values()], pred_list[list(test_set.keys())])
    ndcg = [all_ndcg[x-1] for x in [5, 10, 20]]

    model.train()

    return recall, ndcg


def ndcg_func(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        # calculate idcg
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len-1]

        dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(actual)
    true_users = 0
    for i, v in actual.items():
        act_set = set(v)
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    assert num_users == true_users
    return sum_recall / true_users


def unpack_input(opt, x, evaluation=False):

    if evaluation:
        uids = x[0]
        iids = x[1]
    else:
        uids, iids = list(zip(*x))
        uids = list(uids)
        iids = list(iids)

    user_reviews = opt.users_review_list[uids]
    user_item2id = opt.user2itemid_list[uids]  # 检索出该user对应的item id

    user_doc = opt.user_doc[uids]

    item_reviews = opt.items_review_list[iids]
    item_user2id = opt.item2userid_list[iids]  # 检索出该item对应的user id
    item_doc = opt.item_doc[iids]

    data = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc]
    data = list(map(lambda x: torch.LongTensor(x).cuda(), data))
    return data


if __name__ == "__main__":
    fire.Fire()
