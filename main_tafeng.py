'''
Reference: https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch
'''
import os
import argparse
import pickle
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model_tafeng import AILNModel
import pandas as pd
from data_loader1 import *
import metric
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--tvt_path', default='Tafeng/tvt_ailn.pkl')#IJCAI/Yoo
parser.add_argument('--item_num', type=int, default=7069 , help='item number')
parser.add_argument('--cate_num', type=int, default=990, help='item number')
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--embed_dim', type=int, default=100, help='the dimension of item embedding')
parser.add_argument('--feature_dim', type=int, default=50, help='the dimension of item feature')
parser.add_argument('--epoch', type=int, default=1000 , help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--testloader_file', default='Tafeng/test_ailn.pkl',help='the path of test set')
parser.add_argument('--trainloader_file', default='Tafeng/train_ailn.pkl',help='the path of train set')
parser.add_argument('--valloader_file', default='Tafeng/val_ailn.pkl',help='the path of validation set')
args = parser.parse_args()
item_ids = torch.arange(args.item_num + 1).to(device)
cate_ids = torch.zeros_like(item_ids).to(device)


def get_itemcate():#for IJCAI and Tafeng datasets
    itemcate = {0:0}
    data = pd.read_csv('Tafeng/reid_data_fin.csv', names=['time_stamp',"user_id", 'cate_id', 'item_id'],
                       dtype={'time_stamp': str, 'user_id': int, 'cate_id': int, 'item_id': int},
                       header=None, skiprows=1)
    for i in range(len(data)):
        item = data.at[i, 'item_id']
        if item not in itemcate.keys():
            cate = data.at[i, 'cate_id']
            itemcate[item] = cate
    return itemcate
itemcate = get_itemcate()
for i,item in enumerate(item_ids):
    item_id = item.item()
    cate_ids[i] = torch.tensor(itemcate[item_id]).long().to(device)

def main():
    print('Loading data...')
    if os.path.exists(args.trainloader_file) and os.path.exists(args.testloader_file) and os.path.exists(args.valloader_file):
        with open(args.trainloader_file, 'rb') as f:
            train_dataset = pickle.load(f)
        with open(args.testloader_file, 'rb') as f:
            test_dataset = pickle.load(f)
        with open(args.valloader_file, 'rb') as f:
            val_dataset = pickle.load(f)
    else:
        with open(args.tvt_path, 'rb') as f:
            data = pickle.load(f)
            train_data,val_data, test_data = data[0], data[1], data[2]
        train_cur_sess, train_cur_cate,train_pos_item, train_pos_cate= \
            train_data[0], train_data[1], train_data[2], train_data[3]

        val_cur_sess, val_cur_cate,  val_pos_item, val_pos_cate = \
            val_data[0], val_data[1], val_data[2], val_data[3]

        test_cur_sess, test_cur_cate, test_pos_item, test_pos_cate = \
            test_data[0], test_data[1], test_data[2], test_data[3]


        train_dataset = UnidirectTrainDataset( train_cur_sess, train_cur_cate,

                                              train_pos_item,train_pos_cate)
        val_dataset = UnidirectTestDataset(val_cur_sess, val_cur_cate,
                                             val_pos_item, val_pos_cate)

        test_dataset = UnidirectTestDataset( test_cur_sess, test_cur_cate,
                                            test_pos_item, test_pos_cate)

        with open(args.trainloader_file, 'wb') as f:
            pickle.dump(train_dataset, f, pickle.HIGHEST_PROTOCOL)
        with open(args.testloader_file, 'wb') as f:
            pickle.dump(test_dataset, f, pickle.HIGHEST_PROTOCOL)
        with open(args.valloader_file, 'wb') as f:
            pickle.dump(val_dataset, f, pickle.HIGHEST_PROTOCOL)


    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = AILNModel(args.embed_dim, args.feature_dim, args.item_num,args.cate_num).to(device)
    train_instance = len(train_loader)
    optimizer = optim.Adam(model.parameters(),args.lr)
    criterion = nn.CrossEntropyLoss()

    max_mrr = 0.
    tem = 0
    for epoch in (range(args.epoch)):

        trainForEpoch(train_instance, train_loader, model, optimizer, criterion)
        recall, mrr = validate(test_loader, model)  # 每个循环验证一次
        for i in range(3):
            print('Epoch {} validation: Recall: {:.4f}, MRR: {:.4f} \n'.format(epoch + 1, recall[i],
                                                                               mrr[i]))
        if mrr[2] >= max_mrr:
            max_mrr = mrr[2]
            tem = 0
        else:
            tem += 1
            if tem == 20:
                break

def trainForEpoch(train_instance,train_loader, model, optimizer,criterion):
    model.train()
    sum_epoch_loss = 0
    batch_loss = torch.tensor(0).float().to(device)
    for i, data in enumerate(train_loader):
        print(data)
        item_represent = model.dropout((model.item_encode(item_ids))).to(device)
        cate_represent = model.dropout(model.cate_encode(cate_ids)).to(device)
        item_represent = torch.cat((item_represent, cate_represent), dim=-1)
        cur_sess = data[0].squeeze(0).to(device)
        cur_cate = data[1].squeeze(0).to(device)
        pos_items = data[2].squeeze(0).to(device)
        optimizer.zero_grad()
        intents = model( cur_sess,cur_cate)
        score = torch.sum(torch.mul(intents,(item_represent)), dim=-1)
        score = score.unsqueeze(dim=0)
        loss = criterion(score, pos_items)
        batch_loss += loss
        loss_val = loss.item()
        sum_epoch_loss += loss_val
        if ((i+1) % 128) == 0 or (i+1) == train_instance:
            lamda = torch.tensor(0.1).to(device)
            l2_regularization = torch.tensor(0, dtype=torch.float32).to(device)
            for param in model.parameters():
                l2_regularization += torch.norm(param, 2)  # L2 正则化
            batch_loss += lamda * l2_regularization
            batch_loss .backward()
            optimizer.step()
            batch_loss = torch.tensor(0).float().to(device)

    #
    # print(f'sum_epoch_loss:{sum_epoch_loss}')


def validate(test_loader, model):
    print('validating')
    model.eval()
    recalls = [[], [], []]
    mrrs = [[], [], []]
    item_represent = ((model.item_encode(item_ids))).to(device)
    cate_represent = (model.cate_encode(cate_ids)).to(device)
    item_represent = torch.cat((item_represent, cate_represent), dim=-1)
    with torch.no_grad():
        for data in (test_loader):
            cur_sess = data[0].squeeze(0).to(device)
            cur_cate = data[1].squeeze(0).to(device)
            pos_items = data[2].squeeze(0).to(device)
            intents = model(cur_sess,cur_cate)  # 一个用户的所有数据  一个用户一个batch   True, False
            score = torch.sum(torch.mul(intents, item_represent), dim=-1)
            logits = F.softmax(score, dim=0)
            K = [5,15,20]
            for i,k in enumerate(K):
                recall, mrr = metric.evaluate(logits,pos_items, k)
                recalls[i].append(recall)
                mrrs[i].append(mrr)
    mean_recall = []
    mean_mrr = []
    for rc in recalls:
        mean_recall.append(np.mean(rc))
    for mr in mrrs:
        mean_mrr.append(np.mean(mr))
    return mean_recall, mean_mrr
#
#
if __name__ == '__main__':
    main()
