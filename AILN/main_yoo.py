import os
import argparse
import pickle
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model_yoo import MIModel
from data_loader import *
import metric
here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
parser = argparse.ArgumentParser()
parser.add_argument('--tvt_path', default='Yoo/tvt.pkl')
parser.add_argument('--item_num', type=int, default=8154, help='item number')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--feature_dim', type=int, default=100, help='the dimension of feature embedding')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')#
parser.add_argument('--reg', type=float, default=0.1, help='regular1')#
parser.add_argument('--epoch', type=int, default=1000 , help='the number of epochs to train for')
parser.add_argument('--testloader_file', default='Yoo/test.pkl')
parser.add_argument('--trainloader_file', default='Yoo/train.pkl')
parser.add_argument('--validateloader_file', default='Yoo/validate.pkl')
args = parser.parse_args()
item_ids = torch.arange(args.item_num).to(device)

def main():
    print('Loading data...')
    if os.path.exists(args.trainloader_file):
        with open(args.trainloader_file, 'rb') as f:
            train_dataset = pickle.load(f)
        with open(args.testloader_file, 'rb') as f:
            test_dataset = pickle.load(f)
        with open(args.validateloader_file, 'rb') as f:
            val_dataset = pickle.load(f)


    else:
        with open(args.tvt_path, 'rb') as f:
            data = pickle.load(f)
            train_data,val_data, test_data = data[0], data[1], data[2]
        train_cur_sess,  train_pos_item = train_data[0], train_data[1]
        print(train_cur_sess)

        test_cur_sess,  test_pos_item = test_data[0], test_data[1]
        val_cur_sess, val_pos_item = val_data[0], val_data[1]

        train_dataset = UnidirectTrainDataset(train_cur_sess,train_pos_item)
        test_dataset = UnidirectTestDataset(test_cur_sess,test_pos_item)
        val_dataset = UnidirectValidateDataset(val_cur_sess,val_pos_item)

        with open(args.trainloader_file, 'wb') as f:
            pickle.dump(train_dataset, f, pickle.HIGHEST_PROTOCOL)
        with open(args.testloader_file, 'wb') as f:
            pickle.dump(test_dataset, f, pickle.HIGHEST_PROTOCOL)
        with open(args.validateloader_file, 'wb') as f:
            pickle.dump(val_dataset, f, pickle.HIGHEST_PROTOCOL)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)




    model = MIModel(args.feature_dim, args.item_num).to(device)
    train_instance = len(train_loader)

    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    max_mrr = 0.
    tem = 0
    for epoch in range(args.epoch):
        trainForEpoch(train_instance, train_loader, model, optimizer, criterion)
        recall, mrr = validate(test_loader, model)  # 每个循环验证一次
        for i in range(3):
            print('Epoch {} validation: Recall: {:.4f}, MRR: {:.4f}, \n'.format(epoch + 1, recall[i],
                                                                                mrr[i]))
        if mrr[2] >= max_mrr:
            max_mrr = mrr[2]
            tem = 0
        else:
            tem += 1
            if tem == 20:
                break

def trainForEpoch(train_instance, train_loader, model, optimizer, criterion):
    model.train()
    batch_loss = torch.tensor(0).float().to(device)

    for i, data in enumerate(train_loader):
        item_represent = model.dropout((model.item_encode(item_ids))).to(device)
        cur_sess = data[0].squeeze(0).to(device)
        pos_items = data[1].squeeze(0).to(device)
        optimizer.zero_grad()
        rep = model(cur_sess)  #
        score = torch.sum(torch.mul(rep, (item_represent)), dim=-1)
        score = score.unsqueeze(dim=0)
        loss = criterion(score, pos_items)
        batch_loss += loss
        if ((i + 1) % args.batch_size) == 0 or (i + 1) == train_instance:
            l2_regularization = torch.tensor(0, dtype=torch.float32).to(device)
            for name, param in model.named_parameters():
                l2_regularization += args.reg * torch.norm(param, 2)  # L2 正则化
            batch_loss += l2_regularization
            batch_loss.backward()
            optimizer.step()
            batch_loss = torch.tensor(0).float().to(device)





def validate(test_loader, model):
    print('validating')
    model.eval()
    recalls = [[], [], []]
    mrrs = [[], [], []]
    item_represent = ((model.item_encode(item_ids))).to(device)

    with torch.no_grad():
        for data in (test_loader):
            cur_sess = data[0].squeeze(0).to(device)
            pos_items = data[1].squeeze(0).to(device)
            rep = model(cur_sess)
            score = torch.sum(torch.mul(rep, item_represent), dim=-1)
            logits = F.softmax(score, dim=0)
            K = [ 5, 15, 20]
            for i, k in enumerate(K):
                recall, mrr = metric.evaluate(logits, pos_items, k)
                recalls[i].append(recall)
                mrrs[i].append(mrr)


    mean_recall = []
    mean_mrr = []
    for rc in recalls:
        mean_recall.append(np.mean(rc))
    for mr in mrrs:
        mean_mrr.append(np.mean(mr))
    return mean_recall, mean_mrr


if __name__ == '__main__':
    main()
