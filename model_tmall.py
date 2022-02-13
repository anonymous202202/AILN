import torch
import torch.nn as nn
import numpy as np
import copy
class AILNModel(nn.Module):
    def __init__(self, item_dim, cate_dim, item_num, cate_num):
        super(AILNModel, self).__init__()
        self.win_len = 3
        self.relation_threshold = 0.1
        self.item_dim = item_dim
        self.cate_dim = cate_dim
        self.item_num = item_num
        self.cate_num = cate_num
        self.item_encode = torch.nn.Embedding(self.item_num + 1,
                                              self.item_dim,
                                              padding_idx=0)  # Item embedding layer, 商品编码 +1是因为还有padding
        self.cate_encode = torch.nn.Embedding(self.cate_num + 1,
                                              self.cate_dim,
                                              padding_idx=0)  # Item embedding layer, 商品编码 +1是因为还有padding
        self.W_q = nn.Linear((self.cate_dim + self.item_dim), (self.cate_dim + self.item_dim), bias=False)  # intent
        self.att_pre_w = nn.Linear((self.cate_dim + self.item_dim), 1, bias=False)
        self.ct_dropout = nn.Dropout(0.75)
        self.dropout = nn.Dropout(0.25)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.rnn = torch.nn.GRU(
            input_size=(self.cate_dim + self.item_dim),
            hidden_size=(self.cate_dim + self.item_dim),
            num_layers=1
        )
    def final_ip(self, intents):
        all_p_h_state, p_h_state = self.rnn(intents.unsqueeze(-2))
        tem = (all_p_h_state).squeeze(-2)
        relevance = self.att_pre_w(tem).to(self.device)
        sim = torch.softmax(((relevance)), dim=0)
        f_i = (torch.sum(torch.mul(sim, tem), dim=0))
        return f_i
    def intent_update(self, o_item, n_item):
        keys = torch.cat((o_item, n_item.unsqueeze(0)))
        o_q = self.W_q(o_item)
        n_q = self.W_q(n_item.unsqueeze(0))
        relevance = torch.sum(torch.mul(o_q, keys), dim=-1)
        weights = torch.softmax(torch.div(relevance, np.sqrt((self.cate_dim + self.item_dim))), dim=-1)
        relevance1 = torch.sum(torch.mul(n_q, keys), dim=-1)
        weights1 = torch.softmax(torch.div(relevance1, np.sqrt((self.cate_dim + self.item_dim))), dim=-1)
        new_item0 = torch.sum(torch.mul(weights.unsqueeze(-1), keys), dim=0)
        new_item1 = torch.sum(torch.mul(weights1.unsqueeze(-1), keys), dim=0)
        final_item = torch.div(torch.add(new_item0, new_item1), 2)
        return final_item

    def forward(self, cur_sess, cur_cate):  #
        if len(cur_sess) <= self.win_len:
            item_memory = []  #
            intent_memory = []  #
            item_flag = {}  #
            for j, (item, cate) in enumerate(zip(cur_sess, cur_cate)):
                update_itemset = []
                item_memory.append(item)
                item_flag[item] = True
                new_item_embedding = self.ct_dropout(self.item_encode(item))
                new_cate_embedding = self.ct_dropout(self.cate_encode(cate))
                new_embedding = torch.cat((new_item_embedding, new_cate_embedding), dim=0)
                if j == 0:
                    intent_memory.append(new_embedding.unsqueeze(0))
                else:  # capture the relevance between the new item and intent in memory
                    k = len(intent_memory)
                    intent_memory.append(new_embedding.unsqueeze(0))
                    tem_list = []
                    for n in range(k):
                        old_item = item_memory[n]
                        update = True
                        if type(old_item) == tuple:
                            for unit in old_item:
                                if update == False:
                                    break
                                else:
                                    for tem in tem_list:
                                        if unit in tem:
                                            update = False
                                            break
                        if update:
                            old_intent = intent_memory[n]
                            relevance = torch.sigmoid(torch.sum(torch.mul(old_intent, new_embedding), dim=-1))
                            if relevance.item() > self.relation_threshold:  # represent a same intention
                                item_flag[item] = False
                                item_flag[old_item] = False
                                a = copy.copy(old_item)
                                if type(a) == tuple:
                                    b = a + tuple([item, ])
                                else:
                                    b = tuple([a, ]) + tuple([item, ])
                                item_memory.append(b)
                                update_itemset.append(b)
                                item_flag[b] = True
                                updated_item = self.intent_update(old_intent, new_embedding)  # gru update
                                intent_memory.append(updated_item.unsqueeze(0))
                            else:
                                tem_list.append(old_item)
                intent_num = len(update_itemset)  # ac abc ac=false
                for i in range(intent_num - 1):
                    item = update_itemset[i]
                    item_num1 = len(item)
                    for j in range(i + 1, intent_num):
                        new_item = update_itemset[j]
                        item_num2 = len(new_item)
                        if item_num1 < item_num2:
                            g = 0
                            for tem in (item):
                                if tem in new_item:
                                    g += 1
                            if g == item_num1:
                                item_flag[item] = False
                                break
            final_intent_sets = []
            for (key, intent_sets) in zip(item_flag.keys(), intent_memory):
                if item_flag[key]:
                    final_intent_sets.append(intent_sets)
            final_intent_sets = torch.cat(tuple(final_intent_sets), 0)
            intent = self.final_ip(final_intent_sets)
            return intent
        else:
            item_memory = []
            intent_memory = []
            item_flag = {}
            new_item_pos = {}
            for j, (item, cate) in enumerate(zip(cur_sess, cur_cate)):
                tem_list = []
                update_itemset = []
                new_item_pos[item.item()] = len(intent_memory)
                item_memory.append(item)
                item_flag[item] = True
                new_item_embedding = self.ct_dropout(self.item_encode(item))
                new_cate_embedding = self.ct_dropout(self.cate_encode(cate))
                new_embedding = torch.cat((new_item_embedding, new_cate_embedding), dim=0)
                if j == 0:
                    intent_memory.append(new_embedding.unsqueeze(0))
                elif j < self.win_len:  # capture the relevance between the new item and intent in memory
                    k = len(intent_memory)
                    intent_memory.append(new_embedding.unsqueeze(0))
                    for n in range(k):
                        old_item = item_memory[n]
                        update = True
                        if type(old_item) == tuple:
                            for unit in old_item:
                                if update == False:
                                    break
                                else:
                                    for tem in tem_list:
                                        if unit in tem:
                                            update = False
                                            break
                        if update:
                            old_intent = intent_memory[n]
                            relevance =torch.sigmoid (torch.sum(torch.mul(old_intent, new_embedding), dim=-1))
                            if relevance.item() > self.relation_threshold:  #
                                item_flag[item] = False
                                item_flag[old_item] = False
                                a = copy.copy(old_item)
                                if type(a) == tuple:
                                    b = a + tuple([item, ])
                                else:
                                    b = tuple([a, ]) + tuple([item, ])
                                item_memory.append(b)
                                update_itemset.append(b)
                                item_flag[b] = True
                                updated_item = self.intent_update(old_intent, new_embedding)  # gru update
                                intent_memory.append(updated_item.unsqueeze(0))
                            else:
                                tem_list.append(old_item)
                else:
                    item_tuple = []
                    intent_memory.append(new_embedding.unsqueeze(0))
                    idx = j - (self.win_len - 1)#begin index
                    win_items = cur_sess[idx:j]
                    begin_item_pos = new_item_pos[cur_sess[idx].item()]
                    intent = intent_memory[begin_item_pos]
                    item_set = item_memory[begin_item_pos]
                    item_tuple.append(item_set)
                    relevance = torch.sigmoid(torch.sum(torch.mul(intent, new_embedding), dim=-1))
                    if relevance.item() > self.relation_threshold:  #
                        item_flag[item] = False
                        item_flag[item_set] = False
                        a = copy.copy(item_set)
                        b = tuple([a, ]) + tuple([item, ])
                        update_itemset.append(b)
                        item_memory.append(b)
                        item_flag[b] = True
                        updated_item = self.intent_update(intent, new_embedding)  # gru update
                        intent_memory.append(updated_item.unsqueeze(0))
                    else:
                        tem_list.append(item_set)
                    for k,im in enumerate(win_items[1:]):
                        length = len(item_tuple)
                        item_tuple.append(im)
                        tem = []
                        for im1 in item_tuple[:-1]:
                            c_im1 = copy.copy(im1)
                            if type(c_im1) == torch.Tensor:
                                n_im1 = tuple([ c_im1,]) + tuple([im, ])
                            else:
                                n_im1 = c_im1 + tuple([im, ])
                            tem.append(n_im1)
                        item_tuple += tem
                        for tp in item_tuple[length:]:
                            b_idx = new_item_pos[cur_sess[idx + k + 1].item()]
                            e_idx = new_item_pos[cur_sess[idx+ k + 2].item()]
                            for item_tp,intent_tp in zip(item_memory[b_idx:e_idx],intent_memory[b_idx:e_idx]):
                                if  type(item_tp) == type(tp)  and item_tp == tp :
                                    update = True
                                    if type(item_tp) == tuple:
                                        for unit in item_tp:
                                            if update == False:
                                                break
                                            else:
                                                for tem in tem_list:
                                                    if unit in tem:
                                                        update = False
                                                        break
                                    if update:
                                        relevance = torch.sigmoid(torch.sum(torch.mul(intent_tp, new_embedding), dim=-1))
                                        if relevance.item() > self.relation_threshold:  #
                                            item_flag[item] = False
                                            item_flag[item_tp] = False
                                            a = copy.copy(tp)
                                            if type(a) == torch.Tensor:
                                                b = tuple([a,]) + tuple([item, ])
                                            else:b =a + tuple([item, ])
                                            update_itemset.append(b)
                                            item_memory.append(b)
                                            item_flag[b] = True
                                            updated_item = self.intent_update(intent_tp, new_embedding)  # gru update
                                            intent_memory.append(updated_item.unsqueeze(0))
                                        else:
                                            tem_list.append(item_tp)
                                    break
                intent_num = len(update_itemset)  # ac abc ac=false
                for i in range(intent_num - 1):
                    item = update_itemset[i]
                    item_num1 = len(item)
                    for j in range(i + 1, intent_num):
                        new_item = update_itemset[j]
                        item_num2 = len(new_item)
                        if item_num1 < item_num2:
                            g = 0
                            for tem in (item):
                                if tem in new_item:
                                    g += 1
                            if g == item_num1:
                                item_flag[item] = False
                                break
            final_intent_sets = []
            for (key, intent_sets) in zip(item_flag.keys(), intent_memory):
                if item_flag[key]:
                    final_intent_sets.append(intent_sets)
            final_intent_sets = torch.cat(tuple(final_intent_sets), 0)
            intent = self.final_ip(final_intent_sets)
            return intent



