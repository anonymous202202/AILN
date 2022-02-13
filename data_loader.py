import torch
#The code belowing is  for Yoochoose

# class UnidirectTrainDataset(torch.utils.data.Dataset):
#
#     def __init__(self,train_cur_sess,train_pos_item):
#         self.train_cur_sess = train_cur_sess
#         self.train_pos_item = train_pos_item
#         self.train_size = len(train_cur_sess)
#     def __getitem__(self, index):
#         return torch.LongTensor(self.train_cur_sess[index]), torch.LongTensor(self.train_pos_item[index])
#     def __len__(self):
#         return self.train_size
#
#
# class UnidirectTestDataset(torch.utils.data.Dataset):
#     def __init__(self, test_cur_sess,test_pos_item):
#         self.test_cur_sess = test_cur_sess
#         self.test_pos_item = test_pos_item
#         self.test_size = len(test_cur_sess)
#
#     def __getitem__(self, index):
#         return torch.LongTensor(self.test_cur_sess[index]), torch.LongTensor(self.test_pos_item[index])
#     def __len__(self):
#         return self.test_size
#
# class UnidirectValidateDataset(torch.utils.data.Dataset):
#     def __init__(self,validate_cur_sess,
#                 validate_pos_item):
#         self.validate_cur_sess = validate_cur_sess
#         self.validate_pos_item = validate_pos_item
#         self.validate_size = len(validate_cur_sess)
#
#     def __getitem__(self, index):
#         return torch.LongTensor(self.validate_cur_sess[index]), torch.LongTensor(self.validate_pos_item[index])
#
#     def __len__(self):
#         return self.validate_size



#The code belowing is  for Tmall and Tafeng

class UnidirectTrainDataset(torch.utils.data.Dataset):

    def __init__(self,train_cur_sess,train_cur_cate,
                 train_pos_item,train_pos_cate):


        self.train_cur_sess = train_cur_sess
        self.train_cur_cate = train_cur_cate
        self.train_pos_item = train_pos_item
        self.train_pos_cate = train_pos_cate
        self.train_size = len(train_cur_sess)


    def __getitem__(self, index):


        return torch.LongTensor(self.train_cur_sess[index]), \
               torch.LongTensor(self.train_cur_cate[index]),\
               torch.LongTensor(self.train_pos_item[index]),\
               torch.LongTensor(self.train_pos_cate[index])

    def __len__(self):
        return self.train_size


class UnidirectTestDataset(torch.utils.data.Dataset):

    def __init__(self, test_cur_sess, test_cur_cate,
                test_pos_item, test_pos_cate):

        self.test_cur_sess = test_cur_sess
        self.test_cur_cate = test_cur_cate
        self.test_pos_item = test_pos_item
        self.test_pos_cate = test_pos_cate
        self.test_size = len(test_cur_sess)

    def __getitem__(self, index):

        return torch.LongTensor(self.test_cur_sess[index]), \
               torch.LongTensor(self.test_cur_cate[index]), \
               torch.LongTensor(self.test_pos_item[index]), \
               torch.LongTensor(self.test_pos_cate[index])

    def __len__(self):
        return self.test_size


class UnidirectValidateDataset(torch.utils.data.Dataset):

    def __init__(self,validate_cur_sess,validate_cur_cate,
                 validate_pos_item,validate_pos_cate):
        self.validate_cur_sess = validate_cur_sess
        self.validate_cur_cate = validate_cur_cate
        self.validate_pos_item = validate_pos_item
        self.validate_pos_cate = validate_pos_cate
        self.validate_size = len(validate_cur_sess)


    def __getitem__(self, index):

        return torch.LongTensor(self.validate_cur_sess[index]), \
               torch.LongTensor(self.validate_cur_cate[index]), \
               torch.LongTensor(self.validate_pos_item[index]), \
               torch.LongTensor(self.validate_pos_cate[index])

    def __len__(self):
        return self.validate_size


