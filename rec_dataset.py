import torch
from torch_sparse import spspmm
import numpy as np
import os, pdb
from time import time
import scipy.sparse as sp
from scipy.sparse import coo_matrix, diags
import numba as nb
import random
import argparse


@nb.njit()
def negative_sampling(training_user, training_item, traindata, num_item, num_negative):
    '''
    return: [u,i,j] for training, u interacted with i, not interacted with j
    '''
    trainingData = []
    for k in range(len(training_user)):
        u = training_user[k]
        pos_i = training_item[k]
        for _ in range(num_negative):
            neg_j = np.random.randint(0, num_item - 1)
            while neg_j in traindata[u]:
                neg_j = np.random.randint(0, num_item - 1)
            trainingData.append([u, pos_i, neg_j])
    return np.array(trainingData)


@nb.njit()
def Uniform_sampling(batch_users, traindata, num_item):
    trainingData = []
    for u in batch_users:
        pos_items = traindata[u]
        pos_id = np.random.randint(low=0, high=len(pos_items), size=1)[0]
        pos_item = pos_items[pos_id]
        neg_item = random.randint(0, num_item - 1)
        while neg_item in pos_items:
            neg_item = random.randint(0, num_item - 1)
        trainingData.append([u, pos_item, neg_item])
    return np.array(trainingData)


class Dataset(object):
    def __init__(self, args):
        self.device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
        self.args = args
        self.social_dict = {}
        self.data_path = args.data_path
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.num_node = self.num_user + self.num_item
        self.batch_size = args.batch_size
        ### load and process dataset ###
        self.load_data()
        self.data_to_numba_dict()

        # prepare matrix A for Z_matrix
        self.social_edge = len(self.social_i)

        self.get_A_deg()
        self.get_AA()
        # prepare matrix B for R_matrix
        self.community_edge = len(self.training_user)

        self.get_B_deg()

    def sp_eye(self, n):
        indices = torch.Tensor([list(range(n)), list(range(n))])
        values = torch.FloatTensor([1.0] * n)
        return torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n]).to(self.device)

    def txtSplit(self, filename):
        f = open(filename)  ## May should be specific for different subtasks
        train_hash_data = {}
        val_hash_data = {}
        test_ratio = 0.2
        for _, line in enumerate(f):
            arr = line.strip().split("\t")
            rnd = random.random()
            x, y = int(arr[0]), int(arr[1])
            if rnd <= test_ratio:
                if x in val_hash_data:
                    val_hash_data[x].append(y)
                else:
                    val_hash_data[x] = [y]
            else:
                if x in train_hash_data:
                    train_hash_data[x].append(y)
                else:
                    train_hash_data[x] = [y]
        return train_hash_data, val_hash_data

    def txt2np(self, filename):
        f = open(filename)  ## May should be specific for different subtasks
        hash_data = {}
        for _, line in enumerate(f):
            arr = line.strip().split("\t")
            rnd = random.random()
            x, y = int(arr[0]), int(arr[1])
            if x in hash_data:
                hash_data[x].append(y)
            else:
                hash_data[x] = [y]
        return hash_data

    def load_data(self):
        self.traindata, self.valdata = self.txtSplit(self.data_path + 'rating.train' + str(self.args.fold))
        # self.valdata = np.load(self.data_path + 'valdata.npy', allow_pickle=True).tolist()
        self.testdata = self.txt2np(self.data_path + 'rating.test' + str(self.args.fold))
        try:
            self.user_users = self.txt2np(self.data_path + 'links.txt')
            user_users_dict = {}
            for u, users in self.user_users.items():
                for v in users:
                    if u not in user_users_dict:
                        user_users_dict[u] = set()
                    user_users_dict[u].add(v)

                    if v not in user_users_dict:
                        user_users_dict[v] = set()
                    user_users_dict[v].add(u)

            # self-loop
            if self.args.self_loop == 'yes':
                for u in range(self.num_user):
                    if u not in user_users_dict:
                        user_users_dict[u] = set()
                    user_users_dict[u].add(u)

            self.social_i, self.social_j = [], []
            for u, users in user_users_dict.items():
                self.social_i.extend([u] * len(users))
                self.social_j.extend(users)
            assert(len(self.social_i) == len(self.social_j))
            print('successfull load social networks')

            self.training_user, self.training_item = [], []
            for u, items in self.traindata.items():
                self.training_user.extend([u] * len(items))
                self.training_item.extend(items)
            assert(len(self.training_user) == len(self.training_item))
            print('successfull load community networks')
            # attacked_user_users = self.add_noisy_social_links(ratio=0.2)
            # np.save(self.data_path + 'attacked_user_users_0.2.npy', attacked_user_users)
        except:
            pass

    # 转换回 PyTorch 稀疏张量
    def to_torch_sparse(self, matrix):
        coo = matrix.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        return torch.sparse_coo_tensor(indices, values, matrix.shape, dtype=torch.float32).coalesce()

    def get_B_deg(self):
        # 创建稀疏矩阵 B
        data = np.ones(len(self.training_user))
        B = coo_matrix((data, (self.training_user, self.training_item)), shape=(self.num_user, self.num_item))
        B = B.tocsr()  # 转换为 CSR 格式以加速操作

        # 计算度矩阵
        deg = np.sqrt(B.sum(axis=1).A1)
        deg_T = np.sqrt(B.sum(axis=0).A1)

        # 计算倒数，保持0度点为0
        d_values = np.where(deg != 0, 1 / deg, 0)
        d_T_values = np.where(deg_T != 0, 1 / deg_T, 0)

        # 计算归一化矩阵
        D = diags(d_values)
        D_T = diags(d_T_values)
        B_norm = D @ B @ D_T

        # 计算行归一化和列归一化矩阵
        B_row_norm = D @ D @ B
        B_line_norm = B @ D_T @ D_T

        self.B = self.to_torch_sparse(B).to(self.device)
        self.B_norm = self.to_torch_sparse(B_norm).to(self.device)
        self.B_deg = torch.sparse_coo_tensor(
            indices=torch.LongTensor([[i for i in range(self.num_user)], [0] * self.num_user]),
            values=torch.FloatTensor(deg),
            size=[self.num_user, 1]
        ).to(self.device)
        self.B_deg_t = torch.sparse_coo_tensor(
            indices=torch.LongTensor([[0] * self.num_item, [i for i in range(self.num_item)]]),
            values=torch.FloatTensor(deg_T),
            size=[1, self.num_item]
        ).to(self.device)
        self.B_row_norm = self.to_torch_sparse(B_row_norm).to(self.device)
        self.B_line_norm = self.to_torch_sparse(B_line_norm).to(self.device)

    def get_A_deg(self):
        # 创建稀疏矩阵 A
        data = np.ones(len(self.social_i))
        A = coo_matrix((data, (self.social_i, self.social_j)), shape=(self.num_user, self.num_user))
        A = A.tocsr()  # 转换为 CSR 格式以加速操作

        # 计算度矩阵
        deg = A.sum(axis=1).A1
        deg_sqrt = np.sqrt(deg)
        deg_inv_sqrt = np.where(deg != 0, 1.0 / deg_sqrt, 0)

        # 计算归一化矩阵
        D_inv_sqrt = diags(deg_inv_sqrt)
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        self.A = self.to_torch_sparse(A).to(self.device)
        self.A_norm = self.to_torch_sparse(A_norm).to(self.device)
        self.A_deg = torch.sparse_coo_tensor(
            indices=torch.LongTensor([[i for i in range(self.num_user)], [0] * self.num_user]),
            values=torch.FloatTensor(deg_sqrt),
            size=[self.num_user, 1]
        ).to(self.device)
        self.A_deg_t = torch.sparse_coo_tensor(
            indices=torch.LongTensor([[0] * self.num_user, [i for i in range(self.num_user)]]),
            values=torch.FloatTensor(deg_sqrt),
            size=[1, self.num_user]
        ).to(self.device)

        deg_log = np.log(deg)
        deg_inv_log = np.where(deg_log !=0, 1.0 / deg_log, 0 )
        D_inv_log = diags(deg_inv_log)
        self.D_log = self.to_torch_sparse(D_inv_log).to(self.device)
        D_inv = D_inv_sqrt @ D_inv_sqrt
        self.D = self.to_torch_sparse(D_inv).to(self.device)

        A_row_norm = D_inv_sqrt @ D_inv_sqrt @ A
        self.A_row_norm = self.to_torch_sparse(A_row_norm).to(self.device)
        A_sqrt = D_inv_sqrt @ A
        self.A_sqrt = self.to_torch_sparse(A_sqrt).to(self.device)
        ##AA & Z2 part
        if self.args.Z2_type == 'AA':
            # row norm
            # AA_norm = D_inv_sqrt @ D_inv_sqrt @ A
            # self.AA_norm = self.to_torch_sparse(AA_norm).to(self.device)
            self.AA_norm = self.A_row_norm
        elif self.args.Z2_type == 'AA*A': # AA*A
            # AA * A
            AA_norm_scipy = A_row_norm.dot(A_row_norm.T).multiply(A)
            # AA_scipy = A.dot(A).multiply(A)
            # deg = np.array(AA_scipy.sum(axis=1)).flatten()
            # inv_sqrt_deg = np.where(deg != 0, 1/deg, 0)
            # D_inv_sqrt = diags(inv_sqrt_deg)
            # AA_norm_scipy = D_inv_sqrt.dot(AA_scipy)
            self.AA_norm = self.to_torch_sparse(AA_norm_scipy).to(self.device)
        else :
            self.AA_norm = self.A_row_norm

    def Sparse_Mul(self, mat1, mat2, m, k, n):
        out_indices, out_values = spspmm(
            indexA=mat1.indices(), valueA=mat1.values(),
            indexB=mat2.indices(), valueB=mat2.values(),
            m=m, k=k, n=n
        )
        new_mat = torch.sparse_coo_tensor(indices=out_indices, values=out_values, size=[m, n]).to(self.device)
        new_mat = new_mat.coalesce()
        return new_mat
    def get_AA(self):
        pass

    def data_to_numba_dict(self):
        self.traindict = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.int64[:], )
        for key, values in self.traindata.items():
            if len(values) > 0:
                self.traindict[key] = np.asarray(values, dtype=np.int64)

        self.valdict = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.int64[:], )
        for key, values in self.valdata.items():
            if len(values) > 0:
                self.valdict[key] = np.asarray(values, dtype=np.int64)

        self.testdict = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.int64[:], )
        for key, values in self.testdata.items():
            if len(values) > 0:
                self.testdict[key] = np.asarray(values, dtype=np.int64)


    def convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        # pdb.set_trace()
        return indices, coo.data, coo.shape


    def _uniform_sampling(self):
        batch_num = int(len(self.training_user) / self.batch_size) + 1
        for _ in range(batch_num):
            batch_users = random.sample(list(self.traindata.keys()), self.batch_size)
            batch_data = Uniform_sampling(nb.typed.List(batch_users), self.traindict, self.num_item)
            yield batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]

    def _batch_sampling(self, num_negative):
        t1 = time()
        ### 三元组采样使用numba加速
        triplet_data = negative_sampling(nb.typed.List(self.training_user), nb.typed.List(self.training_item),
                                         self.traindict, self.num_item, num_negative)
        print('prepare training data cost time:{:.4f}'.format(time() - t1))
        batch_num = int(len(triplet_data) / self.batch_size) + 1
        indexs = np.arange(triplet_data.shape[0])
        np.random.shuffle(indexs)
        for k in range(batch_num):
            index_start = k * self.batch_size
            index_end = min((k + 1) * self.batch_size, len(indexs))
            if index_end == len(indexs):
                index_start = len(indexs) - self.batch_size
            batch_data = triplet_data[indexs[index_start:index_end]]
            yield batch_data[:, 0], batch_data[:, 1], batch_data[:, 2]
