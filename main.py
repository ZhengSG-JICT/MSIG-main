import argparse
import os
import torch
import torch.nn as nn
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader, Dataset
from granule_decom import Granlue_Decom
from MSIG import MSIG
from exp import EXP
import time
class ProccessDataset(Dataset):
    def __init__(self, data, data_y, seq_length, pred_len):
        self.data = data
        self.data_y = data_y
        self.seq_length = seq_length
        self.pred_len = pred_len
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_length
        r_begin = s_end
        r_end = r_begin + self.pred_len
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y
    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_len + 1
def seed_random(seed=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MSIG')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='根路径')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='数据文件名')
    parser.add_argument('--data', type=str, default='ETTh1', help='data')
    parser.add_argument('--alpha', type=str, default='0.3', help='the granularity level of granule')
    parser.add_argument('--beta', type=str, default='0.3', help='the trend factor')
    parser.add_argument('--last_li', type=int, default=16, help='Minimum number of granules')
    parser.add_argument('--target', type=str, default='OT', help='Predict target characteristics')
    parser.add_argument('--train_epochs', type=int, default=10, help='Training frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size of training input data')
    parser.add_argument('--seq_len', type=int, default=96, help='Input length')
    parser.add_argument('--pred_len', type=int, default=96, help='Predict length')
    parser.add_argument('--itr', type=int, default=15, help='Number of experiments')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--d_model', type=int, default=512, help='d_model')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning_rate')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--num_layers', type=int, default=8, help='num_layers')
    parser.add_argument('--inverse', action='store_true', help='inverse', default=False)
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    print('Args in experiment:')
    print(args)
    seed_random(2)

    data = pd.read_csv(args.root_path + args.data_path, usecols=[args.target])


    GD = Granlue_Decom(args.alpha, args.beta, args.last_li)
    scaler = MinMaxScaler()
    data[args.target] = scaler.fit_transform(data[args.target].values.reshape(-1, 1))

    #Divide training set and test set
    train_data, test_data = train_test_split(data[args.target].values, test_size=0.3, shuffle=False)
    Q_train = GD(train_data)
    Q_test = GD(test_data)
    #Calculate compression size
    Q_train_PE = []
    Q_test_PE = []
    for ii in range(1, len(Q_train)):
        pe = (len(data) - Q_train[ii]) / (args.seq_len * (len(Q_train)-1))
        Q_train_PE.append(int(pe))
    for ii in range(1, len(Q_test)):
        pe = (len(data) - Q_test[ii]) / (args.seq_len * (len(Q_test)-1))
        Q_test_PE.append(int(pe))
    if np.amin(Q_train_PE) > 20 :
        for ii in range(len(Q_train_PE)):
            Q_train_PE[ii] = Q_train_PE[ii] // 10
    if np.amin(Q_test_PE) > 20 :
        for ii in range(len(Q_test_PE)):
            Q_test_PE[ii] = Q_test_PE[ii] // 10
    SE = int((len(data) - args.seq_len) / (args.seq_len * (len(Q_test))))
    #Data preprocessing, building multi-scale series
    train_data_multi = []
    test_data_multi = []
    for i in range(len(Q_train)-1):
        train_data_multi.append(torch.tensor(train_data[Q_train[i]:Q_train[len(Q_train)-1]], dtype=torch.float32))
    for j in range(len(Q_test)-1):
        test_data_multi.append(torch.tensor(test_data[Q_test[j]:Q_test[len(Q_test)-1]], dtype=torch.float32))
    train_data_new = train_data_multi[0].reshape(-1, 1)
    if len(Q_train)!=2:
        tran_len = len(train_data_multi[0])
        for i in range(1,len(train_data_multi)):
            train_data_multi[i] = nn.functional.pad(train_data_multi[i], (0, tran_len - len(train_data_multi[i])))
            train_data_new = torch.cat((train_data_new, train_data_multi[i].reshape(-1,1)), dim=1)
    train_data_new = train_data_new.detach().cpu().numpy()
    test_data_new = test_data_multi[0].reshape(-1, 1)
    if len(Q_test) != 2:
        test_len = len(test_data_multi[0])
        for i in range(1, len(test_data_multi) - 1):
            test_data_multi[i] = nn.functional.pad(test_data_multi[i], (0, test_len - len(test_data_multi[i])))
            test_data_new = torch.cat((test_data_new, test_data_multi[i].reshape(-1,1)), dim=1)
    test_data_new = test_data_new.detach().cpu().numpy()
    #Create data loader for training set and test set
    train_data_y = train_data_multi[0].reshape(-1, 1)
    test_data_y = test_data_multi[0].reshape(-1, 1)
    train_dataset = ProccessDataset(train_data_new, train_data_y, seq_length=args.seq_len, pred_len=args.pred_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataset = ProccessDataset(test_data_new, test_data_y, seq_length=args.seq_len, pred_len=args.pred_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    train_dim = train_data_new.shape[1]
    test_dim = test_data_new.shape[1]

    for ii in range(args.itr):
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp = EXP(args.learning_rate, args.train_epochs, args.pred_len,args.d_model, args.seq_len,
                  args.dropout, args.gpu, train_dim, Q_train_PE, SE, args.num_layers, scaler, args.inverse)
        time_now = time.time()
        exp.train(train_dataset, train_loader, test_dataset, test_loader)
        print('train time: ', time.time() - time_now)
        print('>>>>>>>start testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        time_now = time.time()
        exp.test(test_dataset, test_loader)
        print('test time: ', time.time() - time_now)
        torch.cuda.empty_cache()

