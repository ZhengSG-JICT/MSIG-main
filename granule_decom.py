import pandas as pd
import argparse
import numpy as np
import torch
import torch.nn as nn
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit

class Granlue_Decom(nn.Module):
    def __init__(self, alpha, beta, last_li):
        super(Granlue_Decom, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.last_li = last_li
    #Initialize Information Granule
    def init_IG(self, data, i, ig_small):
        '''Quality requirements for computing information particles'''
        newdata = data[i: i + ig_small]
        m = int(np.sqrt(len(newdata))) + 1
        n = len(newdata)
        chazhi = m * m - n
        zero = np.zeros(chazhi)
        data_1 = np.concatenate((newdata, zero))
        datanew = np.reshape(data_1, (m, m))
        u = np.mean(newdata)
        V1 = ig_small
        s = 1 / (V1 - 1) * np.transpose(datanew - np.ones((m, m)) * u) * (datanew - np.ones((m, m)) * u)
        d = np.linalg.eigvals(s)
        vol = np.sum(d)
        V2 = np.exp(-np.sqrt(vol))
        Q_1 = np.sqrt(V1) * np.real(V2)

        '''Trend requirements for computing information granule'''
        x = np.arange(1, len(data))
        data_x = x[i:i + ig_small]
        new_data = newdata.reshape(-1)
        data_y = new_data
        a, b = self.calculate(data_x, data_y)
        T_1 = a
        return V1, u, Q_1, T_1

    #Building Information Granules
    def build_IG(self, data, i, j, ig_small):
        '''Quality requirements for computing information granules'''
        newdata = data[i: j]
        m = int(np.sqrt(len(newdata))) + 1
        n = len(newdata)
        u = np.mean(newdata)
        chazhi = m * m - n
        zero = np.zeros(chazhi)
        data_1 = np.concatenate((newdata, zero))
        datanew = np.reshape(data_1, (m, m))
        V1 = j - i
        s = 1 / (V1 - 1) * np.transpose(datanew - np.ones((m, m)) * u) * (datanew - np.ones((m, m)) * u)
        d = np.linalg.eigvals(s)
        vol = np.sum(d)
        V2 = np.exp(-np.sqrt(vol))
        Q_1 = np.sqrt(V1) * np.real(V2)
        '''Trend requirements for computing information granule'''
        x = np.arange(1, len(data))
        data_x = x[i:j]
        new_data = newdata.reshape(-1)
        data_y = new_data
        a, b = self.calculate(data_x, data_y)
        T_1 = a
        return Q_1, T_1, u
    #Linear fitting trend factor
    def func(self, x, a, b):
        return a * x + b
    #Cosine fitting trend factor
    def funcos(self, x, a, b):
        return a * np.cos(x) + b
    #Least square fitting
    def calculate(self, x, y):
        popt, pcov = curve_fit(self.func, x, y)
        a = popt[0]
        b = popt[1]
        return a, b
    def forward(self, data):
        df = data
        N = len(df)
        Q_n = [0]
        T = []  # Trend indicators of information granules
        Q = []  # Quality indicators of information granules

        i = 0
        j = 0
        t_1 = self.last_li + 1
        n_1, u_1, Q_1, T_1 = self.init_IG(df, i, self.last_li)
        Q_new = [0]  # Quality requirements after storing each data
        T_new = [0]  # Trend requirements after storing each data
        Q_new.append(Q_1)
        T_new.append(T_1)

        while t_1 < N:
            Q_2, T_2, u_new = self.build_IG(df, i, t_1, self.last_li)
            thr = np.var(Q_new)
            tre = np.amin(np.abs(np.diff(T_new)))
            Q_new.append(Q_2)
            thr_new = np.var(Q_new)
            T_new.append(T_2)
            #Whether the calculation meets the granulation requirements
            if ((Q_2 - u_new) < float(self.alpha) * thr and np.abs(T_new[len(T_new) - 1] - T_new[len(T_new) - 2]) > float(
                    self.beta) * tre):
                Q.append(Q_2)
                T.append(T_2)
                Q_n.append(t_1)
                Q_new = [0]
                T_new = [0]
                i = t_1
                t_1 = t_1 + self.last_li
                n_1, u_1, Q_1, T_1 = self.init_IG(df, i, self.last_li)
                Q_new.append(Q_1)
                T_new.append(T_1)
            t_1 = t_1 + 1
        Q_n.append(N)
        return Q_n
