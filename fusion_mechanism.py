import torch.nn as nn
import torch
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

        self.initialize_weight(self.layer1)
        self.initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

    def initialize_weight(self, x):
        nn.init.xavier_uniform_(x.weight)
        if x.bias is not None:
            nn.init.constant_(x.bias, 0)


class Feature_capture(nn.Module):
    def __init__(self, feature_size=512, dropout=0.05, kernel=[24], padding_kernel=6, device='cuda'):
        super(Feature_capture, self).__init__()
        self.src_mask = None
        self.kernel = kernel
        #print(kernel)
        self.padding_kernel = padding_kernel
        self.device = device
        self.local_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=i, padding=i, stride=i)
                                   for i in kernel])
        self.padding_conv = nn.ModuleList([nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                             kernel_size=i, padding=0, stride=1)
                                   for i in kernel])
        self.global_conv = nn.ModuleList([nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size,
                                                            kernel_size=i, padding=0, stride=i)
                                         for i in kernel])

        self.fnn = FeedForwardNetwork(feature_size, feature_size*4, dropout)
        self.fnn_norm = torch.nn.LayerNorm(feature_size)
        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)

    # 局部-全局模块体系结构
    def local_padding_global(self, input, local_feature, padding, global_feature):
        batch, seq_len, channel = input.shape
        x = input.permute(0, 2, 1)

        x1 = self.drop(self.act(local_feature(x)))
        x = torch.cat((x, x1), dim=-1)
        x = self.drop(self.act(padding(x)))
        x = self.norm((x).permute(0, 2, 1)).permute(0, 2, 1)

        x = self.drop(self.act(global_feature(x)))
        x = x[:, :, :seq_len]   # truncate
        x = self.norm(x.permute(0, 2, 1) + input)
        return x1, x

    def forward(self, feat):
        multi_local = []
        multi_global = []
        for i in range(len(self.kernel)):
            src_local, src_global = self.local_padding_global(feat, self.local_conv[i], self.global_conv[i], self.padding_conv[i])
            src_local = src_local.permute(0, 2, 1)
            multi_local.append(src_local)
            multi_global.append(src_global)
        return multi_local, multi_global

#局部特征整合
class decoder_cat(nn.Module):
    def __init__(self, hidden_size):
        super(decoder_cat, self).__init__()
        self.wh = nn.Linear(hidden_size, hidden_size)
        self.wc = nn.Linear(hidden_size, hidden_size)

    def forward(self, queries, cells):
        bi = []
        for i in range(len(queries)):
            queries[i] = queries[i] + self.wh(cells)
            b = F.cosine_similarity(queries[i], cells, dim=0)
            bi.append(b)

        att = queries[0] * bi[0]
        for j in range(1, len(queries)):
            newatt = queries[j] * bi[j]
            att = torch.cat((newatt, att), dim=1)
        return att