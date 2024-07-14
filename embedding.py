import torch
import torch.nn as nn
import math
class DataEmbedding_enc(nn.Module):
    def __init__(self, d_model, dropout=0.1, c_in=4, device=torch.device('cuda:0')):
        super(DataEmbedding_enc, self).__init__()
        self.device = device
        self.position = PositionalEmbedding(d_model=d_model, max_len=5000)
        self.token = TokenEmbedding(c_in=c_in, d_model=d_model, device=self.device)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x):
        x = self.position(x) + self.token(x)
        return self.dropout(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, device):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.dim = c_in
        self.device = device
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        if x.shape[2] < self.dim:
            zeros = torch.zeros((x.shape[0], x.shape[1], self.dim-x.shape[2]), device=self.device)
            x = torch.cat((x, zeros), dim=-1)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x