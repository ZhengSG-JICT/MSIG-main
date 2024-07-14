import torch
import torch.nn as nn

import numpy as np
import copy
import math
from scipy import spatial
from fusion_mechanism import FeedForwardNetwork, Feature_capture, decoder_cat
from two_stage_mechanism import att_model, GlobalAttention
from embedding import DataEmbedding_enc
class MSIG(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, seq_length, dropout, dim, PE, SE, device=torch.device('cuda:0')):
        super(MSIG, self).__init__()
        self.output = output_dim
        self.seq_length = seq_length
        self.hidden = hidden_dim
        self.device = device
        self.num_layers = num_layers
        self.PE = PE
        self.SE = SE
        self.wh = nn.Linear(hidden_dim, hidden_dim)
        self.wc = nn.Linear(hidden_dim, hidden_dim)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.act = torch.nn.ReLU()
        self.Rule = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.05)
        # Encoding position coding layer
        self.embedding = DataEmbedding_enc(d_model=hidden_dim, dropout=dropout, c_in=dim, device=self.device)
        #Multi-scale fusion mechanism
        self.local_global = Feature_capture(feature_size=hidden_dim, dropout=dropout, kernel=PE, padding_kernel=6, device='cuda')
        self.decoder_cat =decoder_cat(hidden_size=hidden_dim)
        #Two-stage attention mechanism
        self.ATT = att_model(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, output_size=hidden_dim, dropout=dropout)
        self.globalattention = nn.Sequential(
            GlobalAttention(feature_size=hidden_dim, dropout=dropout, conv_kernel=self.SE,
                            kernel=self.SE//4, device='cuda', output=self.output),
            nn.Dropout(dropout),
            torch.nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        self.fnn = FeedForwardNetwork(hidden_size=hidden_dim, filter_size=hidden_dim, dropout_rate=dropout)
        self.projection = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, x):
        x = x.to(self.device)
        # Encoding position coding
        emb_x = self.embedding(x)
        # Multi-scale fusion mechanism
        enc_local, enc_global = self.local_global(emb_x)
        inter_local = self.decoder_cat(enc_global, emb_x)
        inter_local = self.norm(self.act(self.drop(self.fnn(inter_local))))
        enc_local.append(inter_local)
        #Two-stage attention mechanism
        dec_h = self.ATT(enc_local)
        dec_pred = self.globalattention(dec_h)
        dec_pred = self.fnn(dec_pred)
        dec_pred = self.norm(self.Rule(dec_pred))
        # prediction granule
        dec_pred = dec_pred[:, :self.output, :]
        dec_pred = self.projection(dec_pred)
        return dec_pred

