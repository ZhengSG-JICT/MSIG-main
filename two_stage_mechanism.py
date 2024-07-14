
import torch
import torch.nn as nn
from fusion_mechanism import FeedForwardNetwork

#Phase I Attention Mechanism
class att_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.01):
        super(att_model, self).__init__()
        self.input_size = input_size  # number of features OF input
        self.hidden_size = hidden_size  # number of features in the hidden state
        self.output_size = output_size  # number of pred day
        self.num_layers = num_layers
        self.fnn = FeedForwardNetwork(hidden_size, hidden_size * 4, dropout)
        self.fnn_norm = torch.nn.LayerNorm(hidden_size)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers)
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.dp = nn.Dropout(dropout),
        self.tanh = nn.Tanh()

    def forward(self, input):
        pred = []
        for ii in range(len(input)):
            lstm_out, self.hidden_cell = self.lstm(input[ii])
            hid = self.hidden_cell[0]
            cid = self.hidden_cell[1]
            lstm_out_fnn = self.fnn(lstm_out)

            lstm_pred = self.fnn_norm(lstm_out_fnn)

            lstm_pred = self.tanh(lstm_pred)
            pred.append(lstm_pred)
        first_att = pred[0]
        for ii in range(1, len(pred)):
            first_att = torch.cat((first_att, pred[ii]), dim=1)
        return first_att


#Phase II Attention Mechanism
class GlobalAttention(nn.Module):
    def __init__(self, feature_size=512, n_heads=8, dropout=0.05, conv_kernel=24,
                 kernel=18, device='cuda', output=192):
        super(GlobalAttention, self).__init__()
        self.src_mask = None
        self.conv_kernel = conv_kernel
        self.kernel = kernel
        self.device = device
        self.output = output
        self.down_conv = nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                              kernel_size=kernel, padding=kernel, stride=kernel)

        self.padding_conv = nn.Conv1d(in_channels=feature_size, out_channels=feature_size,
                                    kernel_size=kernel, padding=0, stride=1)
        self.up_conv = nn.ConvTranspose1d(in_channels=feature_size, out_channels=feature_size,
                                         kernel_size=kernel, padding=0, stride=kernel//8)
        self.norm = torch.nn.LayerNorm(feature_size)
        self.act = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.05)
        self.wc = nn.Linear(feature_size, feature_size)
        self.wh = nn.Linear(feature_size, feature_size)

    def forward(self, input):

        x = input.permute(0, 2, 1)
        x1 = self.drop(self.act(self.down_conv(x)))

        x = torch.cat((x1, x), dim=-1)
        x = self.drop(self.act(self.padding_conv(x)))
        x = self.norm(self.wh(x.permute(0, 2, 1))).permute(0, 2, 1)
        x = self.drop(self.act(self.up_conv(x)))
        x = self.norm(self.wc(x.permute(0, 2, 1)))
        return x
