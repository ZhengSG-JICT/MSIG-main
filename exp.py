
import torch.nn as nn
import torch
import numpy as np
from torch import optim
from MSIG import MSIG
from metrics import metric
class EXP(nn.Module):
    def __init__(self, learning_rate, train_epochs, pred_len, d_model, seq_len, dropout, gpu,train_dim, Q_train_PE, SE, num_layers, scaler, inverse):
        super(EXP, self).__init__()
        self.learning_rate = learning_rate
        self.train_epochs = train_epochs
        self.pred_len = pred_len
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout
        self.gpu = gpu
        self.dim = train_dim
        self.Q_train_PE = Q_train_PE
        self.SE = SE
        self.num_layers = num_layers
        self.scaler = scaler
        self.inverse = inverse
        self.model = self._build_model()
    def _build_model(self, ):
        model = MSIG(output_dim=self.pred_len, hidden_dim=self.d_model, seq_length=self.seq_len, dropout=self.dropout,
                     device=self.gpu, dim=self.dim, PE=self.Q_train_PE, SE=self.SE, num_layers=self.num_layers).to(
            self.gpu)
        return model

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for x, y in vali_loader:
            outputs = self.model(x)
            loss = criterion(outputs, y.to(self.gpu))
            total_loss.append(loss.item())
            loss = criterion(outputs.detach().cpu(), y.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, train_data, train_loader, test_data, test_loader):
        path = './checkpoint'
        #model = MFIG(output_dim=self.pred_len, hidden_dim=self.d_model, seq_length=self.seq_len, dropout=self.dropout,
                     #device=self.gpu, dim=self.dim, PE = self.Q_train_PE, SE= self.SE, num_layers=self.num_layers).to(self.gpu)
        train_steps = len(train_loader)
        model_optim = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(verbose=True)
        for epoch in range(self.train_epochs):
            self.model.train()
            iter_count = 0
            train_loss = []
            for x, y in train_loader:
                iter_count += 1
                model_optim.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y.to(self.gpu))
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
            train_loss = np.average(train_loss)
            valid_loss = self.vali(test_data, test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Valid Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss))
            
            early_stopping(valid_loss, self.model, path)

            #adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return

    def test(self, test_data, test_loader):
        path = './checkpoint'
        #test_data, test_loader = self._get_data(flag='test', scale=True, inverse=False)
        best_model_path = path+'/'+'checkpoint.pth'
        criterion = nn.MSELoss()
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        total_loss = []
        preds = []
        trues = []
        for x, y in test_loader:
            outputs = self.model(x)
            #loss = criterion(outputs, y.to(self.gpu))
            #total_loss.append(loss.item())
            preds.append(outputs.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())
        #total_loss = np.average(total_loss)
        preds = np.array(preds)
        trues = np.array(trues)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        if self.inverse:
            predictions = self.scaler.inverse_transform(preds.reshape(-1, 1))
            actual = self.scaler.inverse_transform(trues.reshape(-1, 1))
           
            preds = predictions.reshape(-1, self.pred_len, 1)
            trues = actual.reshape(-1, self.pred_len, 1)
        # result save
        folder_path = './result'+'/'

        pred_leng = self.pred_len
        mae, mse, rmse, mape, mspe, mase_l, smape, R2, mase = metric(preds, trues, pred_leng)
        print('mae:{}, mse:{}, rmse:{}, mape:{}, mspe:{}, mase_l:{}, smape:{}, R2:{}, mase:{}'.format(mae, mse, rmse, mape, mspe, mase,
                                                                                    smape, R2, mase))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, mase, smape]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self,verbose=False, delta=0):
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss
