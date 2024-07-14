import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def SMAPE(pred, true):
    return np.mean((2.0 * np.abs(pred - true) / (np.abs(true)+np.abs(pred))))

def naive_forecast(y:np.array, season:int=7):
  "naive forecast: season-ahead step forecast, shift by season step ahead"
  return y[:-season]

def MASE_L(pred, true, season:int=7):
  # Mean Absolute Value
  return np.mean(np.abs(pred-true) / MAE(true[season:], naive_forecast(true, season)))

def MASE(pred, true):
  # Mean Absolute Value
  return np.mean(np.abs(pred-true) / MAE(true[1:], naive_forecast(true, 1)))

def R2(pred, true):
    return 1-(np.sum((pred - true.mean(0))**2) / np.sum((true - true.mean(0))**2))

def metric(pred, true, pred_len):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    mase = MASE(pred, true)
    mase_l = MASE_L(pred, true, pred_len)
    smape = SMAPE(pred, true)
    r2 = R2(pred, true)
    return mae,mse,rmse,mape,mspe, mase_l, smape, r2, mase