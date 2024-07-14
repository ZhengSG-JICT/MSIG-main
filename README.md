# MSIG-main
Multi-Scale Information Granule-Based Time Series Forecasting Model with Two-Stage Prediction Mechanism
# Requirements
The model is implemented using Python 3.9 with dependencies specified in requirements.txt


# Data Preparation

## Multivariate time series datasets

Download Traffic, Electricity, Exchange-rate datasets from https://github.com/laiguokun/multivariate-time-series-data. Uncompress them and move them to the data folder.

## Setup

### 1. Create conda environment(Optional)
```
conda create -n basisformer -y python=3.9 
conda activate MSIG
```

### 2. Install dependecies
Install the required packages
```
pip install -r requirements.txt
```

### 3. Download the data
We follow the same setting as previous work. The datasets for all the six benchmarks can be obtained from [[Autoformer](https://github.com/thuml/Autoformer)]. The datasets are placed in the 'all_six_datasets' folder of our project. The tree structure of the files are as follows:


```
MSIG\data\ETT
│
├─ECL
│
├─ETTh1
│
├─EXCHG
│
├─TRFC
│
├─ETTm2
│
├─BJPM5
│
├─SYPM5
│
└─SHPM5
│
└─covid_19
```

### 4. Experimental setup
The length of the historical input sequence is maintained at $96$, whereas the length of the sequence to be predicted is selected from a range of values, i.e., $\{96, 192, 336, 720\}$. Note that the input length is fixed to be 96 for all methods for a fair comparison. For ETTh2, ETTm2, ECL, TRFC, EXCHG, Covid_19 data sets, the evaluation is based on mean square error (MSE) and mean absolute error (MAE) measures. Other data sets are measured by RMSE, SMAPE, and MASEL.

## Main Results



## Train and Evaluates
### 1. Univariate forecasting
```
sh script/MSIG.sh
```

## Contact

If there are any issues, please ask in the GitHub Issue module

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/MAZiqing/FEDformer

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data


