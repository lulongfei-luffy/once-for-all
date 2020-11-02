from numpy import ravel
from sklearn.svm import SVR
from Net.data_process import Dataprocess
# from main import dataloader
import torch
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor  # 集成算法
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet  # 批量导入要实现的回归算法
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
import logging
import time
import os
def dataloader(ratio, batch_size, file_path):
    dataprocess = Dataprocess(file_path)
    all_feats, gpu, cpu = dataprocess.encode()

    totalnum = len(cpu)
    cpu = torch.tensor(cpu).reshape(totalnum, -1)
    gpu = torch.tensor(gpu).reshape(totalnum, -1)
    # feature = all_feats[:int(0.8*totalnum)]
    train_dataset = TensorDataset(all_feats[:int(ratio[0] * totalnum)], cpu[:int(ratio[0] * totalnum)])
    # valid_dataset = TensorDataset(all_feats[int(ratio[0] * totalnum):int((ratio[1]+ratio[0]) * totalnum)], cpu[int(ratio[0] * totalnum):int((ratio[1]+ratio[0]) * totalnum)])
    test_dataset =  TensorDataset(all_feats[int((ratio[0]) * totalnum):],cpu[int((ratio[0]) * totalnum):])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    # valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=True, drop_last=True)
    return train_loader, test_loader

time_ = time.strftime('%Y%m%d%H%M%S%S')
if not os.path.exists('../ML_logs/{}'.format(time_)):
    os.mkdir('../ML_logs/{}'.format(time_))
logname = '../ML_logs/{}/log.txt'.format(time_)
logging.basicConfig(level=logging.INFO, filename=logname)
logger = logging.getLogger('ML-models')

file_path = '../dataset/latency_dataset_2_50000.csv'
ratio = [0.8,0.]
batch_size = 64

logger.info(file_path)
logger.info(ratio)
logger.info(batch_size)

# dataprocess = Dataprocess(file_path=file_path)
# all_feats, gpu, cpu = dataprocess.encode()
# linear_svr = SVR(kernel='linear')
train_loader, test_loader = dataloader(ratio=ratio,batch_size=batch_size,file_path=file_path)
# for i, (input, label) in enumerate(train_loader):
#     linear_svr.fit(input, ravel(label))
# for j, (input,label) in enumerate(test_loader):
#     linear_svr_pre_y = linear_svr.predict(input)
#     print('MSE %f' % (mean_squared_error(label,linear_svr_pre_y)))

model_br =BayesianRidge(alpha_1=1e-5, alpha_2=1e-5, lambda_1=1e-5,lambda_2=1e-5)
model_lr = LinearRegression()
model_etc = ElasticNet()
model_svr =SVR()
model_gbr = GradientBoostingRegressor()
model_names = ['BayesianRidge','LinearRegression','ElasticNet','SVR','GradientBoostingRegressor']
model_dic =  [model_br,model_lr,model_etc,model_svr,model_gbr]
cv_score_list, pre_y_list = [], []
for model in model_dic:
    # train
    logger.info(model)
    print(model)
    print('training............')
    for i, (input, label) in enumerate(train_loader):
        model.fit(input,ravel(label))

    # test
    print('testing.............')
    RMSE = 0.
    for j, (input, label) in enumerate(test_loader):
        pre_y = model.predict(input)
        # logger.info('pre_y',pre_y,'label', label)
        RMSE += mean_squared_error(label, pre_y)
    print('MSE %f' % (RMSE/float(j+1)))
    logger.info('MSE %f' % (RMSE/float(j+1)))
    logger.info('-'*80)
    print('-'*60)


import numpy as np  # numpy库

from sklearn.svm import SVR  # SVM中的回归算法

from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # 批量导入指标算法
import pandas as pd  # 导入pandas
import matplotlib.pyplot as plt  # 导入图形展示库


