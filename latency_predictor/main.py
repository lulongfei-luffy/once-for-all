# import math
# import time
# import os
# from os.path import join
#
# from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
# import numpy
# import argparse
# from Net import model
# from torch.utils.data import TensorDataset, DataLoader
# import torch.nn as nn
# import torch.optim
# from Net.data_process import Dataprocess
# import matplotlib.pyplot as plt
# import logging
#
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
# device = torch.device("cuda:3" if torch.cuda.is_available() else 'cpu')
# time_ = time.strftime('%Y%m%d%H%M%S%S')
# parser = argparse.ArgumentParser()
# parser.add_argument('-Epochs', '--epochs', help='trainging epochs', type=int,  default=130)
# parser.add_argument('-bs','--batch_size',type=int,default=64)
# parser.add_argument('-lr','--learning_rate',type=float,default=0.001)
# parser.add_argument('-ratio','--ratio',help='the ratio of train_data and valid_data',type=list,default=[0.8,0.1])
# parser.add_argument('-file_path',type=str, help='dataset file path', default='./dataset/final_dataset/final_.csv')
# parser.add_argument('-input_size', type=int, default=3)
# parser.add_argument('--hidden_size',type=int, default=4)
# parser.add_argument('-output_size',type=int, default=1)
# parser.add_argument('-num_layers',type=int, default=1)
# # parser.add_argument('--optimizer',type=str,default='cycilclr')
# parser.add_argument('-time',type=str,help='the time training',default=time_)
# parser.add_argument('-root_log_path',type=str,default='./lstm_logs')
#
# args = parser.parse_args()
# def dataloader(ratio, batch_size, file_path):
#     dataprocess = Dataprocess(file_path)
#     all_feats, gpu, cpu ,d_list= dataprocess.encode()
#
#     totalnum = len(cpu)
#     cpu = torch.tensor(cpu).reshape(totalnum, -1)
#     gpu = torch.tensor(gpu).reshape(totalnum, -1)
#     # feature = all_feats[:int(0.8*totalnum)]
#     train_dataset = TensorDataset(all_feats[:int(ratio[0] * totalnum)], gpu[:int(ratio[0] * totalnum)])
#     valid_dataset = TensorDataset(all_feats[int(ratio[0] * totalnum):int((ratio[1]+ratio[0]) * totalnum)], gpu[int(ratio[0] * totalnum):int((ratio[1]+ratio[0]) * totalnum)])
#     test_dataset =  TensorDataset(all_feats[int((ratio[1]+ratio[0]) * totalnum):],gpu[int((ratio[1]+ratio[0]) * totalnum):])
#
#     train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
#     valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#     test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=True, drop_last=True)
#     return train_loader, valid_loader, test_loader
#
# def train(net, epochs, lr,train_loader,valid_loader):
#
#     # train_loader, _ ,__= dataloader(ratio=ratio, batch_size=BATCH_SIZE, file_path=file_path)
#     optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#     loss_ls =[]
#     for j in range(epochs):
#         for i, data in enumerate(train_loader):
#             inputs, labels = data
#             inputs =inputs.to(device)
#             labels = labels.to(device)
#             outputs = net(inputs)
#             criterion = nn.MSELoss()
#             loss = criterion(outputs, labels)
#             loss_ls.append(loss)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()  #updata
#             if i % 5 == 0:
#                 RMSE = valid(net,valid_loader)
#
#                 logger.info('epoch %d-%d -----loss %f----- valid RMSE error %f' % (j + 1, i, loss, RMSE))
#                 print('epoch %d-%d-----loss %f----- valid RMSE error %f' % (j + 1,i, loss, RMSE))
#
#     torch.save(net.state_dict(),'./logs/{}/net_params.pt'.format(time_))
#     plt.figure()
#     plt.plot(range((len(loss_ls))), loss_ls)
#     # plt.savefig('./jpg/losses-{}'.format(time.strftime('%Y%m%d%H%M%S%S')))
#     plt.savefig('./logs/{}/loss.jpg' .format(time_))
#
# def valid(net,valid_loader):
#
#     MSE = 0
#     i =0
#     criterion = nn.MSELoss()
#     for data in valid_loader:
#         inputs, labels, d_list = data
#         inputs = inputs.view(inputs.size(1), inputs.size(0), -1)
#         inputs = inputs.to(device)
#         labels = labels.to(device)
#         # print(inputs, labels)
#         pred_ = net(inputs, d_list)
#         loss = criterion(pred_, labels)
#         MSE += loss
#         i += 1
#     return MSE/i
#
# def test(net,test_loader):
#     criterion = nn.MSELoss()
#     outputs, labels = None, None
#     MSE = 0
#     with torch.no_grad():
#         for k, data in enumerate(test_loader):
#             input, label,d_list = data
#             tmp = input
#             input = input.view(input.size(1), input.size(0), -1)
#             input = input.to(device)
#             label = label.to(device)
#             output = net(input, d_list)
#             loss = criterion(output,label)
#             if k == 0:
#                 outputs = output.reshape([-1,1]).cpu().numpy()
#                 labels = label.reshape([-1,1]).cpu().numpy()
#             else:
#                 outputs = np.append(outputs, output.reshape([-1,1]).cpu().numpy())
#                 labels = np.append(labels, label.reshape([-1,1]).cpu().numpy())
#
#     MAPE = (abs(outputs-labels)/labels).sum()/len(outputs)
#     print('MSE: {}'.format(mean_squared_error(labels, outputs)))
#     print('MAPE: {}'.format(MAPE))
#     print('MAE: {}'.format(mean_absolute_error(labels,outputs)))
#     print('R^2: {}'.format(r2_score(labels,outputs)))
#     print('corrcoef {}'.format(np.corrcoef(outputs, labels)))
#
#     # logger.info('r: {}'.format(SSR / SST))
#     logger.info('MSE: {}'.format(mean_squared_error(labels, outputs)))
#     logger.info('MAPE: {}'.format(MAPE))
#     logger.info('MAE: {}'.format(mean_absolute_error(labels,outputs)))
#     logger.info('R^2: {}'.format(r2_score(labels,outputs)))
#     logger.info('corrcoef {}'.format(np.corrcoef(outputs, labels)))
#
#     plt.figure()
#     plt.title('predicets & labels')
#     plt.xlabel('predict')
#     plt.ylabel('label')
#     plt.scatter(outputs, labels)
#     # plt.xlim(20,70)
#     # plt.ylim(20,70)
#     plt.plot(range(0, 30), range(0, 30), color='red')
#
#     plt.text(45,65,'relative_MSE: %.4f' %MSE, ha='center',va='top')
#     # plt.text(45, 45, 'MSE:{}'.format(MSE), ha='center', va='top')
#     plt.savefig(join(logfile,'result.jpg'))
#
#     plt.figure()
#     plt.title('relative result')
#     plt.xlabel('sample')
#     plt.ylabel('predicte/label')
#     plt.ylim(-1,3)
#     relative_value = [out / lab for out, lab in zip(outputs, labels)]
#     plt.scatter(range(len(relative_value)), relative_value)
#     plt.text(len(relative_value) * args.batch_size/2, 2.5, 'relative_MSE: %.4f' %(MAPE), ha='center', va='top')
#     plt.savefig(join(logfile, 'relative result.jpg'))
#
# # def valid(net,valid_loader):
# #     # _, valid_loader,_ = dataloader(ratio=ratio,batch_size=BATCH_SIZE,file_path=file_path)
# #     RMSE = 0
# #     i =0
# #     for data in valid_loader:
# #         inputs, labels = data
# #         inputs = inputs.to(device)
# #         labels = labels.to(device)
# #         # print(inputs, labels)
# #         pred_ = net(inputs)
# #         criterion = nn.MSELoss()
# #         loss = criterion(pred_, labels)
# #         RMSE += loss
# #         i += 1
# #     return RMSE/i
# #
# # def test(net,test_loader):
# #     # _, __, test_loader = dataloader(ratio=ratio, batch_size=BATCH_SIZE,file_path=file_path)
# #     num = 0
# #     outputs, labels = [], []
# #     RMSE = 0
# #     with torch.no_grad():
# #         for k, data in enumerate(test_loader):
# #             input, label = data
# #             input = input.to(device)
# #             label = label.to(device)
# #             output = net(input)
# #             # outputs.append(output.cpu().numpy())
# #             # labels.append(label.cpu().numpy())
# #             # print('output:',output, 'label:',label)
# #             criterion = nn.MSELoss()
# #             loss = criterion(output,label)
# #             outputs.append(output.reshape([-1,1]).cpu().numpy())
# #             labels.append(label.reshape([-1,1]).cpu().numpy())
# #             RMSE+=loss
# #             num += len(label)
# #             # tem += (output - label)**2
# #     MAPE = 0
# #     count = 0
# #     out_mean = numpy.mean(outputs)
# #     lab_mean = numpy.mean(labels)
# #     var_out, var_lab = 0., 0.
# #     SSR = 0.
# #     for i, j in zip(outputs, labels):
# #         for out, lab in zip(i, j):
# #             # print(m, n)
# #             SSR += (out[0] - out_mean) * (lab[0] - lab_mean)
# #             var_out += (out[0] - out_mean) ** 2
# #             var_lab = (lab[0] - lab_mean) ** 2
# #             logger.info('outputs:%.3f, labels:%.3f' % (out[0], lab[0]))
# #             MAPE += abs(out[0] / lab[0] - 1)
# #             count += 1
# #     MAPE /= count
# #     SST = math.sqrt(var_out * var_lab)
# #     print('r: {}'.format(SSR / SST))
# #
# #     print('MSE: {}'.format(mean_squared_error(labels[0], outputs[0])))
# #     print('MAPE: {}'.format(MAPE))
# #     print('MAE: {}'.format(mean_absolute_error(labels[0], outputs[0])))
# #     print('R^2: {}'.format(r2_score(labels[0], outputs[0])))
# #
# #     logger.info('MSE: {}'.format(mean_squared_error(labels[0], outputs[0])))
# #     logger.info('MAPE: {}'.format(MAPE))
# #     logger.info('MAE: {}'.format(mean_absolute_error(labels[0], outputs[0])))
# #     logger.info('R^2: {}'.format(r2_score(labels[0], outputs[0])))
# #     logger.info('r: {}'.format(SSR / SST))
# #
# #     # draw
# #     plt.figure()
# #     plt.title('predicets & labels')
# #     plt.xlabel('predict')
# #     plt.ylabel('label')
# #     plt.scatter(outputs,labels)
# #     plt.plot(range(0,5),range(0,5),color='red')
# #     plt.savefig('./logs/{}/result.jpg' .format(time_))
#
# EPOCHS = 100
# BATCH_SIZE = 64
# lr =0.001
# ratio = [0.8, 0.1]
# file_path = './dataset/latency_dataset_tr_1w_resolution.csv'
# para = [128,64,64,64,16,1]
#
# time_ = time.strftime('%Y%m%d%H%M%S%S')
# if not os.path.exists('./logs/{}'.format(time_)):
#     os.mkdir('./logs/{}'.format(time_))
# logname = './logs/{}/log.txt'.format(time_)
# logging.basicConfig(level=logging.INFO, filename=logname)
# logger = logging.getLogger('main.py')
#
# def main():
#     net = model.FC_net(para)
#     if torch.cuda.device_count()>1:
#         net = nn.DataParallel(net, device_ids=[3])
#     net = model.FC_net(para).to(device)
#     train_loader, valid_loader, test_loader = dataloader(ratio=args.ratio, batch_size=args.batch_size,
#                                                          file_path=args.file_path)
#
#     logger.info(time_)
#     logger.info('epochs{}-lr{}-batch_size{}-ratio{}'.format(EPOCHS,lr,BATCH_SIZE,ratio))
#     logger.info(para)
#     logger.info(file_path)
#     logger.info('encode-spec2feats')
#     train(net, epochs=EPOCHS, lr=lr,)
#     # valid(net)
#     test(net)
#
# if __name__ == '__main__':
#     main()

import math
import time
from os.path import join
from Net import model
import numpy as np

from LSTM import lstm_model
from Net.data_process import *
import argparse
import torch
# print(torch.__version__)
import torch.nn as nn
import logging
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

time_ = time.strftime('%Y%m%d%H%M%S%S')
parser = argparse.ArgumentParser()
parser.add_argument('-Epochs', '--epochs', help='trainging epochs', type=int, default=100)
parser.add_argument('-bs', '--batch_size', type=int, default=64)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-ratio', '--ratio', help='the ratio of train_data and valid_data', type=list, default=[0.8, 0.1])
parser.add_argument('-file_path', type=str, help='dataset file path', default='./dataset/final_dataset/final_.csv')
parser.add_argument('-input_size', type=int, default=3)
parser.add_argument('--hidden_size', type=int, default=4)
parser.add_argument('-output_size', type=int, default=1)
parser.add_argument('-num_layers', type=int, default=1)
# parser.add_argument('--optimizer',type=str,default='cycilclr')
parser.add_argument('-time', type=str, help='the time training', default=time_)
parser.add_argument('-root_log_path', type=str, default='./fc_logs')

args = parser.parse_args()
logfile = args.root_log_path + '/{}'.format(time_)

if not os.path.exists(logfile):
    os.mkdir(logfile)

logging.basicConfig(level=logging.INFO, filename=join(logfile, 'log.txt'))
logger = logging.getLogger('main.py')
logger.info(args)


def train(net, epochs, lr, train_loader, valid_loader):
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_ls = []
    for j in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data

            # inputs = inputs.view(inputs.size(1), inputs.size(0), -1)
            inputs = inputs.to(device,dtype=torch.float)
            labels = labels.to(device,dtype=torch.float)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss_ls.append(loss)
            # scheduler.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()  #updata
            if i % 20 == 0:
                MSE = valid(net, valid_loader=valid_loader)
                logger.info('epoch %d-%d -----loss %f----- valid RMSE error %f' % (j + 1, i, loss, MSE))
                print('epoch %d-%d-----loss %f----- valid RMSE error %f' % (j + 1, i, loss, MSE))

    torch.save(net.state_dict(), './fc_logs/{}/net_params.pt'.format(time_))
    plt.figure()
    plt.plot(range((len(loss_ls))), loss_ls)
    plt.ylim(0, 50)
    plt.savefig('./fc_logs/{}/loss.jpg'.format(time_))


def valid(net, valid_loader):
    MSE = 0
    i = 0
    criterion = nn.MSELoss()
    for data in valid_loader:
        inputs, labels = data
        # inputs = inputs.view(inputs.size(1), inputs.size(0), -1)
        inputs = inputs.to(device,dtype=torch.float)
        labels = labels.to(device,dtype=torch.float)
        # print(inputs, labels)
        pred_ = net(inputs)
        loss = criterion(pred_, labels)
        MSE += loss
        i += 1
    return MSE / i


def test(net, test_loader):
    criterion = nn.MSELoss()
    outputs, labels = None, None
    MSE = 0
    with torch.no_grad():
        for k, data in enumerate(test_loader):
            input, label = data
            tmp = input
            # input = input.view(input.size(1), input.size(0), -1)
            input = input.to(device,dtype=torch.float)
            label = label.to(device,dtype=torch.float)
            output = net(input)
            # loss = criterion(output, label)
            if k == 0:
                outputs = output.reshape([-1, 1]).cpu().numpy()
                labels = label.reshape([-1, 1]).cpu().numpy()
            else:
                outputs = np.append(outputs, output.reshape([-1, 1]).cpu().numpy())
                labels = np.append(labels, label.reshape([-1, 1]).cpu().numpy())

    MAPE = (abs(outputs - labels) / labels).sum() / len(outputs)
    print('MSE: {}'.format(mean_squared_error(labels, outputs)))
    print('MAPE: {}'.format(MAPE))
    print('MAE: {}'.format(mean_absolute_error(labels, outputs)))
    print('R^2: {}'.format(r2_score(labels, outputs)))
    print('corrcoef {}'.format(np.corrcoef(outputs, labels)))

    # logger.info('r: {}'.format(SSR / SST))
    logger.info('MSE: {}'.format(mean_squared_error(labels, outputs)))
    logger.info('MAPE: {}'.format(MAPE))
    logger.info('MAE: {}'.format(mean_absolute_error(labels, outputs)))
    logger.info('R^2: {}'.format(r2_score(labels, outputs)))
    logger.info('corrcoef {}'.format(np.corrcoef(outputs, labels)))

    plt.figure()
    plt.title('predicets & labels')
    plt.xlabel('predict')
    plt.ylabel('label')
    plt.scatter(outputs, labels)
    # plt.xlim(20,70)
    # plt.ylim(20,70)
    plt.plot(range(0, 30), range(0, 30), color='red')

    plt.text(45, 65, 'relative_MSE: %.4f' % MSE, ha='center', va='top')
    # plt.text(45, 45, 'MSE:{}'.format(MSE), ha='center', va='top')
    plt.savefig(join(logfile, 'result.jpg'))

    plt.figure()
    plt.title('relative result')
    plt.xlabel('sample')
    plt.ylabel('predicte/label')
    plt.ylim(-1, 3)
    relative_value = [out / lab for out, lab in zip(outputs, labels)]
    plt.scatter(range(len(relative_value)), relative_value)
    plt.text(len(relative_value) * args.batch_size / 2, 2.5, 'relative_MSE: %.4f' % (MAPE), ha='center', va='top')
    plt.savefig(join(logfile, 'relative result.jpg'))


para = [1480, 64, 64, 64, 16, 1]


def main():


    net_FC = model.FC_net(para).to(device)
    fc_train_loader, fc_valid_loader, fc_test_loader = FC_dataloader(ratio=args.ratio, batch_size=args.batch_size,
                                                                  file_path=args.file_path, lstm=False)
    train(net_FC, epochs=args.epochs, lr=args.learning_rate, train_loader=fc_train_loader, valid_loader=fc_valid_loader)
    test(net_FC, fc_test_loader)
    logger.info(args)


if __name__ == '__main__':
    main()
