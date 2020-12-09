import math
import time
from os.path import join
# from .Net import model
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
parser.add_argument('-Epochs', '--epochs', help='trainging epochs', type=int,  default=150)
parser.add_argument('-bs','--batch_size',type=int,default=64)
parser.add_argument('-lr','--learning_rate',type=float,default=0.0003)
parser.add_argument('-ratio','--ratio',help='the ratio of train_data and valid_data',type=list,default=[0.8,0.1])
parser.add_argument('-file_path',type=str, help='dataset file path', default='./dataset/final_dataset/final_.csv')
parser.add_argument('-input_size', type=int, default=37)
parser.add_argument('--hidden_size',type=int, default=256)
parser.add_argument('-output_size',type=int, default=1)
parser.add_argument('-num_layers',type=int, default=1)
# parser.add_argument('--optimizer',type=str,default='cycilclr')
parser.add_argument('-time',type=str,help='the time training',default=time_)
parser.add_argument('-root_log_path',type=str,default='./lstm_logs')

args = parser.parse_args()
logfile = args.root_log_path+'/{}'.format(time_)

if not os.path.exists(logfile):
    os.mkdir(logfile)

logging.basicConfig(level=logging.INFO, filename=join(logfile,'log.txt'))
logger = logging.getLogger('main.py')
logger.info(args)

def train(net, epochs, lr,train_loader,valid_loader):

    optimizer = torch.optim.SGD(net.parameters(),lr=lr)
    criterion = nn.MSELoss()
    loss_ls =[]
    for j in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, labels, d_list = data

            inputs = inputs.view(inputs.size(1),inputs.size(0),-1)
            inputs =inputs.to(device,dtype=torch.float)
            labels = labels.to(device,dtype=torch.float)
            outputs = net(inputs, d_list) ## _x is input, size (seq_len, batch, input_size)
            loss = criterion(outputs, labels)
            loss_ls.append(loss)
            # scheduler.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()  #updata
            if i % 20 == 0:
                MSE = valid(net,valid_loader=valid_loader)
                logger.info('epoch %d-%d -----loss %f----- valid RMSE error %f' % (j + 1, i, loss, MSE))
                print('epoch %d-%d-----loss %f----- valid RMSE error %f' % (j + 1,i, loss, MSE))

    torch.save(net.state_dict(), './lstm_logs/{}/net_params.pt'.format(time_))
    plt.figure()
    plt.plot(range((len(loss_ls))), loss_ls)
    plt.ylim(0,50)
    plt.savefig('./lstm_logs/{}/loss.jpg'.format(time_))
def valid(net,valid_loader):

    MSE = 0
    i =0
    criterion = nn.MSELoss()
    for data in valid_loader:
        inputs, labels, d_list = data
        inputs = inputs.view(inputs.size(1), inputs.size(0), -1)
        inputs = inputs.to(device,dtype =torch.float)
        labels = labels.to(device,dtype=torch.float)
        # print(inputs, labels)
        pred_ = net(inputs, d_list)
        loss = criterion(pred_, labels)
        MSE += loss
        i += 1
    return MSE/i

def test(net,test_loader):
    criterion = nn.MSELoss()
    outputs, labels = None, None
    MSE = 0
    with torch.no_grad():
        for k, data in enumerate(test_loader):
            input, label,d_list = data
            tmp = input
            input = input.view(input.size(1), input.size(0), -1)
            input = input.to(device,dtype=torch.float)
            label = label.to(device,dtype=torch.float)
            output = net(input, d_list)
            loss = criterion(output,label)
            if k == 0:
                outputs = output.reshape([-1,1]).cpu().numpy()
                labels = label.reshape([-1,1]).cpu().numpy()
            else:
                outputs = np.append(outputs, output.reshape([-1,1]).cpu().numpy())
                labels = np.append(labels, label.reshape([-1,1]).cpu().numpy())

    MAPE = (abs(outputs-labels)/labels).sum()/len(outputs)
    print('MSE: {}'.format(mean_squared_error(labels, outputs)))
    print('MAPE: {}'.format(MAPE))
    print('MAE: {}'.format(mean_absolute_error(labels,outputs)))
    print('R^2: {}'.format(r2_score(labels,outputs)))
    print('corrcoef {}'.format(np.corrcoef(outputs, labels)))

    # logger.info('r: {}'.format(SSR / SST))
    logger.info('MSE: {}'.format(mean_squared_error(labels, outputs)))
    logger.info('MAPE: {}'.format(MAPE))
    logger.info('MAE: {}'.format(mean_absolute_error(labels,outputs)))
    logger.info('R^2: {}'.format(r2_score(labels,outputs)))
    logger.info('corrcoef {}'.format(np.corrcoef(outputs, labels)))

    plt.figure()
    plt.title('predicets & labels')
    plt.xlabel('predict')
    plt.ylabel('label')
    plt.scatter(outputs, labels)
    # plt.xlim(20,70)
    # plt.ylim(20,70)
    plt.plot(range(0, 30), range(0, 30), color='red')

    plt.text(45,65,'relative_MSE: %.4f' %MSE, ha='center',va='top')
    # plt.text(45, 45, 'MSE:{}'.format(MSE), ha='center', va='top')
    plt.savefig(join(logfile,'result.jpg'))

    plt.figure()
    plt.title('relative result')
    plt.xlabel('sample')
    plt.ylabel('predicte/label')
    plt.ylim(-1,3)
    relative_value = [out / lab for out, lab in zip(outputs, labels)]
    plt.scatter(range(len(relative_value)), relative_value)
    plt.text(len(relative_value) * args.batch_size/2, 2.5, 'relative_MSE: %.4f' %(MAPE), ha='center', va='top')
    plt.savefig(join(logfile, 'relative result.jpg'))

para = [128,64,64,64,16,1]
def main():
    net = lstm_model.LstmRNN(input_size=args.input_size, hidden_size=args.hidden_size,output_size=args.output_size,num_layers=args.num_layers)
    if torch.cuda.device_count()>1:
        net = nn.DataParallel(net, device_ids=[1,2,3])
    net = lstm_model.LstmRNN(input_size=args.input_size, hidden_size=args.hidden_size,output_size=args.output_size,num_layers=args.num_layers).to(device)

    train_loader, valid_loader, test_loader = dataloader(ratio=args.ratio, batch_size=args.batch_size, file_path=args.file_path,lstm=True)
    train(net, epochs=args.epochs, lr=args.learning_rate,train_loader=train_loader,valid_loader=valid_loader)
    test(net,test_loader)
    logger.info(args)

    # net_FC = model.FC_net(para).to(device)
    # fc_train_loader, fc_valid_loader, fc_test_loader = dataloader(ratio=args.ratio, batch_size=args.batch_size, file_path=args.file_path,lstm=False)
    # train(net_FC, epochs=args.epochs, lr=args.learning_rate, train_loader=fc_train_loader, valid_loader=fc_valid_loader)
    # test(net_FC, fc_test_loader)
    # logger.info(args)


if __name__ == '__main__':
    main()



