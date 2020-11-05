import time
from os.path import join

from LSTM import lstm_model
from Net.data_process import *
import argparse
import torch
import torch.nn as nn
import logging
import os
from matplotlib import pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

time_ = time.strftime('%Y%m%d%H%M%S%S')
parser = argparse.ArgumentParser()
parser.add_argument('-Epochs', '--epochs', help='trainging epochs', type=int,  default=150)
parser.add_argument('-bs','--batch_size',type=int,default=256)
parser.add_argument('-lr','--learning_rate',type=float,default=0.001)
parser.add_argument('-ratio','--ratio',help='the ratio of train_data and valid_data',type=list,default=[0.8,0.1])
parser.add_argument('-file_path',type=str, help='dataset file path', default='./dataset/latency_dataset_2_50000_temp.csv')
parser.add_argument('-input_size', type=int, default=3)
parser.add_argument('-hidden_size',type=int, default=128)
parser.add_argument('-output_size',type=int, default=1)
parser.add_argument('-num_layers',type=int, default=1)
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

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_ls =[]
    for j in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, labels, d_list = data
            inputs = inputs.view(inputs.size(1),inputs.size(0),-1)
            inputs =inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs, d_list)
            loss = criterion(outputs, labels)
            loss_ls.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  #updata
            if i % 5 == 0:
                RMSE = valid(net,valid_loader=valid_loader)
                logger.info('epoch %d-%d -----loss %f----- valid RMSE error %f' % (j + 1, i, loss, RMSE))
                print('epoch %d-%d-----loss %f----- valid RMSE error %f' % (j + 1,i, loss, RMSE))

    torch.save(net.state_dict(), './lstm_logs/{}/net_params.pt'.format(time_))
    plt.figure()
    plt.plot(range((len(loss_ls))), loss_ls)
    plt.ylim(10,100)
    plt.savefig('./lstm_logs/{}/loss.jpg'.format(time_))
def valid(net,valid_loader):

    RMSE = 0
    i =0
    criterion = nn.MSELoss()
    for data in valid_loader:
        inputs, labels, d_list = data
        inputs = inputs.view(inputs.size(1), inputs.size(0), -1)
        inputs = inputs.to(device)
        labels = labels.to(device)
        # print(inputs, labels)
        pred_ = net(inputs, d_list)
        loss = criterion(pred_, labels)
        RMSE += loss
        i += 1
    return RMSE/i

def test(net,test_loader):
    criterion = nn.MSELoss()
    num = 0
    outputs, labels = [], []
    RMSE = 0
    with torch.no_grad():
        for k, data in enumerate(test_loader):
            input, label,d_list = data
            input = input.view(input.size(1), input.size(0), -1)
            input = input.to(device)
            label = label.to(device)
            output = net(input, d_list)
            loss = criterion(output,label)
            outputs.append(output.reshape([-1,1]).cpu().numpy())
            labels.append(label.reshape([-1,1]).cpu().numpy())
            RMSE+=loss
            num += len(label)
    logger.info('RMSE %f' % (RMSE / (k + 1)))
    print('RMSE %f' % (RMSE / (k + 1)))

    for i, j in zip(outputs,labels):
        for m, n in zip(i, j):
            print(m, n)
            logger.info('outputs:{}, labels:{}'.format(m, n))

    plt.figure()
    plt.title('predicets & labels')
    plt.xlabel('predict')
    plt.ylabel('label')
    plt.scatter(outputs, labels)
    plt.xlim(10,60)
    plt.ylim(10,60)
    plt.plot(range(20, 50), range(20, 50), color='red')
    plt.savefig(join(logfile,'result.jpg'.format(time_)))

def main():
    net = lstm_model.LstmRNN(input_size=args.input_size, hidden_size=args.hidden_size,output_size=args.output_size,num_layers=args.num_layers)
    if torch.cuda.device_count()>1:
        net = nn.DataParallel(net, device_ids=[1,2,3])
    net = lstm_model.LstmRNN(input_size=args.input_size, hidden_size=args.hidden_size,output_size=args.output_size,num_layers=args.num_layers).to(device)
    train_loader, valid_loader, test_loader = dataloader(ratio=args.ratio, batch_size=args.batch_size, file_path=args.file_path)
    train(net, epochs=args.epochs, lr=args.learning_rate,train_loader=train_loader,valid_loader=valid_loader)
    test(net,test_loader)

if __name__ == '__main__':
    main()



