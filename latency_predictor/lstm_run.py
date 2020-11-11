import time
from os.path import join
from LSTM import lstm_model, variable_lstm
from Net.data_process import *
import argparse
import torch
print(torch.__version__)
import torch.nn as nn
import logging
import os
from matplotlib import pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

time_ = time.strftime('%Y%m%d%H%M%S%S')
parser = argparse.ArgumentParser()
parser.add_argument('-Epochs', '--epochs', help='trainging epochs', type=int,  default=130)
parser.add_argument('-bs','--batch_size',type=int,default=256)
parser.add_argument('-lr','--learning_rate',type=float,default=0.001)
parser.add_argument('-ratio','--ratio',help='the ratio of train_data and valid_data',type=list,default=[0.8,0.1])
parser.add_argument('-file_path',type=str, help='dataset file path', default='./dataset/latency_dataset_2_50000_temp.csv')
parser.add_argument('-input_size', type=int, default=3)
parser.add_argument('--hidden_size',type=int, default=128)
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

    # optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999),weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer,base_lr=lr,max_lr=lr*5,step_size_up=800,
    #                                               step_size_down=120,cycle_momentum=False)
    optimizer = torch.optim.SGD(net.parameters(),lr=lr)
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
            # scheduler.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()  #updata
            if i % 20 == 0:
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
    MSE = 0
    with torch.no_grad():
        for k, data in enumerate(test_loader):
            input, label,d_list = data
            tmp = input
            input = input.view(input.size(1), input.size(0), -1)
            input = input.to(device)
            label = label.to(device)
            output = net(input, d_list)
            for i, bia in enumerate(output-label):
                if abs(bia) > 10:
                    print(tmp[i,:,:])
                    print('output{}, label{}'.format(output[i], label[i]))

            loss = criterion(output,label)
            outputs.append(output.reshape([-1,1]).cpu().numpy())
            labels.append(label.reshape([-1,1]).cpu().numpy())

            MSE+=loss
            num += len(label)
    logger.info('MSE %f' % (MSE / (k + 1)))
    print('MSE %f' % (MSE / (k + 1)))

    relative_MSE = 0
    count = 0
    for i, j in zip(outputs,labels):
        for m, n in zip(i, j):
            # print(m, n)
            logger.info('outputs:{}, labels:{}'.format(m, n))
            relative_MSE += (m[0]/n[0] - 1)**2
            count += 1
    relative_MSE /= count
    print('relative_MSE:{}'.format(relative_MSE))
    plt.figure()
    plt.title('predicets & labels')
    plt.xlabel('predict')
    plt.ylabel('label')
    plt.scatter(outputs, labels)
    plt.xlim(20,70)
    plt.ylim(20,70)
    plt.plot(range(20, 70), range(20, 70), color='red')

    plt.text(45,65,'relative_MSE: %.4f' %MSE, ha='center',va='top')
    # plt.text(45, 45, 'MSE:{}'.format(MSE), ha='center', va='top')
    plt.savefig(join(logfile,'result.jpg'))

    plt.figure()
    plt.title('relative result')
    plt.xlabel('sample')
    plt.ylabel('predicte/label')
    plt.ylim(-1,3)
    relative_value = [out / lab for out, lab in zip(outputs, labels)]
    plt.scatter(range(len(relative_value) * args.batch_size), relative_value)
    plt.text(len(relative_value) * args.batch_size/2, 2.5, 'relative_MSE: %.4f' %(relative_MSE), ha='center', va='top')
    plt.savefig(join(logfile, 'relative result.jpg'))

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



