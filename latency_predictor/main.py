import time
import os
from Net import model
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim
from Net.data_process import Dataprocess
import matplotlib.pyplot as plt
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
def dataloader(ratio, batch_size, file_path):
    dataprocess = Dataprocess(file_path)
    all_feats, gpu, cpu = dataprocess.encode()

    totalnum = len(cpu)
    cpu = torch.tensor(cpu).reshape(totalnum, -1)
    gpu = torch.tensor(gpu).reshape(totalnum, -1)
    # feature = all_feats[:int(0.8*totalnum)]
    train_dataset = TensorDataset(all_feats[:int(ratio[0] * totalnum)], cpu[:int(ratio[0] * totalnum)])
    valid_dataset = TensorDataset(all_feats[int(ratio[0] * totalnum):int((ratio[1]+ratio[0]) * totalnum)], cpu[int(ratio[0] * totalnum):int((ratio[1]+ratio[0]) * totalnum)])
    test_dataset =  TensorDataset(all_feats[int((ratio[1]+ratio[0]) * totalnum):],cpu[int((ratio[1]+ratio[0]) * totalnum):])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=True, drop_last=True)
    return train_loader, valid_loader, test_loader

def train(net, epochs, lr):

    train_loader, _ ,__= dataloader(ratio=ratio, batch_size=BATCH_SIZE, file_path=file_path)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_ls =[]
    for j in range(epochs):
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs =inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            criterion = nn.MSELoss()
            loss = criterion(outputs, labels)
            loss_ls.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  #updata
            if i % 5 == 0:
                RMSE = valid(net)

                logger.info('epoch %d-%d -----loss %f----- valid RMSE error %f' % (j + 1, i, loss, RMSE))
                print('epoch %d-%d-----loss %f----- valid RMSE error %f' % (j + 1,i, loss, RMSE))

    torch.save(net.state_dict(),'./logs/{}/net_params.pt'.format(time_))

    plt.figure()
    plt.plot(range((len(loss_ls))), loss_ls)
    # plt.savefig('./jpg/losses-{}'.format(time.strftime('%Y%m%d%H%M%S%S')))
    plt.savefig('./logs/{}/loss.jpg' .format(time_))


def valid(net):
    _, valid_loader,_ = dataloader(ratio=ratio,batch_size=BATCH_SIZE,file_path=file_path)
    RMSE = 0
    valid_num = 0
    for i, data in enumerate(valid_loader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # print(inputs, labels)
        valid_num += len(labels)
        # net = latency_net(n_feature=128, n_hid0=400, n_hid1=400, n_hid2=400, n_hid3=400, output=1)
        pred_ = net(inputs)
        criterion = nn.MSELoss()
        loss = criterion(pred_, labels)
        RMSE += loss
        # RMSE += (pred_ - labels)**2
    # all =0
    # for i in RMSE:
    #     all+=i
    return RMSE/(i+1)

def test(net):
    _, __, test_loader = dataloader(ratio=ratio, batch_size=BATCH_SIZE,file_path=file_path)
    num = 0
    outputs, labels = [], []
    RMSE = 0
    with torch.no_grad():
        for k, data in enumerate(test_loader):
            input, label = data
            input = input.to(device)
            label = label.to(device)
            output = net(input)
            # outputs.append(output.cpu().numpy())
            # labels.append(label.cpu().numpy())
            # print('output:',output, 'label:',label)
            criterion = nn.MSELoss()
            loss = criterion(output,label)
            outputs.append(output.reshape([-1,1]).cpu().numpy())
            labels.append(label.reshape([-1,1]).cpu().numpy())
            RMSE+=loss
            num += len(label)
            # tem += (output - label)**2
    for i, j in zip(outputs,labels):
        for m, n in zip(i, j):
            print(m, n)
    logger.info('outputs:{}'.format(outputs))
    logger.info('labels{}'.format( labels))
    logger.info('RMSE %f' % (RMSE/(k+1)))
    print('RMSE %f' % (RMSE/(k+1)))
    # draw
    plt.figure()
    plt.title('predicets & labels')
    plt.xlabel('predict')
    plt.ylabel('label')
    plt.scatter(outputs,labels)
    plt.plot(range(20,50),range(20,50),color='red')
    plt.savefig('./logs/{}/result.jpg' .format(time_))

EPOCHS = 150
BATCH_SIZE = 1000
lr =0.001
ratio = [0.8, 0.1]
file_path = './dataset/latency_dataset_2_50000.csv'
para = [128,400,400,400,400,1]

time_ = time.strftime('%Y%m%d%H%M%S%S')
if not os.path.exists('./logs/{}'.format(time_)):
    os.mkdir('./logs/{}'.format(time_))
logname = './logs/{}/log.txt'.format(time_)
logging.basicConfig(level=logging.INFO, filename=logname)
logger = logging.getLogger('main.py')

def main():
    net = model.latency_net(para)
    if torch.cuda.device_count()>1:
        net = nn.DataParallel(net, device_ids=[0,1,2,3])
    net = model.latency_net(para).to(device)
    logger.info(time_)
    logger.info('epochs{}-lr{}-batch_size{}-ratio{}'.format(EPOCHS,lr,BATCH_SIZE,ratio))
    logger.info(para)
    logger.info(file_path)
    logger.info('encode-spec2feats_v2')
    train(net, epochs=EPOCHS, lr=lr)
    valid(net)
    test(net)

if __name__ == '__main__':
    main()
