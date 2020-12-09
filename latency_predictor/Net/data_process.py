import pandas as pd
import torch
import copy
import os
import random

from numpy import sort
from torch.utils.data import TensorDataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def construct_maps(keys):
    d = dict()
    keys = sorted(list(set(keys)))
    for k in keys:
        if k not in d:
            d[k] = len(list(d.keys()))
    return d

ks_map = construct_maps(keys=(3, 5, 7))  # {3: 0, 5: 1, 7: 2}
ex_map = construct_maps(keys=(3, 4, 6))
dp_map = construct_maps(keys=(2, 3, 4))
reso_map = construct_maps(keys=(160, 320, 512, 768, 1024))
channel_map = construct_maps(keys=(24,32,48,96,136,192,232,272,304,384,576))
stride_map = construct_maps(keys=(1,2))

class Dataprocess():
    def __init__(self,file_path):
        self.file_path = file_path

    def encode(self,lstm=True):
        all_feats = []
        gpu, cpu =[], []
        d_lists = []
        df = pd.read_csv(self.file_path, sep='\t')
        population = df.values
        random.shuffle(population)
        for sample in population:
            sample = eval(sample[0])
            para = sample[0]
            ks_list = copy.deepcopy(para['ks'])
            ex_list = copy.deepcopy(para['e'])
            d_list = copy.deepcopy(para['d'])
            d_lists.append(d_list)
            r = copy.deepcopy(para['r'])[0]
            if lstm:
                feats = self.spec2feats_lstm_all(ks_list, ex_list, d_list, r)
            else:
                feats = self.spec2feats_fc_all(ks_list, ex_list, d_list, r)
            gpu.append(sample[1])
            cpu.append(sample[2])
            all_feats.append(feats)
            # print(all_feats)
        all_feats = torch.cat(all_feats, 0) #row
        # pred = self.model(all_feats).cpu()
        return all_feats, gpu, cpu,d_lists


    @staticmethod
    def spec2feats(ks_list, ex_list, d_list, r):
        # This function converts a network config to a feature vector (128-D).
        start = 0
        end = 4
        for d in d_list:
            for j in range(start + d, end):
                ks_list[j] = 0
                ex_list[j] = 0
            start += 4
            end += 4

        # convert to onehot
        ks_onehot = [0 for _ in range(120)]
        ex_onehot = [0 for _ in range(120)]
        r_onehot = [0 for _ in range(16)]

        for i in range(40):
            start = i * 3
            if ks_list[i] != 0:
                ks_onehot[start + ks_map[ks_list[i]]] = 1
            if ex_list[i] != 0:
                ex_onehot[start + ex_map[ex_list[i]]] = 1

        r_onehot[(r - 112) // 19] = 1
        return torch.Tensor(ks_onehot + ex_onehot + r_onehot)

    @staticmethod
    def spec2feats_lstm(ks_list, ex_list, d_list, r):
        tmp = []
        for ks, ex in zip(ks_list, ex_list):
            tmp.append(torch.tensor([ks, ex, r/100]))
        ten = torch.cat(tmp, -1)
        ten = torch.unsqueeze(ten,dim=0).view(-1, 20, 3)
        return ten

    @staticmethod
    def spec2feats_lstm_with_0(ks_list, ex_list, d_list, r):
        start = 0
        end = 4
        for d in d_list:
            for j in range(start + d, end):
                ks_list[j] = 0
                ex_list[j] = 0
            start += 4
            end += 4
        tmp = []

        for ks, ex in zip(ks_list, ex_list):
            tmp.append(torch.tensor([ks, ex, r / 100]))
        ten = torch.cat(tmp, -1)
        ten = torch.unsqueeze(ten, dim=0).view(-1, 40, 3)
        return ten

    @staticmethod
    def spec2feats_var_lstm(ks_list, ex_list, d_list, r):
        tmp = []
        start = 0
        end = 4
        for d in d_list:
            for j in range(start + d, end):
                ks_list[j] = 0
                ex_list[j] = 0
            start += 4
            end += 4

        for ks, ex in zip(ks_list, ex_list):
            if ks != 0 and ex!= 0:
                tmp.append(torch.tensor([ks, ex, r / 100]))
        ten = torch.cat(tmp, -1)
        ten = torch.unsqueeze(ten, dim=0).view(-1, 20, 3)
        return ten

    '''
    se_stages = [False, False, True, False, True, True, False, True, False, True, True]
    stride_stages = [1, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2]
    [3,4,5,4]
    channel 24,    32,48,96,136,192,    232,272,304,384,576,   1152,1536,
    '''

    @staticmethod
    def spec2feats_lstm_all(ks_list, ex_list, d_list, r):
        channels = [24,  32,48,96,136,192,    232,272,304,384,576]
        strides = [2, 2, 2, 1, 2,   2, 2, 2, 1, 2]
        # se_stages = [False, True, False, True, True,      False, True, False, True, True]
        se= [0,1,0,1,1,  0,1,0,1,1]
        start = 0
        end = 4
        detail_stide = []
        for d in d_list:
            for j in range(start + d, end):
                ks_list[j] = 0
                ex_list[j] = 0
            start += 4
            end += 4
        tmp = []
        for i,  (ks, ex) in enumerate(zip(ks_list, ex_list)):
            ks_ten, ex_ten = [0, 0, 0], [0, 0, 0]
            reso = [0 for _ in range(5)]
            inchannel = [0 for _ in range(11)]
            outchannel = [0 for _ in range(11)]
            stride = [0 for _ in range(2)]
            shortcut = 0
            if ks != 0:
                if i==0:
                    in_channel=24
                    out_channel=32
                    ks_ten[ks_map[ks]],ex_ten[ex_map[ex]],reso[reso_map[r]],inchannel[channel_map[in_channel]],outchannel[channel_map[out_channel]]=1,1,1,1,1
                    tmp.append(torch.cat([torch.tensor(a) for a in [ks_ten, ex_ten, reso,inchannel, outchannel,[0,1],[0,0]]]))
                else:
                    if i % 4 == 0 :
                        in_channel = channels[i//4]
                        out_channel = channels[i//4+1]
                        now_stride = strides[i//4]
                    else:
                        in_channel=channels[i//4+1]
                        out_channel=channels[i//4+1]
                        now_stride=1
                        shortcut = 1
                    ks_ten[ks_map[ks]], ex_ten[ex_map[ex]], reso[reso_map[r]], inchannel[channel_map[in_channel]], \
                    outchannel[channel_map[out_channel]] = 1, 1, 1, 1, 1
                    stride[stride_map[now_stride]]=1
                    tmp.append(torch.cat([torch.tensor(a) for a in [ks_ten, ex_ten,reso, inchannel, outchannel, stride,[se[i//4],shortcut]]]))
                    # tmp.append(torch.tensor([ks_ten[ks_map[ks]], ex_ten[ex_map[ex]], reso[reso_map[r]],channel[channel_map[in_channel]],
                    #                              channel[channel_map[out_channel]],stride[stride_map[now_stride]],se[i//4],shortcut]))
            else:
                tmp.append(torch.tensor([0]*37))
        ten = torch.cat(tmp, -1)
        ten = torch.unsqueeze(ten, dim=0).view(-1, 40, 37)
        return ten


    @staticmethod
    def spec2feats_v2(ks_list, ex_list, d_list, r):
        # This function converts a network config to a feature vector (128-D).
        start = 0
        end = 4
        for d in d_list:
            for j in range(start + d, end):
                ks_list[j] = 0
                ex_list[j] = 0
            start += 4
            end += 4

        # convert to onehot
        ks_onehot = [0 for _ in range(240)]
        # ex_onehot = [0 for _ in range(60)]
        r_onehot = [0 for _ in range(8)] # self.resolutions = [160, 320, 512, 768, 1024]

        for i in range(40):
            start = i * 6
            if ks_list[i] != 0:
                ks_onehot[start + ks_map[ks_list[i]]] = 1
                ks_onehot[start + 3 + ex_map[ex_list[i]]] = 1
        r_onehot[r // 146] = 1
        return torch.Tensor(ks_onehot + r_onehot)

    @staticmethod
    def spec2feats_fc_all(ks_list, ex_list, d_list, r):
        channels = [24, 32, 48, 96, 136, 192, 232, 272, 304, 384, 576]
        strides = [2, 2, 2, 1, 2, 2, 2, 2, 1, 2]
        # se_stages = [False, True, False, True, True,      False, True, False, True, True]
        se = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
        start = 0
        end = 4
        detail_stide = []
        for d in d_list:
            for j in range(start + d, end):
                ks_list[j] = 0
                ex_list[j] = 0
            start += 4
            end += 4
        tmp = []
        for i, (ks, ex) in enumerate(zip(ks_list, ex_list)):
            ks_ten, ex_ten = [0, 0, 0], [0, 0, 0]
            reso = [0 for _ in range(5)]
            inchannel = [0 for _ in range(11)]
            outchannel = [0 for _ in range(11)]
            stride = [0 for _ in range(2)]
            shortcut = 0
            if ks != 0:
                if i == 0:
                    in_channel = 24
                    out_channel = 32
                    ks_ten[ks_map[ks]], ex_ten[ex_map[ex]], reso[reso_map[r]], inchannel[channel_map[in_channel]], \
                    outchannel[channel_map[out_channel]] = 1, 1, 1, 1, 1
                    tmp.append(torch.cat(
                        [torch.tensor(a) for a in [ks_ten, ex_ten, reso, inchannel, outchannel, [0, 1], [0, 0]]]))
                else:
                    if i % 4 == 0:
                        in_channel = channels[i // 4]
                        out_channel = channels[i // 4 + 1]
                        now_stride = strides[i // 4]
                    else:
                        in_channel = channels[i // 4 + 1]
                        out_channel = channels[i // 4 + 1]
                        now_stride = 1
                        shortcut = 1
                    ks_ten[ks_map[ks]], ex_ten[ex_map[ex]], reso[reso_map[r]], inchannel[channel_map[in_channel]], \
                    outchannel[channel_map[out_channel]] = 1, 1, 1, 1, 1
                    stride[stride_map[now_stride]] = 1
                    tmp.append(torch.cat([torch.tensor(a) for a in [ks_ten, ex_ten, reso, inchannel, outchannel, stride,
                                                                    [se[i // 4], shortcut]]]))
                    # tmp.append(torch.tensor([ks_ten[ks_map[ks]], ex_ten[ex_map[ex]], reso[reso_map[r]],channel[channel_map[in_channel]],
                    #                              channel[channel_map[out_channel]],stride[stride_map[now_stride]],se[i//4],shortcut]))
            else:
                tmp.append(torch.tensor([0] * 37))
        ten = torch.cat(tmp, -1)
        return ten

def dataloader(ratio, batch_size, file_path,lstm=True):
    dataprocess = Dataprocess(file_path)
    all_feats, gpu, cpu, d_list= dataprocess.encode(lstm=lstm)
    totalnum = len(cpu)
    cpu = torch.tensor(cpu).reshape(totalnum, -1)
    gpu = torch.tensor(gpu).reshape(totalnum, -1)
    d_list =torch.tensor(d_list).reshape(totalnum, -1)
    # feature = all_feats[:int(0.8*totalnum)]
    train_dataset = TensorDataset(all_feats[:int(ratio[0] * totalnum)], gpu[:int(ratio[0] * totalnum)],
                                  d_list[:int(ratio[0] * totalnum)])
    valid_dataset = TensorDataset(all_feats[int(ratio[0] * totalnum):int((ratio[1]+ratio[0]) * totalnum)],
                                  gpu[int(ratio[0] * totalnum):int((ratio[1]+ratio[0]) * totalnum)],
                                  d_list[int(ratio[0] * totalnum):int((ratio[1]+ratio[0]) * totalnum)])
    test_dataset =  TensorDataset(all_feats[int((ratio[1]+ratio[0]) * totalnum):],gpu[int((ratio[1]+ratio[0]) * totalnum):],
                                  d_list[int((ratio[1]+ratio[0]) * totalnum):])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=True, drop_last=True)
    return train_loader, valid_loader, test_loader

def FC_dataloader(ratio, batch_size, file_path,lstm):
    dataprocess = Dataprocess(file_path)
    all_feats, gpu, cpu, d_list= dataprocess.encode(lstm=lstm)
    totalnum = len(cpu)
    all_feats = all_feats.reshape(-1,37*40)
    cpu = torch.tensor(cpu).reshape(totalnum, -1)
    gpu = torch.tensor(gpu).reshape(totalnum, -1)
    d_list =torch.tensor(d_list).reshape(totalnum, -1)
    # feature = all_feats[:int(0.8*totalnum)]
    train_dataset = TensorDataset(all_feats[:int(ratio[0] * totalnum)], gpu[:int(ratio[0] * totalnum)])
    valid_dataset = TensorDataset(all_feats[int(ratio[0] * totalnum):int((ratio[1]+ratio[0]) * totalnum)],
                                  gpu[int(ratio[0] * totalnum):int((ratio[1]+ratio[0]) * totalnum)])
    test_dataset =  TensorDataset(all_feats[int((ratio[1]+ratio[0]) * totalnum):],gpu[int((ratio[1]+ratio[0]) * totalnum):])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=True, drop_last=True)
    return train_loader, valid_loader, test_loader

class var_lstm_data(torch.utils.data.Dataset):
    def __init__(self, data_seq, data_label,d_list):
        self.data_seq = data_seq
        self.data_label = data_label
        self.d_list = d_list
    def __len__(self):
        return len(self.data_seq)
    def __getitem__(self, idx):
        return self.data_seq[idx], self.data_label[idx]

    def collate(data):
        data.sort(key=lambda x:len(x[0]), reverse=True)
        data_length = [len(sq[0]) for sq in data]
        x = [i[0] for i in data]
        y = [i[1] for i in data]
        data = torch.nn.utils.rnn.pad_sequence(x,batch_first=True,padding_value=0)
        return data.unsqueeze(-1), data_length, torch.tensor(y, dtype=torch.float32)

def var_lstm_dataloader(ratio, batch_size, file_path):
    dataprocess = Dataprocess(file_path)
    all_feats, gpu, cpu, d_list= dataprocess.encode()
    totalnum = len(cpu)
    cpu = torch.tensor(cpu).reshape(totalnum, -1)
    gpu = torch.tensor(gpu).reshape(totalnum, -1)
    d_list =torch.tensor(d_list).reshape(totalnum, -1)
    # feature = all_feats[:int(0.8*totalnum)]

    train_dataset = var_lstm_data(all_feats[:int(ratio[0] * totalnum)], cpu[:int(ratio[0] * totalnum)],d_list[:int(ratio[0] * totalnum)])
    valid_dataset = var_lstm_data(all_feats[int(ratio[0] * totalnum):int((ratio[1]+ratio[0]) * totalnum)],
                                  cpu[int(ratio[0] * totalnum):int((ratio[1]+ratio[0]) * totalnum)],
                                  d_list[:int(ratio[0] * totalnum)])
    test_dataset =  var_lstm_data(all_feats[int((ratio[1]+ratio[0]) * totalnum):],cpu[int((ratio[1]+ratio[0]) * totalnum):],
                                  d_list[:int(ratio[0] * totalnum)])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=True, drop_last=True)
    return train_loader, valid_loader, test_loader





