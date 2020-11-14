import pandas as pd
import torch
import copy
import os
import random

from torch.utils.data import TensorDataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def construct_maps(keys):
    d = dict()
    keys = list(set(keys))
    for k in keys:
        if k not in d:
            d[k] = len(list(d.keys()))
    return d

ks_map = construct_maps(keys=(3, 5, 7))  # {3: 0, 5: 1, 7: 2}
ex_map = construct_maps(keys=(3, 4, 6))
dp_map = construct_maps(keys=(2, 3, 4))

class Dataprocess():
    def __init__(self,file_path):
        self.file_path = file_path

    def encode(self):
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
            feats= self.spec2feats(ks_list, ex_list, d_list, r)
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
        ks_onehot = [0 for _ in range(120)]
        # ex_onehot = [0 for _ in range(60)]
        r_onehot = [0 for _ in range(8)]

        for i in range(40):
            start = i * 3
            if ks_list[i] != 0:
                ks_onehot[start + ks_map[ks_list[i]]] = 1
                ks_onehot[start + 3 + ex_map[ex_list[i]]] = 1

        r_onehot[(r - 112) // 16] = 1
        return torch.Tensor(ks_onehot + r_onehot)

def dataloader(ratio, batch_size, file_path):
    dataprocess = Dataprocess(file_path)
    all_feats, gpu, cpu, d_list= dataprocess.encode()
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





