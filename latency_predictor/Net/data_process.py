import pandas as pd
import torch
import copy
import os
import random
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
        df = pd.read_csv(self.file_path, sep='\t')
        population = df.values
        random.shuffle(population)
        for sample in population:
            sample = eval(sample[0])
            para = sample[0]
            ks_list = copy.deepcopy(para['ks'])
            ex_list = copy.deepcopy(para['e'])
            d_list = copy.deepcopy(para['d'])
            r = copy.deepcopy(para['r'])[0]
            feats = self.spec2feats(ks_list, ex_list, d_list, r).reshape(1, -1)
            gpu.append(sample[1])
            cpu.append(sample[2])
            all_feats.append(feats)
            # print(all_feats)
        all_feats = torch.cat(all_feats, 0) #row
        # pred = self.model(all_feats).cpu()
        return all_feats, gpu, cpu


    # @staticmethod
    def spec2feats(self, ks_list, ex_list, d_list, r):
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
        ks_onehot = [0 for _ in range(60)]
        ex_onehot = [0 for _ in range(60)]
        r_onehot = [0 for _ in range(8)]

        for i in range(20):
            start = i * 3
            if ks_list[i] != 0:
                ks_onehot[start + ks_map[ks_list[i]]] = 1
            if ex_list[i] != 0:
                ex_onehot[start + ex_map[ex_list[i]]] = 1

        r_onehot[(r - 112) // 16] = 1
        return torch.Tensor(ks_onehot + ex_onehot + r_onehot)



