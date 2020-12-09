import pandas as pd
from scipy import randn
import csv

PATH = './union_2.csv'
df = pd.read_csv(PATH)
cpu = df.iloc[:,-1].values
gpu = df.iloc[:,-2].values

max_cpu = max(cpu)-16
min_cpu = min(cpu)+5
max_gpu = max(gpu)
min_gpu = min(gpu)

num_dict = {}
for i in range(0, 21):
    # if i not in num_dict.keys():
    num_dict[i] = 0  # use dict to count the number of each cpu time range
print(num_dict)
keys = list(num_dict)
csv_f = open('./union_distilled_2.csv', 'w', encoding='utf-8', newline='')
csv_writer =csv.writer(csv_f)
csv_writer.writerow(
    ['arch_config', 'gpu latency', 'cpu latency',])

for k, gpu_time in  enumerate(gpu):
    detail = df.iloc[k]
    for i in range(len(keys)-1):
        if gpu_time>=keys[i] and gpu_time<keys[i+1]:
            if num_dict[keys[i]]<300:
                num_dict[keys[i]]+=1
                csv_writer.writerow(detail)
                csv_f.flush()
            break
        else:
            continue
print(num_dict)
# print(len())



