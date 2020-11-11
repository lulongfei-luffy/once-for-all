import pandas as pd
from scipy import randn
import csv

PATH = './dataset/latency_dataset_2_50000.csv'
df = pd.read_csv(PATH)
cpu = df.iloc[:,-1].values
gpu = df.iloc[:,-2].values

max_cpu = max(cpu)-16
min_cpu = min(cpu)+5
num_dict = {}
for i in range(int(min_cpu), int(max_cpu)):
    # if i not in num_dict.keys():
    num_dict[i] = 0  # use dict to count the number of each cpu time range
print(num_dict)
keys = list(num_dict)
csv_f = open('./dataset/latency_dataset_2_50000_temp.csv', 'a+', encoding='utf-8', newline='')
csv_writer =csv.writer(csv_f)
csv_writer.writerow(
    ['arch_config', 'gpu latency', 'cpu latency',])

for k, cpu_time in  enumerate(cpu):
    detail = df.iloc[k]
    for i in range(len(keys)-1):
        if cpu_time>=keys[i] and cpu_time<keys[i+1]:
            if num_dict[keys[i]]<1000:
                num_dict[keys[i]]+=1
                csv_writer.writerow(detail)
                csv_f.flush()
            break
        else:
            continue
print(num_dict)



