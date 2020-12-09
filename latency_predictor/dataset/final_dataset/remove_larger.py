import pandas as pd
import csv

PATH = './final.csv'
df = pd.read_csv(PATH)
cpu = df.iloc[:,-1].values
gpu = df.iloc[:,-2].values

csv_f = open('./final_.csv', 'w', encoding='utf-8', newline='')
csv_writer =csv.writer(csv_f)
csv_writer.writerow(
    ['arch_config', 'gpu latency', 'cpu latency',])
for k, gpu_time in  enumerate(gpu):
    detail = df.iloc[k]
    if gpu_time <= 30:
        csv_writer.writerow(detail)





