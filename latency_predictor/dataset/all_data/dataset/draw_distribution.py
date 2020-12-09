from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
# root_path = './all_dataset/version_2'
#
des_path = './union_2.csv'
# csvf = open(des_path, 'w', encoding='utf-8', newline='')
# csvwriter = csv.writer(csvf)
# csvwriter.writerow(['arch_config','gpu latency','cpu latency'])
#
# for dir in os.listdir(root_path):
#     csvfile = open(os.path.join(root_path, dir))
#     csvreader = csv.reader(csvfile)
#     for i, row in enumerate(csvreader):
#         if i > 0:
#            csvwriter.writerow(row)

distilled_path = './all_dataset/version_2/union_distilled.csv'

df = pd.read_csv(des_path)
gpu = df.iloc[:,-2].values

print(gpu)
fig =plt.figure()
plt.hist(gpu,bins=10)
plt.ylabel('probolity')
plt.xlabel('gpu latency')
plt.show()
