from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# path = "./latency_dataset_2_1000.csv"
path = './latency_dataset_2_1000.csv'
df = pd.read_csv(path)
gpu = df.iloc[:,-2].values
cpu = df.iloc[:,-1].values

figrue =plt.figure()
plt.hist(cpu,label='cpu',bins=20,normed=False)
plt.ylabel('probility')
plt.xlabel('cpu latency')
plt.show()

fig =plt.figure()
plt.hist(gpu,bins=20,normed=False)
plt.ylabel('probolity')
plt.xlabel('gpu latency')
plt.show()