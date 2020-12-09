import csv
import random

path = './high_latency_config2.csv'
csvfile = open(path, 'w')
csvwri = csv.writer(csvfile)

for num in range(1000):
    configs = []
    for i in range(10):
        configs.append(random.choices([1, 2, 3, 4], weights=[0.05, 0.1, 0.35, 0.5])[0])
    configs.append(random.choices([160, 320, 512, 768, 1024],weights=[0.05, 0.05, 0.1, 0.4, 0.4])[0])
    if sum(configs[:-1]) >32:
        csvwri.writerow(configs)
