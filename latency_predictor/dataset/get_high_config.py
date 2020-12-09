import csv
import random

path = 'configs/high_latency_config6.csv'
csvfile = open(path, 'w')
csvwri = csv.writer(csvfile)
count = 0
while True:
    configs = []

    if count >= 1000:
        break
    for i in range(10):
        configs.append(random.choices([1, 2, 3, 4], weights=[0.05, 0.05, 0.4, 0.5])[0])
    configs.append(random.choices([160, 320, 512, 768, 1024],weights=[0.01, 0.01, 0.05, 0.43, 0.5])[0])
    if sum(configs[:-1]) > 32:
        csvwri.writerow(configs)
        count += 1
