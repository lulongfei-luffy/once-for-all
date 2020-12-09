import csv
import os

root_path = './latency_dataset/high_latency_dataset'
des_path = './final_dataset/union_high.csv'
csvf = open(des_path, 'w', encoding='utf-8', newline='')
csvwriter = csv.writer(csvf)
csvwriter.writerow(['arch_config','gpu latency','cpu latency'])

for dir in os.listdir(root_path):
    csvfile = open(os.path.join(root_path, dir))
    csvreader = csv.reader(csvfile)
    for i, row in enumerate(csvreader):
        if i > 0:
           csvwriter.writerow(row)