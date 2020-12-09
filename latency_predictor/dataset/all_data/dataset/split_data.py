import pandas as pd
import csv
csv1 = open('./1.csv','w')
csv2 = open('./2.csv','w')
csv3 = open('3.csv', 'w')
csv5 = open('./5.csv','w')
csv4 = open('./4.csv','w')
csvfile = open('./25000.csv','r')
reader = csv.reader(csvfile)
# with csv.reader(open('./25000.csv', 'r')) as reader:
for i,item in enumerate(reader):
    if i <5000:
        w1 = csv.writer(csv1)
        w1.writerow(item)
    elif 5000<= i <10000:
        w2 =csv.writer(csv2)
        w2.writerow(item)
    elif 10000<= i <15000:
        w3 =csv.writer(csv3)
        w3.writerow(item)
    elif 15000<= i < 20000:
        w4 =csv.writer(csv4)
        w4.writerow(item)
    else:
        w5 = csv.writer(csv5)
        w5.writerow(item)
