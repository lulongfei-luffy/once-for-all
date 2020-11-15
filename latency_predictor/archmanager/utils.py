import os
import csv
import numpy.random as random
def MAPE(outputs, labels):
    # mean absolute percent error
    relative_MSE = 0
    count = 0
    for i, j in zip(outputs, labels):
        for m, n in zip(i, j):
            # print(m, n)
            # logger.info('outputs:{}, labels:{}'.format(m, n))
            relative_MSE += (m[0] / n[0] - 1) ** 2
            count += 1
    relative_MSE /= count


def getconfig(depth,resolution):
    config = []
    for i1 in depth:
        for i2 in depth:
            for i3 in depth:
                for i4 in depth:
                    for i5 in depth:
                        for i6 in depth:
                            for i7 in depth:
                                for i8 in depth:
                                    for i9 in depth:
                                        for i10 in depth:
                                            for j in resolution:
                                                depths = []
                                                depths.append(i1)
                                                depths.append(i2)
                                                depths.append(i3)
                                                depths.append(i4)
                                                depths.append(i5)
                                                depths.append(i6)
                                                depths.append(i7)
                                                depths.append(i8)
                                                depths.append(i9)
                                                depths.append(i10)
                                                depths.append(j)
                                                config.append(depths)
    file_path = './dataset/config.csv'
    if not os.path.exists(file_path):
        # csvfile = open(file_path,'w')
        csvfile = open(file_path, 'w', encoding='utf-8', newline='')
        writer = csv.writer(csvfile)
        for fig in config:
            writer.writerow(fig)
            csvfile.flush()

    return config

def random_sample(num_stages, depths,num_blocks,expand_ratios,kernel_sizes):
    sample = {}
    d = []
    e = []
    ks = []
    # for i in range(num_stages):
    #     d.append(random.choice(depths, p=[0.25] * 4))

    for i in range(num_blocks):
        e.append(random.choice(expand_ratios))
        ks.append(random.choice(kernel_sizes))

    sample = {
        'wid': None,
        'ks': ks,
        'e': e,
        'd': d,
        'r': []
    }

    return sample




