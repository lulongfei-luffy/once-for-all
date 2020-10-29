import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import numpy as np
import time
import random
import math
import copy
from matplotlib import pyplot as plt

from ofa.model_zoo import ofa_net
from ofa.utils import download_url,accuracy,AverageMeter
import  datetime
# from ofa.tutorial.accuracy_predictor import AccuracyPredictor
# from ofa.tutorial.flops_table import FLOPsTable
# from ofa.tutorial.latency_table import LatencyTable
# from ofa.tutorial.evolution_finder import EvolutionFinder
# from ofa.tutorial.imagenet_eval_helper import evaluate_ofa_subnet, evaluate_ofa_specialized,
from ofa.tutorial.imagenet_eval_helper import calib_bn, validate
from ofa.tutorial import AccuracyPredictor, FLOPsTable, LatencyTable, EvolutionFinder
from ofa.tutorial import evaluate_ofa_subnet, evaluate_ofa_specialized
import torch.backends.cudnn as cudnn
import csv

# set random seed
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(random_seed)
    print('Using GPU.')
else:
    print('Using CPU.')

ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)
print('The OFA Network is ready.')

if cuda_available:
    # path to the ImageNet dataset
    print("Please input the path to the ImageNet dataset.\n")
    # imagenet_data_path = './imagedata/'
    imagenet_data_path = './imagenet_1_pic/'
    # imagenet_data_path = '/home/lvbo1/lvbo_dir/00_dataset/imagenet'

    # if 'imagenet_data_path' is empty, download a subset of ImageNet containing 2000 images (~250M) for test
    if not os.path.isdir(imagenet_data_path):
        os.makedirs(imagenet_data_path, exist_ok=True)
        print('%s is empty. Download a subset of ImageNet for test.' % imagenet_data_path)

    print('The ImageNet dataset files are ready.')
else:
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')

if cuda_available:
    # The following function build the data transforms for test
    def build_val_transform(size):
        return transforms.Compose([
            transforms.Resize(int(math.ceil(size / 0.875))),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root=os.path.join(imagenet_data_path, 'val'),
            transform=build_val_transform(224)
        ),
        batch_size=1,  # test batch size
        shuffle=True,
        num_workers=16,  # number of workers for the data loader
        pin_memory=True,
        drop_last=False,
    )
    print('The ImageNet dataloader is ready.')
else:
    data_loader = None
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')

target_hardware = 'note10'
latency_table = LatencyTable(device=target_hardware)
print('The Latency lookup table on %s is ready!' % target_hardware)

# accuracy predictor
accuracy_predictor = AccuracyPredictor(
    pretrained=True,
    device='cuda:0' if cuda_available else 'cpu'
)

print('The accuracy predictor is ready!')
print(accuracy_predictor.model)

latency_constraint = 50 # ms, suggested range [15, 33] ms
P = 100  # The size of population in each generation
N = 500  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
params = {
    'constraint_type': target_hardware, # Let's do FLOPs-constrained search
    'efficiency_constraint': latency_constraint,
    'mutate_prob': 0.1, # The probability of mutation in evolutionary search
    'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
    'efficiency_predictor': latency_table, # To use a predefined efficiency predictor.
    'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
}
# build the evolution finder
finder = EvolutionFinder(**params)


population = []  # (validation, sample, latency) tuples
child_pool = []
efficiency_pool = []
population_size = 50000

gpu_ava_delay = AverageMeter()
cpu_ava_delay = AverageMeter()

csv_f = open('./latency_dataset_2_50000.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(csv_f)
csv_writer.writerow(
    ['arch_config', 'gpu latency', 'cpu latency',])
# batch_size = 2
for _ in range(population_size):
    net_config, efficiency = finder.random_sample()
    child_pool.append(net_config)
    efficiency_pool.append(efficiency)

    data_loader.dataset.transform = transforms.Compose([
        transforms.Resize(int(math.ceil(net_config['r'][0] / 0.875))),
        transforms.CenterCrop(net_config['r'][0]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().to(device)

    assert 'ks' in net_config and 'd' in net_config and 'e' in net_config
    assert len(net_config['ks']) == 20 and len(net_config['e']) == 20 and len(net_config['d']) == 5
    ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
    subnet = ofa_network.get_active_subnet().to(device)
    # print('==================={%s}===============' % net_config)
    # calib_bn(subnet, imagenet_data_path, net_config['r'][0], batch_size)

    subnet.eval()
    subnet = subnet.to(device)

    subnet_cpu = copy.deepcopy(subnet)
    subnet_cpu.cpu()

    gpu_ava_delay.reset()
    cpu_ava_delay.reset()

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            # compute output
            start = datetime.datetime.now()
            output = subnet(images)
            gpu_delay = (datetime.datetime.now() - start).total_seconds()*1000
            # measure accuracy and record loss
            # acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            if i != 0:
                gpu_ava_delay.update(gpu_delay)

            images, labels = images.to('cpu'), labels.to('cpu')
            # compute output
            start = datetime.datetime.now()
            output = subnet_cpu(images)
            cpu_delay = (datetime.datetime.now() - start).total_seconds()*1000
            # measure accuracy and record loss
            # acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            if i != 0:
                cpu_ava_delay.update(cpu_delay)

        csv_array = [net_config, round(gpu_ava_delay.avg,2), round(cpu_ava_delay.avg,2)]
        csv_writer.writerow(csv_array)
        csv_f.flush()

        # print("gpu avarage delay:{:.3f}, cpu avarage delay:{:.3f}".format(gpu_ava_delay.avg, cpu_ava_delay.avg))

