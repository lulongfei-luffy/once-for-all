import os
import torch
from archmanager.finder import myArchManager
from archmanager.utils import getconfig,random_sample
from torchvision import transforms, datasets
import numpy as np
import time
import random
import math
import tensorrt as trt
from torch.autograd import Variable
from ofa.model_zoo import ofa_net
from ofa.utils import download_url,accuracy,AverageMeter
import common
from ofa.tutorial import AccuracyPredictor, FLOPsTable, LatencyTable, EvolutionFinder
import torch.backends.cudnn as cudnn
import csv



# set random seed
TRT_LOGGER = trt.Logger()
random_seed = 1
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
print('Successfully imported all packages and configured random seed to %d!'%random_seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = 'cuda:3' if torch.cuda.is_available() else 'cpu'

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

def get_engine(onnx_file_path):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    # def build_engine():
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 32  # 4G
        builder.max_batch_size = 1
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None

        network.get_input(0).shape = [1, 3, net_config['r'][0], net_config['r'][0]]
        # print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")
        return engine

if cuda_available:
    # path to the ImageNet dataset
    print("Please input the path to the ImageNet dataset.\n")
    imagenet_data_path = '../ofa/imagenet_codebase/dataset/mini_imagenet'
    # if 'imagenet_data_path' is empty, download a subset of ImageNet containing 2000 images (~250M) for test
    if not os.path.isdir(imagenet_data_path):
        os.makedirs(imagenet_data_path, exist_ok=True)
        print('%s is empty. Download a subset of ImageNet for test.' % imagenet_data_path)
    print('The ImageNet dataset files are ready.')


    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root=os.path.join(imagenet_data_path, 'val'),
            # transform=build_val_transform(224)
        ),
        batch_size=1,  # test batch size
        shuffle=True,
        num_workers=16,  # number of workers for the data loader
        pin_memory=True,
        drop_last=False,
    )
    print('The ImageNet dataloader is ready.')
else:
    print('Since GPU is not found in the environment, we skip all scripts related to ImageNet evaluation.')


gpu_ava_delay = AverageMeter()
cpu_ava_delay = AverageMeter()
resolution = [160, 176, 192, 208, 224]
onnxpath = './onnxs/tmp2.onnx'
csv_f = open('./dataset/tmep.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(csv_f)
csv_writer.writerow(['arch_config', 'gpu latency', 'cpu latency',])


archmanager = myArchManager()
# configs = getconfig(archmanager.depths,archmanager.resolutions)
configs = []
for item in csv.reader(open('latency_predictor/dataset/configs/high_latency_config1.csv', 'r')):
    configs.append(list(map(int,item)))

for config in configs:
    net_config = random_sample(depths=archmanager.depths, kernel_sizes=archmanager.kernel_sizes,
                               num_blocks=archmanager.num_blocks,
                               expand_ratios=archmanager.expand_ratios, num_stages=archmanager.num_stages)
    net_config = {'wid': None,
                  'ks': [7, 7, 7, 7, 5, 3, 5, 3, 5, 3, 7, 5, 7, 5, 3, 5, 5, 7, 5, 3, 5, 5, 5, 3, 5, 5, 3, 5, 3, 3, 3, 5, 3, 7, 3, 3, 3, 7, 5, 7],
                  'e': [6, 6, 4, 3, 6, 4, 4, 6, 6, 4, 6, 3, 6, 4, 4, 6, 4, 3, 3, 3, 4, 4, 6, 4, 4, 6, 6, 6, 3, 6, 6, 6, 4, 4, 6, 6, 6, 3, 6, 6],
                  'd': [4, 3, 3, 1, 4, 3, 2, 1, 3, 4],
                  'r': [1024]}
    # "{'wid': None,
    # 'ks': [7, 7, 7, 7, 5, 3, 5, 3, 5, 3, 7, 5, 7, 5, 3, 5, 5, 7, 5, 3, 5, 5, 5, 3, 5, 5, 3, 5, 3, 3, 3, 5, 3, 7, 3, 3, 3, 7, 5, 7],
    # 'e': [6, 6, 4, 3, 6, 4, 4, 6, 6, 4, 6, 3, 6, 4, 4, 6, 4, 3, 3, 3, 4, 4, 6, 4, 4, 6, 6, 6, 3, 6, 6, 6, 4, 4, 6, 6, 6, 3, 6, 6],
    # 'd': [4, 3, 3, 1, 4, 3, 2, 1, 3, 4],
    # 'r': [1024]}",44.3389,0

    # config = [4,2,4,4,4,3,4,2,3,4,1024]
    # config = [4]*10
    # config.append(1024)
    # net_config['d'] = config[:-1]
    # net_config['r'] = [config[-1]]
    print(net_config)

    data_loader.dataset.transform = transforms.Compose([
        transforms.Resize(int(math.ceil(int(net_config['r'][0])/0.875))),
        transforms.CenterCrop(net_config['r'][0]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    cudnn.benchmark = True
    assert 'ks' in net_config and 'd' in net_config and 'e' in net_config
    assert len(net_config['ks']) == 40 and len(net_config['e']) == 40 and len(net_config['d']) == 10

    ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
    subnet = ofa_network.get_active_subnet().to(device)

    input_name = ['input']
    output_name = ['output']
    input = Variable(torch.randn(1, 3, net_config['r'][0], net_config['r'][0])).cuda(device)
    torch.onnx.ExportTypes()
    torch.onnx.export(subnet, input, onnxpath, export_params=True,input_names=input_name, output_names=output_name, verbose=False)

    gpu_ava_delay.reset()
    cpu_ava_delay.reset()

    with get_engine(onnxpath) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.numpy(), labels.numpy()
            inputs[0].host = images.astype(np.float32)
            t1 = time.time()
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            t2 =time.time()
            print((t2-t1)*1000)
            if i >5:
                gpu_delay = (t2-t1) * 1000
                gpu_ava_delay.update(gpu_delay)
        csv_array = [net_config, round(gpu_ava_delay.avg, 4), round(cpu_ava_delay.avg, 4)]
        csv_writer.writerow(csv_array)
        csv_f.flush()






