import argparse
from pathlib import Path

import torch
from nets.yolo import YoloBody
from yolo import  YOLO


from utils.general import set_logging, increment_path
from utils.torch_utils import select_device, TracedModel
from utils.attentions import se_block
import torch_pruning as tp

def detect():
    trace, ch_sparsity =not opt.no_trace, opt.ch_sparsity

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model_load = YOLO()
    model = YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], model_load.num_classes, model_load.phi, pretrained=False, phi_attention=1,pruned=1)   # load FP32 model
    model = model.cuda()
    print(model)

    # Pruning
    example_inputs = torch.randn(1, 3, 224, 224).to(device)
    # imp = tp.importance.RandomImportance()  #random权重判断
    imp = tp.importance.BNScaleImportance() #slim random


    #哪些层不进行剪枝，请添加到ignored_layers
    ignored_layers = []
    for m in model.modules():
        if  isinstance(m,torch.nn.Conv2d ) and m.out_channels==24 and m.kernel_size== (1,1):
            ignored_layers.append(m)
        if  isinstance(m,se_block ) :
            ignored_layers.append(m)
    print(ignored_layers)

    iterative_steps = 5 # progressive pruning
    print("====1=====")
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=iterative_steps,
        ch_sparsity=ch_sparsity, # 剪枝率 remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256} 11/16
        ignored_layers=ignored_layers,
    )
    print("====2=====")
    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print("Before Pruning: FLOPs=%f M, #Params=%f"%(base_macs/ 1e6, base_nparams))
    for i in range(iterative_steps):
        pruner.step()
        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(model, example_inputs)
        print("After iter %d Pruning: MACs=%f M, #Params=%f"%(i+1,pruned_macs / 1e6, pruned_nparams))
    print(model)

    if trace:
        model = TracedModel(model, device, opt.img_size,ch_sparsity,iterative_steps)

    if half:
        model.half()  # to FP16

if __name__ == '__main__':
    '''
    本项目引用了Torch-Pruning剪枝方法，剪枝模型时，需要注意以下几点：
    1.剪枝率中，分母应该为4，8的倍数，如1/4，5/8，9/16，否则会导致再训练的时网络结构并不符合，无法再训练
    2.model path 在yolo中修改
    3.剪枝完成后，会自动调用TracedModel()保存权值在model_data/中，如果需要修改保存路径，请进入TracedModel修改。默认保存为 model_data/step{iterative_steps}-prunedmodel-{ch_sparsity}.pth
    4.对于不需要剪枝的模块，请添加到ignored_layers，如果遇到无法剪枝的结构，亦添加到ignored_layers, 查看哪些模块可剪枝：https://github.com/VainF/Torch-Pruning
    5.目前本代码仅支持slim random的重要性判断器，更多细节请参考https://github.com/VainF/Torch-Pruning
    
    若想剪枝更多模型，请参考https://github.com/VainF/Torch-Pruning
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--ch_sparsity', type=float, default=0.5625, help='remove ??% channels')
    opt = parser.parse_args()
    print(opt)

    detect()
