from __future__ import print_function
import os
import argparse
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import models
import time
from PIL import Image
import torchsummary
from prune import prune_network
from parameter import build_parser


def inference(img_name, prune_type, model_type):
    print(prune_type)
    print(model_type)
    classes = ['airplane', 'automobile', 'bird', 'cat','deer','dog','frog','horse','ship','truck']

    print("CUDA")
    print(torch.cuda.is_available())
    cuda = False
    if torch.cuda.is_available():
        print("Cuda availabel!")
        cuda = True

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    transform=transforms.Compose([
                        transforms.Resize((32,32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    if prune_type == "channel":
        if model_type == "resnet":
            model = models.__dict__["resnet"](dataset="cifar10", depth=50, cfg=[44, 56, 64, 100, 62, 61, 72, 51, 64, 182, 125, 128, 120, 121, 128, 138, 118, 127, 177, 126, 128, 418, 254, 256, 92, 191, 253, 226, 243, 256, 347, 250, 254, 545, 256, 254, 543, 254, 255, 20, 12, 175, 7, 3, 41, 4, 4, 77, 1405])
        if model_type == "vgg":
            model = models.__dict__["vgg"](dataset="cifar10", depth=19, cfg=[35, 64, 'M', 127, 128, 'M', 253, 249, 193, 125, 'M', 75, 21, 6, 2, 'M', 8, 9, 13, 67])

        if cuda:
            checkpoint = torch.load(model_type+"_pruned/model_best.pth.tar")
        else:
            checkpoint = torch.load(model_type+"_pruned/model_best.pth.tar", map_location=torch.device('cpu'))

        model.load_state_dict(checkpoint['state_dict'])

    if prune_type == "filter":
        if model_type == "resnet":
            parser = build_parser()
            args = parser.parse_args(["--vgg", "resnet50", "--prune-flag", "--load-path", "resnet_trained_models/check_point.pth", 
                                      "--save-path", "resnet_trained_models/pruning_results/", "--prune-layers", "block1", "block2", "block5", "block6", "block7", 
                                      "block9", "block10", "block11", "block12", "block15", "block16", "--independent-prune-flag"])
            model = prune_network(args, network=None)
            if cuda:
                checkpoint = torch.load("resnet_trained_models/check_point_retrain.pth")
            else:
                checkpoint = torch.load("resnet_trained_models/check_point_retrain.pth", map_location=torch.device('cpu'))

            model.load_state_dict(checkpoint['state_dict'])
        if model_type == "vgg":
            parser = build_parser()
            args = parser.parse_args(["--vgg", "vgg19_bn", "--prune-flag", "--load-path", "vgg_trained_models/check_point.pth",
                                    "--save-path", "vgg_trained_models/pruning_results/", "--prune-layers", 
                                    "conv1", "conv9", "conv10", "conv11", "conv12", "conv13", "conv14", "conv15", "conv16", "--prune-channels",
                                    "46", "256", "256", "256", "256", "256", "256", "256", "256", "--independent-prune-flag"])
            model = prune_network(args, network=None)

    if prune_type == "none":
        if model_type == "resnet":
            model = models.__dict__["resnet"](dataset="cifar10", depth=50)
        if model_type == "vgg":
            model = models.__dict__["vgg"](dataset="cifar10", depth=19)

        if cuda:
            checkpoint = torch.load(model_type+"/model_best.pth.tar")
        else:
            checkpoint = torch.load(model_type+"/model_best.pth.tar", map_location=torch.device('cpu'))
        
        model.load_state_dict(checkpoint['state_dict'])
    
    if cuda:
        model.cuda()

    model.eval()
    
    img = Image.open(img_name).convert('RGB')
    img_t = transform(img)
    print(img_t.shape)
    data = torch.unsqueeze(img_t, 0)
    if cuda:
        data = data.cuda()
    data = Variable(data, volatile=True)

    if cuda:

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        # GPU warms up
        _ = model(data)

        
        starter.record()
        output = model(data)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)/1000
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

    else:
        start = time.time()
        output = model(data)
        curr_time = time.time()-start
        pred = output.data.max(1, keepdim=True)[1]

    s, (total_params, trainable_params) = torchsummary.summary_string(model, (3,32,32))

    return classes[pred], str(curr_time), s, prune_type, model_type
