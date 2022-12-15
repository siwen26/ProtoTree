from prototree.prototree import ProtoTree
from util.log import Log

from util.args import get_args, save_args, get_optimizer
from util.data import get_dataloaders
from util.init import init_tree
from util.net import get_network, freeze
from util.visualize import gen_vis
from util.analyse import *
from util.save import *
from prototree.train import train_epoch, train_epoch_kontschieder
from prototree.test import eval, eval_fidelity
from prototree.prune import prune
from prototree.project import project, project_with_class_constraints
from prototree.upsample import upsample

import torch
from shutil import copy
from copy import deepcopy


if __name__ == '__main__':
    args = get_args()
    trainloader, testloader, classes = get_dataloaders(args)

    sample_train = next(iter(trainloader))
    print("one train batch")
    print(sample_train)
    print('='*50)

    features_net, add_on_layers = get_network(args)
    print("check features_net: bert embedding model")
    print(features_net)
    print('='*50)
    print("check the add on layers model: add on projection and conv2d layers")
    print(add_on_layers)
    print('='*50)

    print('check the call of these two models.')
    sequence_output, pooler_output = features_net(input_ids = sample_train[0], attention_mask = sample_train[1])
    print('bert embedding:')
    print(sequence_output)
    print('='*50)

    print('check the call of add on layers:')
    conv2d_results = add_on_layers(sequence_output, args)
    print(conv2d_results)
    print(conv2d_results.size())

    print('test Done!')

