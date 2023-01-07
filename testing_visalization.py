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

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu')
        
    # Log which device was actually used
    print('Device used: ',str(device))

    # Load trained ProtoTree
    pruned_tree = ProtoTree.load('/content/ProtoTree/runs/protoree_cub/checkpoints/pruned').to(device=device)
    
    project_info, tree = project_with_class_constraints(deepcopy(pruned_tree), trainloader, device, args)
    name = "pruned_and_projected"
    # Upsample prototype for visualization
    project_info = upsample(tree, project_info, trainloader, name, args)

    # visualize tree
    gen_vis(tree, name, args, classes)











