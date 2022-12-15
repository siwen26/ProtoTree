
import numpy as np
import argparse
import os
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Compose, Lambda

import os
import numpy as np
import shutil
import time
from PIL import Image
import unicodedata
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# def get_data(args: argparse.Namespace): 
#     """
#     Load the proper dataset based on the parsed arguments
#     :param args: The arguments in which is specified which dataset should be used
#     :return: a 5-tuple consisting of:
#                 - The train data set
#                 - The project data set (usually train data set without augmentation)
#                 - The test data set
#                 - a tuple containing all possible class labels
#                 - a tuple containing the shape (depth, width, height) of the input images
#     """
#     if args.dataset =='CUB-200-2011':
#         return get_birds(True, './data/CUB_200_2011/dataset/train_corners', './data/CUB_200_2011/dataset/train_crop', './data/CUB_200_2011/dataset/test_full')
#     if args.dataset == 'CARS':
#         return get_cars(True, './data/cars/dataset/train', './data/cars/dataset/train', './data/cars/dataset/test')
#     raise Exception(f'Could not load data set "{args.dataset}"!')

# def get_dataloaders(args: argparse.Namespace):
#     """
#     Get data loaders
#     """
#     # Obtain the dataset
#     trainset, projectset, testset, classes, shape  = get_data(args)
#     c, w, h = shape
#     # Determine if GPU should be used
#     cuda = not args.disable_cuda and torch.cuda.is_available()
#     trainloader = torch.utils.data.DataLoader(trainset,
#                                               batch_size=args.batch_size,
#                                               shuffle=True,
#                                               pin_memory=cuda
#                                               )
#     projectloader = torch.utils.data.DataLoader(projectset,
#                                             #    batch_size=args.batch_size,
#                                               batch_size=int(args.batch_size/4), #make batch size smaller to prevent out of memory errors during projection
#                                               shuffle=False,
#                                               pin_memory=cuda
#                                               )
#     testloader = torch.utils.data.DataLoader(testset,
#                                              batch_size=args.batch_size,
#                                              shuffle=False,
#                                              pin_memory=cuda
#                                              )
#     print("Num classes (k) = ", len(classes), flush=True)
#     return trainloader, projectloader, testloader, classes, c


# def get_birds(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size = 32): 
#     shape = (3, img_size, img_size)
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)
#     normalize = transforms.Normalize(mean=mean,std=std)
#     transform_no_augment = transforms.Compose([
#                             transforms.Resize(size=(img_size, img_size)),
#                             transforms.ToTensor(),
#                             normalize
#                         ])
#     if augment:
#         transform = transforms.Compose([
#             transforms.Resize(size=(img_size, img_size)),
#             transforms.RandomOrder([
#             transforms.RandomPerspective(distortion_scale=0.2, p = 0.5),
#             transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.02,0.02)),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomAffine(degrees=10, shear=(-2,2),translate=[0.05,0.05]),
#             ]),
#             transforms.ToTensor(),
#             normalize,
#         ])
#     else:
#         transform = transform_no_augment

#     trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
#     projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)
#     testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
#     classes = trainset.classes
#     for i in range(len(classes)):
#         classes[i]=classes[i].split('.')[1]
#     return trainset, projectset, testset, classes, shape


# def get_cars(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size = 224): 
#     shape = (3, img_size, img_size)
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)

#     normalize = transforms.Normalize(mean=mean,std=std)
#     transform_no_augment = transforms.Compose([
#                             transforms.Resize(size=(img_size, img_size)),
#                             transforms.ToTensor(),
#                             normalize
#                         ])

#     if augment:
#         transform = transforms.Compose([
#             transforms.Resize(size=(img_size+32, img_size+32)), #resize to 256x256
#             transforms.RandomOrder([
#             transforms.RandomPerspective(distortion_scale=0.5, p = 0.5),
#             transforms.ColorJitter((0.6,1.4), (0.6,1.4), (0.6,1.4), (-0.4,0.4)),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomAffine(degrees=15,shear=(-2,2)),
#             ]),
#             transforms.RandomCrop(size=(img_size, img_size)), #crop to 224x224
#             transforms.ToTensor(),
#             normalize,
#         ])
#     else:
#         transform = transform_no_augment

#     trainset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
#     projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)
#     testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
#     classes = trainset.classes
    
#     return trainset, projectset, testset, classes, shape

def get_train_test_filenames(args: argparse.Namespace):
    # def get_train_test_filenames(dataset_pth:str):
    """
    dataset_pth: parent dir of images.txt etc.
    """
    path_images = os.path.join(args.dataset_pth,'images.txt')
    path_split = os.path.join(args.dataset_pth,'train_test_split.txt')

    idx_to_img = {}
    with open(path_images, 'r') as f:
        img_info = f.read().split('\n')[:-1]
        for each in img_info:
            id = each.split(' ')[0]
            img = each.split(' ')[1]
            img_name = '.'.join(img.split('.')[:-1])
            idx_to_img[id] = img_name
    
    idx_to_split = {}
    with open(path_split, 'r') as f:
        split = f.read().split('\n')[:-1]
        for each in split:
            id = each.split(' ')[0]
            label = int(each.split(' ')[1])
            idx_to_split[id] = label
    
    train_filenames = []
    test_filenames = []
    for k in idx_to_split:
        if idx_to_split[k] == 1:
            train_filenames.append(idx_to_img[k])
        elif idx_to_split[k] == 0:
            test_filenames.append(idx_to_img[k])

    return train_filenames, test_filenames


def get_file_classes(dataset_pth:str, filenames):
    """
    filenames: list contains all train/test filenames.
    return dict type.
    """
    path_classes = os.path.join(dataset_pth,'classes.txt')
    classes_to_idx = {}
    with open(path_classes, 'r') as f:
        classes = f.read().split('\n')[:-1]
        for each in classes:
            id = str(int(each.split(' ')[0])-1)
            label = each.split(' ')[1]
            classes_to_idx[label] = id

    classes_dict = {}
    for item in filenames:
        item_class = classes_to_idx[item.split('/')[0]]
        classes_dict[item] = int(item_class)
    return classes_dict

def get_text_data(filename_list, prefix_dir):
    content_list = []
    for f in filename_list:
        file_pth = prefix_dir + '/' + f + '.txt'
        text = open(file_pth, "r").read()
        content = ' '.join(text.split('\n')[:-1])
        # decode and encode unrecognized character: ascii to utf-8
        decode_content = unicodedata.normalize('NFKD', content).encode('ascii', 'replace').decode('utf-8')
        new_content = ' '.join(decode_content.split('??'))
        content_list.append(new_content)
    return content_list


def preprocess_dataset(args: argparse.Namespace):
    # def preprocess_dataset(dataset_pth:str, prefix_dir:str):
    """
    dataset_pth: parent dir of images.txt etc.;
    prefix_dir: parent directory of the txt files.
    """
    train_filenames, test_filenames = get_train_test_filenames(args)
    train_labels = list(get_file_classes(args.dataset_pth, train_filenames).values())
    test_labels = list(get_file_classes(args.dataset_pth, test_filenames).values())
    train_texts = get_text_data(train_filenames, args.text_pth)
    test_texts = get_text_data(test_filenames, args.text_pth)

    return train_texts, test_texts, train_labels, test_labels


def tokenization(texts, max_length, pretrain_model='bert-base-cased'):
    # Load the BERT tokenizer.
    # print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(pretrain_model, do_lower_case=True)

    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in texts:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_length,           # Pad & truncate all sentences.
                            padding='max_length',
                            truncation = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                      )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    # labels = torch.tensor(train_labels)
    return input_ids, attention_masks


def encoded_dataset(texts, labels, max_length, pretrain_model='bert-base-cased'):
    # def encoded_dataset(texts, labels, max_length, pretrain_model='bert-base-cased'):
    input_ids, attention_masks = tokenization(texts, max_length, pretrain_model)
    target_labels = torch.tensor(labels)
    dataset = TensorDataset(input_ids, attention_masks, target_labels)
    return dataset


def get_dataloaders(args: argparse.Namespace):
    # def get_dataloaders(dataset_pth:str, prefix_dir:str, batch_size, max_length, pretrain_model='bert-base-cased')
    
    # get trainset/testset text and label
    train_texts, test_texts, train_labels, test_labels = preprocess_dataset(args)

    # tokenization and get dataset
    train_dataset = encoded_dataset(train_texts, train_labels, args.max_length, args.pretrain_model)
    test_dataset = encoded_dataset(test_texts, test_labels, args.max_length, args.pretrain_model)

    # get Dataloaders
    train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = args.batch_size # Trains with this batch size.
        )
    
    # For test the order doesn't matter, so we'll just read them sequentially.
    test_dataloader = DataLoader(
                test_dataset, # The test samples.
                sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
                batch_size = args.batch_size # Evaluate with this batch size.
            )
    
    classes = list(set(train_labels))
    print("Num classes (k) = ", len(classes), flush=True)

    return train_dataloader, test_dataloader, classes
