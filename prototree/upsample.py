import os
import cv2
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from prototree.prototree import ProtoTree
from util.log import Log

from util.net import BERT_EMBEDDING
from transformers import BertTokenizer

import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns



# adapted from protopnet
def upsample(tree: ProtoTree, project_info: dict, project_loader: DataLoader, folder_name: str, args: argparse.Namespace, log: Log):
    dir = os.path.join(os.path.join(args.log_dir, args.dir_for_saving_images), folder_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with torch.no_grad():
        sim_maps, project_info, attn_maps = get_similarity_maps(tree, project_info, log)
        log.log_message("\nUpsampling prototypes for visualization...")
        imgs = project_loader.dataset
        for node, j in tree._out_map.items():
            if node in tree.branches: #do not upsample when node is pruned
                prototype_info = project_info[j]
                decision_node_idx = prototype_info['node_ix']
                x = imgs[prototype_info['input_image_ix']][0]
                attentions_array = attn_maps[j]
                fname=os.path.join(dir, '%s_bert_embedding_tsne_image.png'%str(decision_node_idx))
                draw_attention_map(attentions_array, decision_node_idx, fname)
                
                
#                 x.save(os.path.join(dir,'%s_original_image.png'%str(decision_node_idx)))
                    
#                 x_np = np.asarray(x)
#                 x_np = np.float32(x_np)/ 255
#                 if x_np.ndim == 2: #convert grayscale to RGB
#                     x_np = np.stack((x_np,)*3, axis=-1)
                
#                 img_size = x_np.shape[:2]
#                 similarity_map = sim_maps[j]

#                 rescaled_sim_map = similarity_map - np.amin(similarity_map)
#                 rescaled_sim_map= rescaled_sim_map / np.amax(rescaled_sim_map)
#                 similarity_heatmap = cv2.applyColorMap(np.uint8(255*rescaled_sim_map), cv2.COLORMAP_JET)
#                 similarity_heatmap = np.float32(similarity_heatmap) / 255
#                 similarity_heatmap = similarity_heatmap[...,::-1]
#                 plt.imsave(fname=os.path.join(dir,'%s_heatmap_latent_similaritymap.png'%str(decision_node_idx)), arr=similarity_heatmap, vmin=0.0,vmax=1.0)

#                 upsampled_act_pattern = cv2.resize(similarity_map,
#                                                     dsize=(img_size[1], img_size[0]),
#                                                     interpolation=cv2.INTER_CUBIC)
#                 rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
#                 rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
#                 heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_pattern), cv2.COLORMAP_JET)
#                 heatmap = np.float32(heatmap) / 255
#                 heatmap = heatmap[...,::-1]
                
#                 overlayed_original_img = 0.5 * x_np + 0.2 * heatmap
#                 plt.imsave(fname=os.path.join(dir,'%s_heatmap_original_image.png'%str(decision_node_idx)), arr=overlayed_original_img, vmin=0.0,vmax=1.0)

#                 # save the highly activated patch
#                 masked_similarity_map = np.zeros(similarity_map.shape)
#                 prototype_index = prototype_info['patch_ix']
#                 W, H = prototype_info['W'], prototype_info['H']
#                 assert W == H
#                 masked_similarity_map[prototype_index // W, prototype_index % W] = 1 #mask similarity map such that only the nearest patch z* is visualized
                
#                 upsampled_prototype_pattern = cv2.resize(masked_similarity_map,
#                                                     dsize=(img_size[1], img_size[0]),
#                                                     interpolation=cv2.INTER_CUBIC)
#                 plt.imsave(fname=os.path.join(dir,'%s_masked_upsampled_heatmap.png'%str(decision_node_idx)), arr=upsampled_prototype_pattern, vmin=0.0,vmax=1.0) 
                    
#                 high_act_patch_indices = find_high_activation_crop(upsampled_prototype_pattern, args.upsample_threshold)
#                 high_act_patch = x_np[high_act_patch_indices[0]:high_act_patch_indices[1],
#                                                     high_act_patch_indices[2]:high_act_patch_indices[3], :]
#                 plt.imsave(fname=os.path.join(dir,'%s_nearest_patch_of_image.png'%str(decision_node_idx)), arr=high_act_patch, vmin=0.0,vmax=1.0)

#                 # save the original image with bounding box showing high activation patch
#                 imsave_with_bbox(fname=os.path.join(dir,'%s_bounding_box_nearest_patch_of_image.png'%str(decision_node_idx)),
#                                     img_rgb=x_np,
#                                     bbox_height_start=high_act_patch_indices[0],
#                                     bbox_height_end=high_act_patch_indices[1],
#                                     bbox_width_start=high_act_patch_indices[2],
#                                     bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
#                   draw_word_embedding_fugure(x, decision_node_idx)
    
    return project_info


def draw_word_embedding_figure(word_input_ids, decision_node_idx):
        model = BERT_EMBEDDING.from_pretrained("bert-base-cased")
        output, _ = model(word_input_ids)
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
        token_label = tokenizer.convert_ids_to_tokens(word_input_ids.tolist()[0])
        np_output = output.detach().to('cpu').numpy()
        token_embedding = np_output.reshape(np_output.shape[1], -1)
        
        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(token_embedding)
        
        x = []
        y = []
        
        for value in new_values:
            x.append(value[0])
            y.append(value[1])
        
        plt.figure(figsize=(8,8)) 
        for i in range(len(x)):
            plt.scatter(x[i],y[i])
            plt.annotate(token_labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.savefig('%s_bert_embedding_tsne_image.png'%str(decision_node_idx))

  
def draw_attention_map(array_attention, decision_node_idx, filename):
      plt.rcParams['figure.figsize'] = (10, 5)
      color_palette = sns.diverging_palette(250, 0, as_cmap=True)
      heatmap = sns.heatmap(array_attention,
                cmap=color_palette,
                center=0,
                vmin=0,
                vmax=1,
               );
      heatmap.figure.savefig(filename)
        
        

def get_similarity_maps(tree: ProtoTree, project_info: dict, log: Log = None):
    log.log_message("\nCalculating similarity maps (after projection)...")
    
    sim_maps = dict()
    attn_maps = dict()
    for j in project_info.keys():
        nearest_x = project_info[j]['nearest_input']
        with torch.no_grad():
            _, distances_batch, _, attentions = tree.forward_partial(nearest_x, attention_masks=None)
            sim_maps[j] = torch.exp(-distances_batch[0,j,:,:]).cpu().numpy()
            attn_maps[j] = torch.mean(attentions[-1], dim = 1)[0].detach().numpy() # extract last attention layer attention weights.
        del nearest_x
        del project_info[j]['nearest_input']
    return sim_maps, project_info, attn_maps


# copied from protopnet
def find_high_activation_crop(mask,threshold):
    threshold = 1.-threshold
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > threshold:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > threshold:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > threshold:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > threshold:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1

# copied from protopnet
def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imshow(img_rgb_float)
    #plt.axis('off')
    plt.imsave(fname, img_rgb_float)
