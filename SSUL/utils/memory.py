"""
SSUL
Copyright (c) 2021-present NAVER Corp.
MIT License
"""

import math
import json
import os
import numpy as np
import torch
from PIL import Image

from torch.utils import data
from datasets import VOCSegmentation, ADESegmentation, LiveCellSegmentation, GlasSegmentation
from utils import ext_transforms as et
from utils.tasks import get_tasks

def memory_sampling_balanced(opts, prev_model):
    print("curr_step", opts.curr_step)
    fg_idx = 1 if opts.unknown else 0
    
    if opts.norm_type == 'livecell':
        m, s = [128 / 255]*3, [11.58 / 255]*3
    elif opts.norm_type == 'gray':
        m, s = [0.45]*3, [0.22]*3
    else:
        m, s = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform = et.ExtCompose([
        et.ExtResize(opts.crop_size),
        et.ExtCenterCrop(opts.crop_size),
        et.ExtToTensor(),
        et.ExtNormalize(mean=m,
                        std=s),
    ])
    
    if opts.dataset == 'voc':
        dataset = VOCSegmentation
    elif opts.dataset == 'ade':
        dataset = ADESegmentation
    elif opts.dataset == 'livecell':
        dataset = LiveCellSegmentation
    elif opts.dataset == 'glas':
        dataset = GlasSegmentation
    else:
        raise NotImplementedError
    
    if not os.path.isdir(f'./datasets/data/{opts.dataset}/{opts.task}_{opts.ckpt_postfix}'):
        os.mkdir(f'./datasets/data/{opts.dataset}/{opts.task}_{opts.ckpt_postfix}')
    memory_json = f'./datasets/data/{opts.dataset}/{opts.task}_{opts.ckpt_postfix}/memory.json'
    if os.path.isfile(memory_json):
        with open(memory_json, "r") as json_file:
            memory_list = json.load(json_file)
            if memory_list.get(f"step_{opts.curr_step}"):
                print(f"memory.json for step {opts.curr_step} already exist!")
                return
    
    print("...loading past train for memeory replay")
    train_dst = dataset(opts=opts, image_set='train', transform=transform, cil_step=opts.curr_step-1)
    
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, 
        shuffle=False, num_workers=4, drop_last=False)
    
    num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1)]
    prev_num_classes = sum(num_classes[:-1])  # 16
    
    
    if opts.curr_step > 1:
        with open(memory_json, "r") as json_file:
            memory_list = json.load(json_file)

        memory_candidates = memory_list[f"step_{opts.curr_step-1}"]["memory_candidates"]
    else:
        memory_list = {}
        memory_candidates = []
    
    print("...start memory candidates collection")
    for images, targets_all, _, img_names in train_loader:
        
        if opts.model == 'deeplabv3_resnet101_yolov4':
            targets = targets_all["seg"]
            det_targets = targets_all["det"]
        else:
            targets = targets_all
        
        with torch.no_grad():
            # current pseudo labeling
            if opts.curr_step > 1:
                images = images.cuda()
                targets = targets.cuda()
                
                if opts.model == 'deeplabv3_resnet101_yolov4':
                    outputs, det_outputs = prev_model(images)
                else:
                    outputs = prev_model(images)
                
                if opts.loss_type == 'ce_loss':
                    pred_logits = torch.softmax(outputs, 1).detach()
                else:
                    pred_logits = torch.sigmoid(outputs).detach()

                pred_scores, pred_labels = torch.max(pred_logits, dim=1)

                """ pseudo labeling """
                targets = torch.where( (targets <= fg_idx) & (pred_labels > fg_idx) & (pred_scores >= 0.7), 
                                         pred_labels, 
                                         targets)

        for b in range(images.size(0)):
            if opts.dataset == 'livecell':
                file_name = img_names['file_name'][b]
                file_id = img_names['file_id'][b].item()
                category_id = img_names['category_id'][b].item()
            else:
                img_name = img_names[b]
            target = targets[b]

            labels = torch.unique(target).detach().cpu().numpy()
            labels = (labels - 1).tolist() if opts.unknown else labels.tolist()
            
            if -1 in labels:
                labels.remove(-1)
                
            if 0 in labels:
                labels.remove(0)
            
            objs_num = len(labels)
            objs_ratio = int((target > fg_idx).sum())
            
            if opts.dataset == 'livecell':
                memory_candidates.append([{"file_name": file_name, "file_id": file_id, "category_id": category_id}, objs_num, objs_ratio, labels])
            else:
                memory_candidates.append([img_name, objs_num, objs_ratio, labels])

    print("...end memory candidates collection : ", len(memory_candidates))
    
    ####################################################################################################
    
    print("...start memory list generation")
    curr_memory_list = {f"class_{cls}":[] for cls in range(1, prev_num_classes)} # 1~15
    sorted_memory_candidates = memory_candidates.copy()
    np.random.shuffle(sorted_memory_candidates)
    
    random_class_order = list(range(1, prev_num_classes))
    np.random.shuffle(random_class_order)
    num_sampled = 0
    
    while opts.mem_size > num_sampled:
        
        for cls in random_class_order:
            for idx, mem in enumerate(sorted_memory_candidates):
                img_name, objs_num, objs_ratio, labels = mem

                if cls in labels:
                    curr_memory_list[f"class_{cls}"].append(mem)
                    num_sampled += 1
                    del sorted_memory_candidates[idx]
                    break
                    
            if opts.mem_size <= num_sampled:
                break
        
    ###################################### 
    
    """ save memory info """
    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory
    
    if opts.dataset == 'livecell':
        memory_list[f"step_{opts.curr_step}"] = {"memory_candidates": sampled_memory_list, 
                                                 "memory_list": sorted([mem[0] for mem in sampled_memory_list], key=lambda m: m["file_name"]),
                                                 }
    else:
        memory_list[f"step_{opts.curr_step}"] = {"memory_candidates": sampled_memory_list, 
                                                  "memory_list": sorted([mem[0] for mem in sampled_memory_list])
                                                 }
    
    with open(memory_json, "w") as json_file:
        json.dump(memory_list, json_file)
    
    memory_step_json = f'./datasets/data/{opts.dataset}/{opts.task}_{opts.ckpt_postfix}/memory_step{opts.curr_step}.json'
    with open(memory_step_json, "w") as json_file:
        json.dump(memory_list, json_file)


def memory_sampling_balanced_nop(opts):
    print("curr_step", opts.curr_step)
    fg_idx = 1 if opts.unknown else 0
    
    if opts.norm_type == 'livecell':
        m, s = [128 / 255]*3, [11.58 / 255]*3
    elif opts.norm_type == 'gray':
        m, s = [0.45]*3, [0.22]*3
    else:
        m, s = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform = et.ExtCompose([
        et.ExtResize(opts.crop_size),
        et.ExtCenterCrop(opts.crop_size),
        et.ExtToTensor(),
        et.ExtNormalize(mean=m,
                        std=s),
    ])
    
    if opts.dataset == 'voc':
        dataset = VOCSegmentation
    elif opts.dataset == 'ade':
        dataset = ADESegmentation
    elif opts.dataset == 'livecell':
        dataset = LiveCellSegmentation
    elif opts.dataset == 'glas':
        dataset = GlasSegmentation
    else:
        raise NotImplementedError
    
    if not os.path.isdir(f'./datasets/data/{opts.dataset}/{opts.task}_{opts.ckpt_postfix}'):
        os.mkdir(f'./datasets/data/{opts.dataset}/{opts.task}_{opts.ckpt_postfix}')
    memory_json = f'./datasets/data/{opts.dataset}/{opts.task}_{opts.ckpt_postfix}/memory.json'
    if os.path.isfile(memory_json):
        with open(memory_json, "r") as json_file:
            memory_list = json.load(json_file)
            if memory_list.get(f"step_{opts.curr_step}"):
                print(f"memory.json for step {opts.curr_step} already exist!")
                return
    
    print("...loading past train for memeory replay")
    train_dst = dataset(opts=opts, image_set='train', transform=transform, cil_step=opts.curr_step-1)
    
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, 
        shuffle=False, num_workers=4, drop_last=False)
    
    num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1)]
    prev_num_classes = sum(num_classes[:-1])  # 16
    
    
    if opts.curr_step > 1:
        with open(memory_json, "r") as json_file:
            memory_list = json.load(json_file)

        memory_candidates = memory_list[f"step_{opts.curr_step-1}"]["memory_candidates"]
    else:
        memory_list = {}
        memory_candidates = []
    
    print("...start memory candidates collection")
    for images, targets_all, _, img_names in train_loader:
        
        targets = targets_all

        for b in range(images.size(0)):
            if opts.dataset == 'livecell':
                file_name = img_names['file_name'][b]
                file_id = img_names['file_id'][b].item()
                category_id = img_names['category_id'][b].item()
            else:
                img_name = img_names[b]
            target = targets[b]

            labels = torch.unique(target).detach().cpu().numpy()
            labels = (labels - 1).tolist() if opts.unknown else labels.tolist()
            
            if -1 in labels:
                labels.remove(-1)
                
            if 0 in labels:
                labels.remove(0)
            
            objs_num = len(labels)
            objs_ratio = int((target > fg_idx).sum())
            
            if opts.dataset == 'livecell':
                memory_candidates.append([{"file_name": file_name, "file_id": file_id, "category_id": category_id}, objs_num, objs_ratio, labels])
            else:
                memory_candidates.append([img_name, objs_num, objs_ratio, labels])

    print("...end memory candidates collection : ", len(memory_candidates))
    
    ####################################################################################################
    
    print("...start memory list generation")
    curr_memory_list = {f"class_{cls}":[] for cls in range(1, prev_num_classes)} # 1~15
    sorted_memory_candidates = memory_candidates.copy()
    np.random.shuffle(sorted_memory_candidates)
    
    random_class_order = list(range(1, prev_num_classes))
    np.random.shuffle(random_class_order)
    num_sampled = 0
    
    while opts.mem_size > num_sampled:
        
        for cls in random_class_order:
            for idx, mem in enumerate(sorted_memory_candidates):
                img_name, objs_num, objs_ratio, labels = mem

                if cls in labels:
                    curr_memory_list[f"class_{cls}"].append(mem)
                    num_sampled += 1
                    del sorted_memory_candidates[idx]
                    break
                    
            if opts.mem_size <= num_sampled:
                break
        
    ###################################### 
    
    """ save memory info """
    sampled_memory_list = [mem for mem_cls in curr_memory_list.values() for mem in mem_cls]  # gather all memory
    
    if opts.dataset == 'livecell':
        memory_list[f"step_{opts.curr_step}"] = {"memory_candidates": sampled_memory_list, 
                                                 "memory_list": sorted([mem[0] for mem in sampled_memory_list], key=lambda m: m["file_name"]),
                                                 }
    else:
        memory_list[f"step_{opts.curr_step}"] = {"memory_candidates": sampled_memory_list, 
                                                  "memory_list": sorted([mem[0] for mem in sampled_memory_list])
                                                 }
    
    with open(memory_json, "w") as json_file:
        json.dump(memory_list, json_file)
    
    memory_step_json = f'./datasets/data/{opts.dataset}/{opts.task}_{opts.ckpt_postfix}/memory_step{opts.curr_step}.json'
    with open(memory_step_json, "w") as json_file:
        json.dump(memory_list, json_file)
