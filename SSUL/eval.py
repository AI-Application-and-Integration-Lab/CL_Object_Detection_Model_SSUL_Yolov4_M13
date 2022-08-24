"""
SSUL
Copyright (c) 2021-present NAVER Corp.
MIT License
"""

from tqdm import tqdm
import network
import utils
import os
import time
import random
import argparse
import numpy as np
import cv2

from torch.utils import data
from datasets import VOCSegmentation, ADESegmentation, LiveCellSegmentation, GlasSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.utils import AverageMeter
from utils.tasks import get_tasks
from utils.memory import memory_sampling_balanced

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import torchvision.transforms.functional as F
# from torchvision.transforms import InterpolationMode

torch.backends.cudnn.benchmark = True

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/data/DB/VOC2012',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc', choices=['voc', 'ade', 'livecell', 'glas'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None, help="num classes (default: None)")
    
    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--amp", action='store_true', default=False)
    parser.add_argument("--freeze", action='store_true', default=False)
    
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--train_epoch", type=int, default=0,
                        help="epoch number (default: 0")
    parser.add_argument("--curr_itrs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='warm_poly', choices=['poly', 'step', 'warm_poly'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")

    parser.add_argument("--loss_type", type=str, default='bce_loss',
                        choices=['ce_loss', 'focal_loss', 'bce_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    parser.add_argument("--ckpt_postfix", type=str, default='1',
                        help="postfix name for ckpt (default: 1)")
    parser.add_argument("--norm_type", type=str, default="default", help="norm type")
    
    # CIL options
    parser.add_argument("--pseudo", action='store_true', default=False)
    parser.add_argument("--pseudo_thresh", type=float, default=0.7)
    parser.add_argument("--task", type=str, default='15-1')
    parser.add_argument("--curr_step", type=int, default=0)
    parser.add_argument("--overlap", action='store_true', default=False)
    parser.add_argument("--mem_size", type=int, default=0)
    
    parser.add_argument("--bn_freeze", action='store_true', default=False)
    parser.add_argument("--w_transfer", action='store_true', default=False)
    parser.add_argument("--unknown", action='store_true', default=False)
    parser.add_argument("--save_mask", action='store_true', default=False)
    parser.add_argument("--save_mask_dir", type=str, default='./masks', help="path to mask dir")
    parser.add_argument("--task_incremental", action='store_true', default=False)
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.norm_type == 'livecell':
        m, s = [128 / 255]*3, [11.58 / 255]*3
    elif opts.norm_type == 'gray':
        m, s = [0.45]*3, [0.22]*3
    else:
        m, s = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    train_transform = et.ExtCompose([
        #et.ExtResize(size=opts.crop_size),
        et.ExtRandomScale((0.5, 2.0)),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=m,
                        std=s),
    ])
    if opts.crop_val and not opts.save_mask:
        val_transform = et.ExtCompose([
            et.ExtResize(opts.crop_size),
            et.ExtCenterCrop(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=m,
                            std=s),
        ])
        # et.ExtResize((opts.crop_size, opts.crop_size)),
        # et.ExtCenterCrop(opts.crop_size),
    else:
        val_transform = et.ExtCompose([
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
        
    dataset_dict = {}
    if opts.save_mask:
        dataset_dict['train'] = dataset(opts=opts, image_set='train', transform=val_transform, cil_step=0)

        dataset_dict['val'] = dataset(opts=opts,image_set='val', transform=val_transform, cil_step=0)
    
    dataset_dict['test'] = dataset(opts=opts, image_set='test', transform=val_transform, cil_step=0)
    
    if opts.curr_step > 0 and opts.mem_size > 0:
        dataset_dict['memory'] = dataset(opts=opts, image_set='memory', transform=train_transform, 
                                                 cil_step=opts.curr_step, mem_size=opts.mem_size)

    return dataset_dict


def save_img(target, details, opts, img_type):
    colors = [
        [0, 0, 0],
        [120, 120, 120],
        [51, 102, 204],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [255, 0, 0],
        [255, 255, 0],
        [204, 5, 255],
    ]

    if img_type == 0:
        img = (np.transpose(target, (1,2,0)) - target.min()) / (target.max() - target.min())
        img = Image.fromarray(np.uint8(img * 255))
        
        if opts.dataset == "livecell":
            fname = details['file_name'][0].replace('.tif', '_ori.jpg')
        elif opts.dataset == "glas":
            fname = details[0] + '_ori.jpg'
    else:
        img = np.zeros((target.shape[0], target.shape[1], 3))
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                img[i, j] = colors[target[i, j]]
        img = Image.fromarray(np.uint8(img))
        if img_type == 1:
            if opts.dataset == "livecell":
                fname = details['file_name'][0].replace('.tif', '_pred.jpg')
            elif opts.dataset == "glas":
                fname = details[0] + '_pred.jpg'
        else:
            if opts.dataset == "livecell":
                fname = details['file_name'][0].replace('.tif', '_gt.jpg')
            elif opts.dataset == "glas":
                fname = details[0] + '_gt.jpg'
    
    if not os.path.isdir(f'./results/{opts.task}_{opts.ckpt_postfix}'):
        Path(f'./results/{opts.task}_{opts.ckpt_postfix}').mkdir(parents=True, exist_ok=True)
    img.save(f'./results/{opts.task}_{opts.ckpt_postfix}/{fname}')


# for livecell
def remap(preds):
    cls_map = {1:5, 2:6, 3:7, 4:8, 5:1, 6:2, 7:3, 8:4}
    new_preds = np.zeros_like(preds)
    for cls in cls_map:
        new_preds += np.where(preds == cls, cls_map[cls], 0)
    return new_preds


def task_mask(opts, file_name):
    classes = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SkBr", "SKOV3"]
    if opts.task == "4-4":
        mask = torch.Tensor([1, 0, 0, 0, 0, 1, 1, 1, 1])
        for i in range(4):
            if classes[i] in file_name:
                mask = torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0])
                break
    elif opts.task == "4-1":
        mask = torch.Tensor([1, 1, 1, 1, 1, 0, 0, 0, 0])
        if classes[4] in file_name:
            mask = torch.Tensor([1, 0, 0, 0, 0, 1, 0, 0, 0])
        elif classes[5] in file_name:
            mask = torch.Tensor([1, 0, 0, 0, 0, 0, 1, 0, 0])
        elif classes[6] in file_name:
            mask = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 1, 0])
        elif classes[7] in file_name:
            mask = torch.Tensor([1, 0, 0, 0, 0, 0, 0, 0, 1])
    return mask


# +
def validate(opts, model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    interval = 10
    with torch.no_grad():
        for i, (images, labels, _, details) in enumerate(loader):
            if (i + 1) % 10 == 0 or (i + 1) == len(loader):
                print(f"{i + 1} / {len(loader)}")
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            
            outputs = model(images)
            
            if opts.loss_type == 'bce_loss':
                outputs = torch.sigmoid(outputs)
            else:
                outputs = torch.softmax(outputs, dim=1)
                    
            # remove unknown label
            if opts.unknown:
                outputs[:, 1] += outputs[:, 0]
                outputs = outputs[:, 1:]
            
            if opts.task_incremental:
                mask = task_mask(opts, details['file_name'][0])
                preds = (outputs.detach().cpu().permute(0, 2, 3, 1) * mask).permute(0, 3, 1, 2).max(dim=1)[1].numpy()
            else:
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            
#             t_size = (128, 128)
#             preds = Image.fromarray(preds)
#             preds = F.resize(preds, t_size, Image.BILINEAR)
#             preds = np.array(preds)
#             targets = Image.fromarray(targets)
#             targets = F.resize(targets, t_size, Image.NEAREST)
#             targets = np.array(targets)
            if "inv" in opts.task:
                targets = remap(targets)
            
            metrics.update(targets, preds)
            
            if opts.save_mask:
                task_postfix = ''
                if opts.task_incremental:
                    task_postfix = '_task_incre'
                if not os.path.isdir(f'{opts.save_mask_dir}/{opts.dataset}_masks_{opts.task}_{opts.ckpt_postfix}{task_postfix}'):
                    Path(f'{opts.save_mask_dir}/{opts.dataset}_masks_{opts.task}_{opts.ckpt_postfix}{task_postfix}').mkdir(parents=True, exist_ok=True)
                save_dir = f'{opts.save_mask_dir}/{opts.dataset}_masks_{opts.task}_{opts.ckpt_postfix}{task_postfix}'
                if opts.dataset == "livecell":
#                     save_dir = f'{opts.data_root}/livecelldataset_all/{opts.dataset}_masks_{opts.task}_{opts.ckpt_postfix}'
#                     if not os.path.isdir(save_dir):
#                         os.mkdir(save_dir)
                    for pred, file_name in zip(preds, details['file_name']):
                        img = Image.fromarray(np.uint8(pred))
                        file_name = file_name.replace('.tif', '_mask.tif')
                        img.save(f'{save_dir}/{file_name}')
                elif opts.dataset == "glas":
#                     save_dir = f'{opts.data_root}/glas/{opts.dataset}_masks_{opts.task}_{opts.ckpt_postfix}'
#                     if not os.path.isdir(save_dir):
#                         os.mkdir(save_dir)
                    for pred, file_name in zip(preds, details):
                        img = Image.fromarray(np.uint8(pred))
                        file_name = file_name+ '_mask.bmp'
                        img.save(f'{save_dir}/{file_name}')
            break   
            if opts.dataset == "glas":
                interval = 1
#             if (i + 1) % interval == 0:
#                 image = images[0].detach().cpu().numpy()
#                 save_img(image, details, opts, 0)
#                 save_img(preds[0], details, opts, 1)
#                 save_img(targets[0], details, opts, 2)
                
        score = metrics.get_results()
    return score


# -

def main(opts):
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    
    target_cls = get_tasks(opts.dataset, opts.task, opts.curr_step)
    if opts.dataset == "glas":
        opts.num_classes = [len(get_tasks(opts.dataset, opts.task, 0))]
    else:
        opts.num_classes = [len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1)]
    
    if opts.unknown: # [unknown, background, ...]
        opts.num_classes = [1, 1, opts.num_classes[0]-1] + opts.num_classes[1:]
    fg_idx = 1 if opts.unknown else 0
    
    curr_idx = [
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step)), 
        sum(len(get_tasks(opts.dataset, opts.task, step)) for step in range(opts.curr_step+1))
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("==============================================")
    print(f"  task : {opts.task}")
    print(f"  step : {opts.curr_step}")
    print("  Device: %s" % device)
    print( "  opts : ")
    print(opts)
    print("==============================================")

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    
    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride, bn_freeze=opts.bn_freeze)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
        
    # Set up metrics
    metrics = StreamSegMetrics(sum(opts.num_classes)-1 if opts.unknown else sum(opts.num_classes), opt=opts)

    if opts.overlap:
        ckpt_str = f"checkpoint/{opts.task}_{opts.ckpt_postfix}/%s_%s_%s_step_%d_overlap_%s.pth"
    else:
        ckpt_str = f"checkpoint/{opts.task}_{opts.ckpt_postfix}/%s_%s_%s_step_%d_disjoint_%s.pth"
    
    model = nn.DataParallel(model)
    mode = model.to(device)
    
    task, unknown = opts.task, opts.unknown
    opts.task, opts.unknown = 'offline', False
    dataset_dict = get_dataset(opts)
    opts.task, opts.unknown = task, unknown
    for split in dataset_dict:
        test_loader = data.DataLoader(
            dataset_dict[split], batch_size=opts.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)

        print("... Testing Best Model")
        report_dict = dict()
    #     opts.curr_step = 0
        best_ckpt = ckpt_str % (opts.model, opts.dataset, opts.task, opts.curr_step, opts.ckpt_postfix)
        if opts.ckpt:
            best_ckpt = opts.ckpt
        print(best_ckpt)

        checkpoint = torch.load(best_ckpt, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()

        test_score = validate(opts=opts, model=model, loader=test_loader, 
                              device=device, metrics=metrics)
        print(metrics.to_str(test_score))
        report_dict[f'best/test_all_miou'] = test_score['Mean IoU']

        class_iou = list(test_score['Class IoU'].values())
        class_acc = list(test_score['Class Acc'].values())

        first_cls = len(get_tasks(opts.dataset, opts.task, 0)) 

        report_dict[f'best/test_before_mIoU'] = np.mean(class_iou[:first_cls]) 
        report_dict[f'best/test_after_mIoU'] = np.mean(class_iou[first_cls:])  
        report_dict[f'best/test_before_acc'] = np.mean(class_acc[:first_cls])  
        report_dict[f'best/test_after_acc'] = np.mean(class_acc[first_cls:])  

        print(f"...from 0 to {first_cls-1} : best/test_before_mIoU : %.6f" % np.mean(class_iou[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_mIoU : %.6f" % np.mean(class_iou[first_cls:]))
        print(f"...from 0 to {first_cls-1} : best/test_before_acc : %.6f" % np.mean(class_acc[:first_cls]))
        print(f"...from {first_cls} to {len(class_iou)-1} best/test_after_acc : %.6f" % np.mean(class_acc[first_cls:]))


if __name__ == '__main__':
            
    opts = get_argparser().parse_args()
        
    total_step = len(get_tasks(opts.dataset, opts.task))
    opts.curr_step = total_step - 1
    main(opts)

