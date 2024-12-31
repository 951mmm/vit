#! /usr/bin/env python
from typing import List
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
import cv2
import numpy as np
import argparse
from rich.progress import Progress
from rich.console import Console

def imgs_read(img_dir: str, img_files: List[str], progress: Progress):
    task = progress.add_task('read imgs...', total=len(img_files))   
    imgs = []
    for img_file in img_files:
        imgs.append(cv2.imread(os.path.join(img_dir, img_file)))
        progress.update(task, advance=1)
    progress.stop_task(task)
    # imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    return imgs, img_files

def imgs_mask_gen(imgs: List[np.ndarray], sam_mask_gen: SamAutomaticMaskGenerator, progress: Progress):
    task = progress.add_task('generate mask...', total=len(imgs)) 
    masks = []
    for img in imgs:
        masks.append(sam_mask_gen.generate(img))
        progress.update(task, advance=1)
    progress.stop_task(task)
    return masks

def masks_process(masks: List[list], threshold_ratio: float):
    def mask_process_sub(sub_masks: list):
        segmentations = []
        for sub_mask in sub_masks:
            segmentation: np.ndarray = sub_mask['segmentation']
            true_count = np.sum(segmentation)
            total_count = segmentation.size
            true_ratio = true_count / total_count
            if threshold_ratio > true_ratio:
                continue
            segmentations.append(segmentation)
        return segmentations
    
    segmentations = [mask_process_sub(sub_masks) for sub_masks in masks]
    return segmentations
        
def img_crop(img: np.ndarray, segmentation: np.ndarray, fullfill=99):
    cropped_img = np.full_like(img, fullfill)
    cropped_img[segmentation] = img[segmentation]
    return cropped_img

def imgs_crop(imgs: List[np.ndarray], segmentations: List[List[np.ndarray]]):
    def img_crop_sub(img, sub_segmentations):
        cropped_imgs = []

        for segmentation in sub_segmentations:
            cropped_img = img_crop(img, segmentation)
            cropped_imgs.append(cropped_img)
            
        return cropped_imgs
    
    cropped_imgs = [img_crop_sub(img, sub_segmentations) for img, sub_segmentations in zip(imgs, segmentations)]
    return cropped_imgs

def img_crop_none_zero(cropped_img: np.ndarray):
    mask = np.all(cropped_img != [0, 0, 0], axis=-1)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    
    cropped_none_zero_img = cropped_img[top:bottom + 1, left:right + 1]
    return cropped_none_zero_img

CLASS_TABLE = [
    "IGNORE",
    "BACKGROUND",
    "BUILDING",
    "ROAD",
    "WATER",
    "BARREN",
    "FOREST",
    "AGRICULTURAL"
]
def imgs_classify(imgs: List[np.ndarray], label_imgs: List[np.ndarray], num_classes: int, segmentations: List[List[np.ndarray]]):
    classified_map = {}
    def img_classify_sub(img, label_img, sub_segmentations):
        for segmentation in sub_segmentations:
            cropped_label_img = img_crop(label_img, segmentation, num_classes)
            cropped_img = img_crop(img, segmentation, 0)
            unique_elements, counts = np.unique(cropped_label_img, return_counts=True)
            most_common_class = unique_elements[np.argmax(counts)]
            if most_common_class == num_classes:
                most_common_class = unique_elements[np.argsort(counts)[-2]]
            
            most_common_class = CLASS_TABLE[most_common_class] 
            cropped_none_zero_img = img_crop_none_zero(cropped_img)
            if most_common_class not in classified_map:
                classified_map[most_common_class] = [cropped_none_zero_img]
            else:
                classified_map[most_common_class].append(cropped_none_zero_img)

    for img, label_img, sub_segmentations in zip(imgs, label_imgs, segmentations):
        img_classify_sub(img, label_img, sub_segmentations)
    
    return classified_map

def img_write(out_dir: str, img: np.ndarray, basename: str):
    cv2.imwrite(os.path.join(out_dir, basename), img)

def classified_map_write(out_dir: str, classified_map: dict, batch: int, progress: Progress):
    task = progress.add_task('write classified cropped imgs...', total=len(sum(classified_map.values(), [])))
    for cls, imgs in classified_map.items():
        cls_dir = os.path.join(out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for i, img in enumerate(imgs):
            img_write(cls_dir, img, f'{cls}_{batch}_{i}.png')
            progress.update(task, advance=1)
            
        progress.stop_task(task)
        

def imgs_write(out_dir: str, imgs: List[List[np.ndarray]], basenames: List[str]):
    for sub_imgs, basename in zip(imgs, basenames):
        out_file_name = '.'.join(basename.split('.')[:-1])
        out_file_ext = basename.split('.')[-1] 

        for i, sub_img in enumerate(sub_imgs):
            out_file = f'{out_file_name}_{i}.{out_file_ext}'
            print(f'out file name is: {out_file}')
            cv2.imwrite(os.path.join(out_dir, out_file), sub_img)
            
def transfer_batch(img_dir: str, label_dir: str, out_dir: str, sam_checkpoint: str, sam_type: str, device: str, batch_size: int, threshold_ratio: float):
    sam = sam_model_registry[sam_type](sam_checkpoint)
    sam.to(device)
    sam_mask_gen = SamAutomaticMaskGenerator(sam)
    img_files = [img_file for img_file in os.listdir(img_dir) if img_file.endswith('.png')]
    img_files = sorted(img_files, key=lambda x:  int(x.split('.')[0]))
    
    batch_num = len(img_files) // batch_size
    batch_rst = len(img_files) - batch_num * batch_size
    
    with Progress() as progress:
        def transfer(img_files: List[str], batch):
            imgs, _ = imgs_read(img_dir, img_files, progress)

            masks = imgs_mask_gen(imgs, sam_mask_gen, progress)

            print('process masks...')
            segmentations = masks_process(masks, threshold_ratio)

            print('read label imgs...')
            label_imgs, _ = imgs_read(label_dir, img_files, progress)

            print('classify imgs...')
            classified_map = imgs_classify(imgs, label_imgs, 10, segmentations)

            os.makedirs(out_dir, exist_ok=True)
            classified_map_write(out_dir, classified_map, batch, progress)
            
        
        task = progress.add_task('transfer img batches...', total=batch_num if batch_rst == 0 else batch_num + 1)
        
        for batch in range(batch_num):
            transfer(img_files[batch * batch_size:(batch + 1) * batch_size], batch)
            progress.update(task, advance=1)
        if batch_rst != 0:
            transfer(img_files[batch_num * batch_size:], batch_num)
            progress.update(task, advance=1)
            
    
parser = argparse.ArgumentParser(description='transfer data for swinformer')
parser.add_argument('img_dir', type=str, help='image dir')
parser.add_argument('label_dir', type=str, help='label image dir')
parser.add_argument('--out-dir', type=str, default='out', help='output dir')
parser.add_argument('--sam-checkpoint', type=str, default='./sam_vit_l_0b3195.pth', help='sam model checkkpoint path')
parser.add_argument('--sam-type', type=str, default='vit_l', help='sam model type')
parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
parser.add_argument('--batch-size', type=int, default=6, help='batch size to sync with disk, prevent out of mem')
parser.add_argument('--threshold-ratio', type=float, default=0.1, help='threshold prevent seg too small')

args = parser.parse_args()
transfer_batch(args.img_dir, args.label_dir, args.out_dir, args.sam_checkpoint, args.sam_type, args.device, args.batch_size, args.threshold_ratio)