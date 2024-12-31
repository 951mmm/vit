#! /usr/bin/env python
import os
from typing import Dict, List
import numpy as np
import random
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
from mmpretrain.apis import ImageClassificationInferencer

## util
def generate_colors(num_classes):
    colors = []
    random.seed(42)
    for _ in range(num_classes):
        color = [random.randint(0, 255) for _ in range(3)]
        colors.append(color)
    return colors

## index
def seg_infer(in_dir: str, out_dir: str):
    SAM_TYPE = 'vit_l'
    SAM_CHECKPOINT = './sam_vit_l_0b3195.pth'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sam = sam_model_registry[SAM_TYPE](SAM_CHECKPOINT).to(DEVICE)
    mask_generator = SamAutomaticMaskGenerator(model=sam, min_mask_region_area=100)
    
    CONFIG = 'configs/rsmamba/rsmamba_uc_b.py'
    MODEL_CHECKPOINT = 'work_dirs/rsmamba_uc_b/best_single-label_f1-score_epoch_400.pth'
    inferencer = ImageClassificationInferencer(CONFIG, pretrained=MODEL_CHECKPOINT, device=DEVICE)

    NUM_CLASSES = 21
    color_map = generate_colors(NUM_CLASSES)

    BLEND = True
    OPACITY = 0.4
    
    BATCH_SIZE = 50

    def label_infer(img: np.ndarray):
        results = inferencer(img)
        result = results[0]
        return result['pred_label']
    
    def labels_infer(imgs: List[np.ndarray]):
        results = inferencer(imgs)
        results = [result['pred_label'] for result in results]
        return results

    def masks_ann(masks: List[Dict[str, any]], img: np.ndarray):
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        h, w, _ = img.shape
        label_mask = np.zeros((h, w))
        cropped_imgs = []
        for mask in sorted_masks:
            bbox = mask['bbox']
            left = bbox[0]
            top = bbox[1]
            right = bbox[0] + bbox[2]
            bottom = bbox[1] + bbox[3]
            cropped_img = img[top:bottom, left:right]
            cropped_imgs.append(cropped_img)
            # label = infer_label(cropped_img)
            
            # segmentation = mask['segmentation']
            # label_mask[segmentation] = label 
        labels = labels_infer(cropped_imgs)
        for label, mask in zip(labels, sorted_masks):
            segmentation = mask['segmentation']
            label_mask[segmentation] = label
        
        rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(NUM_CLASSES):
            rgb_mask[label_mask == i] = color_map[i]
        
        return rgb_mask

    def img_blend(ann_mask: np.ndarray, img: np.ndarray):
        blended_img = (OPACITY * ann_mask + (1 - OPACITY) * img).astype(np.uint8)
        return blended_img

    def img_seg(img_file: str):
        img = Image.open(os.path.join(in_dir, img_file))
        img = np.array(img)
        masks = mask_generator.generate(img)
        ann_mask = masks_ann(masks, img)
        result = ann_mask if BLEND is False else img_blend(ann_mask, img)
        result = Image.fromarray(result)
        result.save(os.path.join(out_dir, img_file))

       
    img_files = [img for img in os.listdir(in_dir)]    
    os.makedirs(out_dir, exist_ok=True)
    for img_file in img_files:
        img_seg(img_file)
    
seg_infer('./cropped', './annd')
