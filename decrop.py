#! /usr/bin/env python
from PIL import Image
import numpy as np
import argparse
import os
import math
from rich.progress import Progress

def merge_images(image_dir, width, height, out_path):
    # 确保 width 和 height 向上取整为 1024 的整数倍
    width = math.ceil(width / 1024) * 1024
    height = math.ceil(height / 1024) * 1024
    # 获取目录下所有 PNG 图像的文件名
    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')],
                       key=lambda x: int(os.path.splitext(x)[0]))  # 按自然数排序
    
    # img_files = img_files[:116]
    print(f'len of img files is {len(img_files)}')
    # 计算图像数量和每行的图像数
    num_images = len(img_files)
    num_images_per_row = width // 1024  # 每行多少张图像
    num_rows = (num_images // num_images_per_row) + (1 if num_images % num_images_per_row != 0 else 0)
    
    # 计算拼接后的图像高度
    final_height = min(num_rows * 1024, height)  # 如果拼接高度超过目标高度，则限制为目标高度
    final_image = np.zeros((final_height, width, 3), dtype=np.uint8)  # 初始化拼接后的图像

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing...", total=num_images)
    # 逐个加载图像并拼接
        for idx, img_file in enumerate(img_files):
            # 计算图像在大图中的位置
            row = idx // num_images_per_row
            col = idx % num_images_per_row
            img_path = os.path.join(image_dir, img_file)
            
            # 打开图像并转换为 NumPy 数组
            img = Image.open(img_path)
            
            img = np.array(img)
            
            # 计算该图像在大图中的位置并将其插入
            final_image[row * 1024 : (row + 1) * 1024, col * 1024 : (col + 1) * 1024] = img

            progress.update(task, advance=1)
        
    
    # 将拼接后的 NumPy 数组转换为图像并保存
    final_image = Image.fromarray(final_image)
    print('saving img...')
    final_image.save(out_path)
    print(f'Image saved to {out_path}')

def parse_args():
    # 使用 argparse 来处理命令行参数
    parser = argparse.ArgumentParser(description='Merge PNG images into a single large image')
    parser.add_argument('image_dir', type=str, help='Directory containing PNG images')
    parser.add_argument('width', type=int, help='Width of the final merged image')
    parser.add_argument('height', type=int, help='Height of the final merged image')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path to save the output image')

    return parser.parse_args()

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()

    # 调用合并图像的函数
    merge_images(args.image_dir, args.width, args.height, args.output)
