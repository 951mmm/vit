#! /usr/bin/env python
import argparse
from PIL import Image
import os

def crop_image(input_image_path, output_dir, width=1024, height=1024):
    # 打开大图
    img = Image.open(input_image_path)
    img_width, img_height = img.size
    
    # 计算横向和纵向可以裁剪的块数
    num_cols = img_width // width  # 横向的块数
    num_rows = img_height // height  # 纵向的块数
    
    # 如果大图的宽度或高度不能整除目标尺寸，需要调整
    if img_width % width != 0:
        num_cols += 1
    if img_height % height != 0:
        num_rows += 1

    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 按顺序裁剪并保存图像
    counter = 1
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * width
            top = row * height
            right = min(left + width, img_width)  # 确保不超出大图宽度
            bottom = min(top + height, img_height)  # 确保不超出大图高度
            
            # 裁剪当前区域
            cropped_img = img.crop((left, top, right, bottom))
            
            # 计算填充的位置
            pad_left = 0
            pad_top = 0
            # pad_right = max(0, width - cropped_img.width)  # 填充到右侧
            # pad_bottom = max(0, height - cropped_img.height)  # 填充到底部
            
            # 创建一个新的图像，背景为白色
            padded_img = Image.new("RGB", (width, height), (255, 255, 255))
            
            # 将裁剪后的图像粘贴到新的图像上，紧接着裁剪图像的右边界和下边界
            padded_img.paste(cropped_img, (pad_left, pad_top))
            
            # 保存裁剪后图像
            output_filename = os.path.join(output_dir, f"{counter}.png")
            padded_img.save(output_filename)
            print(f"Saved {output_filename}")
            
            counter += 1

def parse_args():
    # 使用 argparse 来处理命令行参数
    parser = argparse.ArgumentParser(description='Crop a large image into smaller chunks and pad with white')
    parser.add_argument('input_image', type=str, help='Path to the input image')
    parser.add_argument('output_dir', type=str, help='Directory to save the cropped images')
    parser.add_argument('--width', type=int, default=1024, help='Width of each cropped image (default 1024)')
    parser.add_argument('--height', type=int, default=1024, help='Height of each cropped image (default 1024)')

    return parser.parse_args()

if __name__ == '__main__':

    Image.MAX_IMAGE_PIXELS = None
    # 解析命令行参数
    args = parse_args()

    # 调用裁剪函数
    crop_image(args.input_image, args.output_dir, args.width, args.height)
