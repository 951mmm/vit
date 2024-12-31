# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import os
import mmcv
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


def main():
    parser = ArgumentParser()
    parser.add_argument('imgs', help='Image dir')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('-o', '--out-dir', default='out', help='Path to output dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    imgs = [os.path.join(args.imgs, img) for img in os.listdir(args.imgs)] 
    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    print()
    
    # test a single image
    batch_size = 20
    batch_cnt = len(imgs) // batch_size
    batch_rst = len(imgs) - batch_size * batch_cnt
    
    print('inference results...')
    prog_bar = mmcv.ProgressBar(batch_cnt if batch_rst == 0 else batch_cnt + 1)
    results = []
    for i in range(0, batch_cnt):
        results.extend(inference_segmentor(model, imgs[i * batch_size:i * batch_size + 20]))
        prog_bar.update()

    if batch_rst != 0:
        results.extend(inference_segmentor(model, imgs[batch_cnt * batch_size:len(imgs)]))
        prog_bar.update()
    print()

    # show the results
#    show_result_pyplot(
#        model,
#        args.img,
#        result,
#        get_palette(args.palette),
#        opacity=args.opacity,
#        out_file=args.out_file)
    print('mix result and img...')
    prog_bar = mmcv.ProgressBar(len(imgs))
    mixed_imgs = []
    if hasattr(model, 'module'):
        model = model.module

    for img, result in zip(imgs, results):
        mixed_imgs.append(model.show_result(img, [result], palette=get_palette(args.palette), show=False, opacity=args.opacity))
        prog_bar.update()
    # mixed_imgs = [model.show_result(img, result, palette=palette, show=False, opacity=opacity) for img, result in zip(imgs, results)]
    print()
    
    print('save mixed imgs...')
    prog_bar = mmcv.ProgressBar(len(mixed_imgs))
    if args.out_dir is not None:
        for mixed_img, img in zip(mixed_imgs, imgs):
            mmcv.imwrite(mixed_img, os.path.join(args.out_dir, os.path.basename(img)))
      
    print()
       # [mmcv.imwrite(mixed_img, os.path.join(args.out_dir, os.path.basename(img)) for mixed_img, img in zip(mixed_imgs, imgs)]


if __name__ == '__main__':
    main()
