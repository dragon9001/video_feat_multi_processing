import torch.multiprocessing as mp
import pickle
import os
import sys
import re
import json
import subprocess
import numpy as np
import torch
from time import time, sleep
from torch import nn
import argparse

from model import generate_model
from mean import get_mean
from classify2 import classify_video
from torch.multiprocessing import Process, Queue, set_start_method

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input', type=str, help='Input file path')
    parser.add_argument('--video_root', default='', type=str, help='Root path of input videos')
    parser.add_argument('--model', default='', type=str, help='Model file path')
    parser.add_argument('--output_root', default='./output', type=str, help='Output file path')
    parser.add_argument('--mode', default='feature', type=str,
                        help='Mode (score | feature). score outputs class scores. feature outputs features (after global average pooling).')

    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=8, type=int,
                        help='Number of threads for multi-thread loading')
    parser.add_argument('--model_name', default='resnext', type=str,
                        help='Currently only support resnet')
    parser.add_argument('--model_depth', default=101, type=int,
                        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str,
                        help='Shortcut type of resnet (A | B)')
    parser.add_argument('--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument('--resnext_cardinality', default=32, type=int, help='ResNeXt cardinality')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    return args

# python extract_feat2.py --input ./input --video_root ./videos --output ./output --model ./PretrainedModels/resnext-101-kinetics.pth --resnet_shortcut B


class Video2frame_worker():
    def __init__(self, class_names, queue, opt):
        self.class_names = class_names
        self.queue = queue
        self.process = Process(target=self.ffmpeg, args=(queue, opt))

    def ffmpeg(self, queue, opt):
        for cls_name in self.class_names:
            input_fns = os.listdir(os.path.join(opt.video_root, cls_name))

            if not os.path.exists(os.path.join(opt.output_root, cls_name)):
                os.mkdir(os.path.join(opt.output_root, cls_name))

            outputs = []
            for input_fn in input_fns:
                input_fn = os.path.join(cls_name, input_fn)
                # tmp_dir = 'tmp_' + input_fn.split('/')[-1]
                tmp_dir = os.path.join('tmp', input_fn)
                if os.path.exists(tmp_dir):
                    cmd = 'rm -rf ' + tmp_dir
                    subprocess.call(cmd, shell=True)

                video_path = os.path.join(opt.video_root, input_fn)
                if os.path.exists(video_path):
                    # 删除残余文件
                    if os.path.exists(tmp_dir):
                        subprocess.call('rm -rf ' + tmp_dir, shell=True)

                    # FFMPEG处理
                    t1 = time()

                    # cmd需要特殊处理' '
                    tmp_dir_cmd = tmp_dir.replace(' ', '\ ')
                    video_path = video_path.replace(' ', '\ ')
                    subprocess.call('mkdir -p ' + tmp_dir_cmd, shell=True)

                    cmd = 'ffmpeg -v 0 -i {} '.format(video_path) + tmp_dir_cmd + '/image_%05d.jpg'
                    print(cmd)
                    subprocess.call(cmd, shell=True)
                    print('FFMPEG processed for', time() - t1, 'seconds')

                    # 放入待处理队列
                    queue.put(tmp_dir)
                else:
                    print('{} does not exist'.format(video_path))


class Extractor():
    def __init__(self, opt, queue, device):
        self.opt = opt
        self.process = Process(target=self.Extract, args=(opt, queue, device))

    def Extract(self, opt, queue, device):
        model = generate_model(opt)
        print('loading model {}'.format(opt.model))

        # 转换model位置
        device_id = int(device.split(':')[-1])
        model = torch.nn.DataParallel(model, device_ids=[device_id])
        model_data = torch.load(opt.model, map_location={'cuda:0': device})

        assert opt.arch == model_data['arch']
        model.load_state_dict(model_data['state_dict'])
        model.to(device)
        model.eval()
        model.share_memory()

        while True:
            tmp_dir = queue.get()
            # 以下2行，还原原始文件名。
            input_fn = tmp_dir.split('_')[1:]
            input_fn = '_'.join(input_fn)
            # 以下2行，还原cls_name
            print('1', tmp_dir)
            match = re.search('/([\s\S]+)/', tmp_dir)
            cls_name = match[1]

            output_fn = opt.output_root + '/' + cls_name + '/' + input_fn + '.pickle'
            print('Extracting', tmp_dir, 'to', output_fn)
            # result is a dict
            if not os.path.exists(output_fn):

                result = classify_video(tmp_dir, input_fn, model, opt)

                with open(output_fn, 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
            else:
                print('feature already exists:', output_fn)

            if os.path.exists(tmp_dir):
                subprocess.call('rm -rf ' + tmp_dir.replace(' ', '\ '), shell=True)


if __name__ == "__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 400

    # FFmpeg processed queue. a list of strings.
    queue = Queue()
    # Testing Code
    class_names = os.listdir(opt.video_root)

    t1 = len(class_names)//4
    t2 = len(class_names)//2
    t3 = len(class_names)//4 * 3
    t4 = len(class_names)

    class_names1 = class_names[:t1]
    class_names2 = class_names[t1:t2]
    class_names3 = class_names[t2:t3]
    class_names4 = class_names[t3:t4]

    ff1 = Video2frame_worker(class_names1, queue, opt)
    ff2 = Video2frame_worker(class_names2, queue, opt)
    ff3 = Video2frame_worker(class_names3, queue, opt)
    ff4 = Video2frame_worker(class_names4, queue, opt)

    ext4 = Extractor(opt, queue, device='cuda:3')
    ext3 = Extractor(opt, queue, device='cuda:2')
    ext2 = Extractor(opt, queue, device='cuda:1')
    ext1 = Extractor(opt, queue, device='cuda:0')

    ff1.process.start()
    ff2.process.start()
    ff3.process.start()
    ff4.process.start()

    ext4.process.start()
    ext3.process.start()
    ext2.process.start()
    ext1.process.start()

    ff1.process.join()
    ff2.process.join()
    ff3.process.join()
    ff4.process.join()

    ext4.process.join()
    ext3.process.join()
    ext2.process.join()
    ext1.process.join()
