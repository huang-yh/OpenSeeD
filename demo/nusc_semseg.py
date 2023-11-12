# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import sys
import logging

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)

import numpy as np
np.random.seed(1)

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from utils.arguments import load_opt_command

from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from openseed.BaseModel import BaseModel
from openseed import build_model
from utils.visualizer import Visualizer
from datasets.nuscenes import nuScenesDataset, \
    custom_collate_fn


logger = logging.getLogger(__name__)

def pass_print(*args, **kwargs):
    pass

def main(local_rank, args=None):
    '''
    Main execution point for PyLearn.
    '''
    opt, cmdline_args = load_opt_command(args)
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['user_dir'] = absolute_user_dir

    gpus = torch.cuda.device_count()
    if gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20506")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1

    # META DATA
    pretrained_pth = os.path.join(opt['WEIGHT'])
    output_root = './output'
    vis_output_root = './output/vis'

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()
    dataset = nuScenesDataset(
        imageset='/data1/code/hyh/github/SelfOcc/data/nuscenes_infos_val_temporal_v1.pkl')
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=1,
        pin_memory=True,
        collate_fn=custom_collate_fn)


    # stuff_classes = ['zebra','antelope','giraffe','ostrich','sky','water','grass','sand','tree']
    stuff_classes = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation']
    stuff_classes = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'person', 'traffic_cone', 'trailer', 'truck',
         'road', 'other_flat', 'sidewalk', 'terrain', 'building',
         'tree', 'sky']
    stuff_classes = [
        'barrier',
        'bicycle',
        'bus',
        'car', # 'wagon', 'van', 'minivan', 'SUV', 'jeep',
        'construction_vehicle', 'crane',
        'motorcycle', # 'vespa', 'Scooter',
        'person', 
        'traffic_cone',
        'trailer', 'trailer_truck',
        'truck',
        'road', 
        'other_flat', 'rail track', 'lake', 'river',
        'sidewalk', 
        'terrain', 'grass', 'hill', # 'sand', 'gravel',
        'building', 'wall', # 'guard rail', 'fence', # 'pole',
        'tree', # 'plant', 
        'sky',
    ]
    stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int).tolist() for _ in range(len(stuff_classes))]
    stuff_dataset_id_to_contiguous_id = {x:x for x in range(len(stuff_classes))}

    MetadataCatalog.get("demo").set(
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(stuff_classes, is_eval=True)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(stuff_classes)

    with torch.no_grad():
        for i_iter, (images, ori_images, img_metas) in enumerate(loader):
            images = images.cuda()
            # image_ori = Image.open(image_pth).convert("RGB")
            # width = image_ori.size[0]
            # height = image_ori.size[1]
            # image = transform(image_ori)
            # image = np.asarray(image)
            # image_ori = np.asarray(image_ori)
            # images = torch.from_numpy(image.copy()).permute(2,0,1).cuda()
            token = img_metas[0]['token']
            for i_cam, image in enumerate(images[0]):
                image_save_dir = os.path.join(
                    output_root, img_metas[0]['cam_types'][i_cam])
                vis_save_dir = os.path.join(
                    vis_output_root, img_metas[0]['cam_types'][i_cam])
                os.makedirs(image_save_dir, exist_ok=True)
                os.makedirs(vis_save_dir, exist_ok=True)

                batch_inputs = [{'image': image, 'height': 900, 'width': 1600}]
                outputs = model.forward(batch_inputs, inference_task="sem_seg")
                sem_seg = outputs[-1]['sem_seg'].max(0)[1]

                if i_iter % 200 == 0:
                    visual = Visualizer(ori_images[0][i_cam], metadata=metadata)
                    demo = visual.draw_sem_seg(sem_seg.cpu(), alpha=0.5) # rgb Image
                    demo.save(os.path.join(vis_save_dir, token + '.png'))
                sem_seg = sem_seg.cpu().numpy().astype(np.uint8)
                np.save(os.path.join(image_save_dir, token+'.npy'), sem_seg)

if __name__ == "__main__":
    # main()

    ngpus = torch.cuda.device_count()

    if ngpus > 1:
        torch.multiprocessing.spawn(main, nprocs=ngpus)
    else:
        main(0)
