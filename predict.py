# ------------------------------------------------------------------------
# HOTR official code : main.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import argparse
import itertools
import cv2
from PIL import Image
import numpy as np
import torch
import os

import hotr.data.datasets as datasets
import hotr.data.transforms.transforms as T
import hotr.util.misc as utils
from hotr.engine.arg_parser import get_args_parser
from hotr.models import build_model

def main(args):
    device = torch.device(args.device)

    # Data Setup
    meta = datasets.builtin_meta._get_coco_instances_meta()
    COCO_CLASSES = meta['coco_classes']
    args.num_classes = len(COCO_CLASSES)
    _valid_obj_ids = [id for id in meta['thing_dataset_id_to_contiguous_id'].keys()]
    with open(args.action_list_file, 'r') as f:
        action_lines = f.readlines()
    _valid_verb_ids, _valid_verb_names = [], []
    for action_line in action_lines[2:]:
        act_id, act_name = action_line.split()
        _valid_verb_ids.append(int(act_id))
        _valid_verb_names.append(act_name)
    args.num_actions = len(_valid_verb_ids)
    args.action_names = _valid_verb_names
    if args.share_enc: args.hoi_enc_layers = args.enc_layers
    if args.pretrained_dec: args.hoi_dec_layers = args.dec_layers

    args.valid_obj_ids = _valid_obj_ids
    correct_mat = np.load(args.correct_path)
    print(args)

    # Model Setup
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

    if args.eval:
        # test only mode
        img_ori = cv2.imread(args.img_dir)
        img = Image.open(args.img_dir).convert('RGB')
        w, h = img.size
        orig_size = torch.as_tensor([h, w]).unsqueeze(0).to(device)
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])
        img, _ = transforms(img, None)
        batch = utils.collate_fn([(img, None)])
        # returns predictions instead of test status
        model.eval()
        preds = []
        hoi_recognition_time = []

        samples = batch[0]
        samples = samples.to(device)
        # targets = [{k: (v.to(device) if k != 'id' else v) f or k, v in t.items()} for t in targets]

        outputs = model(samples)
        # import pdb; pdb.set_trace()
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_size, threshold=0, dataset='hico-det')
        hoi_recognition_time.append(results[0]['hoi_recognition_time'] * 1000)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))

        preds = [img_preds for i, img_preds in enumerate(preds)]
        max_hois = 100
        conf_thres = 0.33
        for img_preds in preds:
            img_preds = {k: v.to('cpu').numpy() for k, v in img_preds.items() if k != 'hoi_recognition_time'}
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(img_preds['boxes'], img_preds['labels'])]
            hoi_scores = img_preds['verb_scores']
            verb_labels = np.tile(np.arange(hoi_scores.shape[1]), (hoi_scores.shape[0], 1))
            subject_ids = np.tile(img_preds['sub_ids'], (hoi_scores.shape[1], 1)).T
            object_ids = np.tile(img_preds['obj_ids'], (hoi_scores.shape[1], 1)).T

            hoi_scores = hoi_scores.ravel()
            verb_labels = verb_labels.ravel()
            subject_ids = subject_ids.ravel()
            object_ids = object_ids.ravel()

            if len(subject_ids) > 0:
                object_labels = np.array([bboxes[object_id]['category_id'] for object_id in object_ids])
                masks = correct_mat[verb_labels, object_labels]
                hoi_scores *= masks

                hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                        subject_id, object_id, category_id, score in zip(subject_ids, object_ids, verb_labels, hoi_scores) if score>conf_thres]
                hois.sort(key=lambda k: (k.get('score', 0)), reverse=True)
                hois = hois[:max_hois]
            else:
                hois = []
            
            img_basename = os.path.basename(args.img_dir)[:-4]
            for i, hoi in enumerate(hois):
                img_save = img_ori.copy()
                sub_id = hoi['subject_id']
                obj_id = hoi['object_id']
                verb = hoi['category_id']
                verb_name = args.action_names[verb]
                verb_name_proper = verb_name.replace('_', ' ')
                sub_box = bboxes[sub_id]['bbox'].astype(np.int32)
                obj_box = bboxes[obj_id]['bbox'].astype(np.int32)
                obj_cls = bboxes[obj_id]['category_id']
                score = hoi['score']
                obj_name = COCO_CLASSES[args.valid_obj_ids[obj_cls]]
                
                # save results
                cv2.rectangle(img_save, (sub_box[0], sub_box[1]), (sub_box[2], sub_box[3]), color=(255, 0, 0))
                cv2.rectangle(img_save, (obj_box[0], obj_box[1]), (obj_box[2], obj_box[3]), color=(0, 0, 255))
                cv2.putText(img_save, f'{verb_name_proper} {obj_name}', (sub_box[0], sub_box[1]-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
                cv2.imwrite(f'./output/{img_basename}_{i}_{verb_name}_{obj_name}.jpg', img_save)
            
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'End-to-End Human Object Interaction training and evaluation script',
        parents=[get_args_parser()]
    )
    parser.add_argument('--action_list_file', default='data/hico_20160224_det/action_list.txt', type=str)
    parser.add_argument('--correct_path', default='data/hico_20160224_det/corre_hico.npy', type=str)
    parser.add_argument('--img_dir', default='test.jpg', type=str, help='image to inference')
    args = parser.parse_args()
    # import glob
    # for d in sorted(glob.glob('/nfs/project/wangyuanbin_i/hotr/hico_20160224_det/images/test2015/HICO_test2015_000000*.jpg')):
    #     args.img_dir = d
    main(args)
